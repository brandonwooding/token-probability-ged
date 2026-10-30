"""Standalone probe: does unigram normalisation separate i from c any better?

Reads archived runs and frequency tables, writes nothing, imports nothing from
standard-eval except the shape helper. Modifies no pipeline. The point is to
find out whether the correction is worth implementing properly before any of
adapt_thresh is touched.

Score under test, per model sub-token:

    s(t) = log p(t | context)  -  alpha * log p_unigram(t)

alpha=0 is the current system exactly (min-pooled probability, since log is
monotone so min log p picks the same sub-token as min p). alpha=1 is
pointwise mutual information, i.e. the token-level form of SLOR. Sub-token
scores are min-pooled per dataset word, matching adapt_thresh, and the
resulting word scores are evaluated against the gold c/i labels.

OOV sub-tokens (count 0) take the mean log unigram frequency of the sub-tokens
actually scored in that run, so their correction is zero and the decision falls
back to p(t|context) alone. That policy is settled by measurement: OOV
sub-tokens on the learner dev sets are 4.5x enriched for gold 'i' -- they are
misspellings and L1-transfer forms -- so add-one smoothing, which would hand
them the largest boost in the table, points the wrong way.

Two metrics, both computed identically at every alpha so the comparison is fair:

  AUC     threshold-free, so it measures separation itself rather than the
          quality of a threshold search. This is the primary number.
  F0.5    best achievable on the set, by exact sweep over every observed score
          rather than a fixed grid, so grid resolution cannot confound the
          comparison. Optimistic in absolute terms -- it is an oracle operating
          point -- but the optimism is identical across alpha.

Note the raw and standardised parameterisations are the same model: with
s_std = z(log p) - a' z(log f), we get s_std = (1/sigma_p)[log p - a'
(sigma_p/sigma_f) log f] + const, a positive affine map of s(t), so AUC and
best-F0.5 are identical at alpha = a' sigma_p / sigma_f. Only one needs
sweeping; the equivalent a' is reported alongside for interpretability.

    python slor_probe.py --model gemma-4-12B --dataset english/en_fce_dev
"""

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "standard-eval"))
import paths  # noqa: E402
from pooling import flatten  # noqa: E402

ALPHAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15,
          0.2, 0.3, 0.5, 0.75, 1.0]

# The agreed dial: what FRACTION of the probability signal's spread the
# frequency correction should carry. Raw alpha means different things in
# different conditions -- sd(log f)/sd(log p) ranges from 0.7 (no prompt) to
# 6.1 (variant_b) -- so a raw sweep silently tests 6x more correction in one
# condition than another. These are converted per run to alpha = a_std *
# sd_logp / sd_logfreq.
STD_ALPHAS = [0.0, 0.025, 0.05, 0.10, 0.125, 0.15, 0.20, 0.30, 0.50, 1.0]


def load_counts(model_id, language):
    path = paths.FREQ_TABLES / f"{model_id}__{language}.freqs.json.gz"
    if not path.exists():
        raise SystemExit(f"no frequency table at {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        table = json.load(f)
    return table["counts"], table["total_tokens"], table["vocab_hash"]


def auc(pos, neg):
    """P(pos ranks below neg) with ties at half credit; positives are gold 'i'.

    Rank-based rather than a threshold sweep: it is exact, handles the ties the
    forward-fill can introduce, and is what makes this number independent of any
    operating point.
    """
    merged = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum = 0.0
    i = 0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        rank_sum += sum(avg_rank for _, lab in merged[i:j] if lab == 1)
        i = j
    n_p, n_n = len(pos), len(neg)
    u = rank_sum - n_p * (n_p + 1) / 2.0
    return 1.0 - u / (n_p * n_n)      # low score => 'i', so invert


def best_f05(pos, neg):
    """Max incorrect-class F0.5 over an exact sweep of every observed score."""
    events = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    n_p = len(pos)
    tp = fp = 0
    best = (0.0, None)
    i = 0
    while i < len(events):
        j = i
        while j < len(events) and events[j][0] == events[i][0]:
            j += 1
        for _, lab in events[i:j]:
            tp += lab
            fp += 1 - lab
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / n_p if n_p else 0.0
        f = (1.25 * prec * rec / (0.25 * prec + rec)) if (prec + rec) else 0.0
        if f > best[0]:
            best = (f, events[j - 1][0])
        i = j
    return best


def score_run(run_dir, gold, counts, total, alphas, pool_mode="pre"):
    """Per-word min-pooled scores at each alpha, plus diagnostics."""
    gold_units = {u["unit_id"]: u for u in gold["units"]}
    data = json.loads((run_dir / "data.json").read_text())
    log_total = math.log(total)

    # Pass 1: collect log p and log f per sub-token so the OOV substitute can be
    # the mean over what this run actually scored.
    rows = []          # (unit_id, word_id, log_p, log_f_or_None)
    for uid, sentence in enumerate(data):
        n_words = len(gold_units[uid]["tokens"])
        for sub in flatten(sentence):
            wid, tid = sub.get("word_id"), sub.get("token_id")
            if wid is None or tid is None or not (0 <= wid < n_words):
                continue
            lp = math.log(max(sub["probability"], 1e-300))
            c = counts[tid] if tid < len(counts) else 0
            rows.append((uid, wid, lp, math.log(c) - log_total if c else None))

    known = [lf for *_, lf in rows if lf is not None]
    mean_lf = statistics.fmean(known)
    n_oov = sum(1 for *_, lf in rows if lf is None)

    lps = [lp for _, _, lp, _ in rows]
    lfs = [lf if lf is not None else mean_lf for *_, lf in rows]
    sd_p, sd_f = statistics.pstdev(lps), statistics.pstdev(lfs)
    # Pearson r between the two terms: how much of the probability signal the
    # correction is subtracting rather than adding to.
    mp, mf = statistics.fmean(lps), statistics.fmean(lfs)
    cov = sum((a - mp) * (b - mf) for a, b in zip(lps, lfs)) / len(lps)
    r = cov / (sd_p * sd_f) if sd_p and sd_f else float("nan")

    out = {}
    for alpha in alphas:
        pooled = {}
        for (uid, wid, lp, _), lf_filled in zip(rows, lfs):
            key = (uid, wid)
            if pool_mode == "pre":
                # Correct each sub-token, then take the minimum. The correction
                # can change WHICH sub-token wins the pool.
                s = lp - alpha * lf_filled
                if key not in pooled or s < pooled[key]:
                    pooled[key] = s
            else:
                # Pick the most surprising sub-token on raw probability -- the
                # same piece the current system picks -- then discount that
                # piece by its own rarity. Closer to the stated intuition,
                # which is about the word being judged rather than about which
                # piece represents it.
                if key not in pooled or lp < pooled[key][0]:
                    pooled[key] = (lp, lf_filled)
        if pool_mode != "pre":
            pooled = {k: lp - alpha * lf for k, (lp, lf) in pooled.items()}
        pos, neg = [], []
        for (uid, wid), s in pooled.items():
            (pos if gold_units[uid]["labels"][wid] == "i" else neg).append(s)
        out[alpha] = (pos, neg)
    return out, {"n_subtokens": len(rows), "n_oov": n_oov, "mean_logfreq": mean_lf,
                 "sd_logp": sd_p, "sd_logfreq": sd_f, "pearson_r": r}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemma-4-12B")
    parser.add_argument("--dataset", default="english/en_fce_dev")
    parser.add_argument("--approach", default="decoder-thresh")
    parser.add_argument("--pool", choices=("pre", "post"), default="pre",
                        help="correct before or after min-pooling")
    args = parser.parse_args()

    language = paths.language(args.dataset)
    stem = paths.stem(args.dataset)
    gold = json.loads(paths.gold(args.dataset).read_text())
    counts, total, vhash = load_counts(args.model, language)

    # Newest run per template, base model only (no adapter in the folder name).
    runs = {}
    for meta_path in sorted((paths.APPROACH_DIRS[args.approach] / "outputs").glob(
            f"{stem}__{args.model}__*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("model_id") != args.model or meta.get("adapter"):
            continue
        runs[meta.get("template", "?")] = (meta_path.parent, meta)

    print(f"model={args.model}  dataset={args.dataset}  table_vocab_hash={vhash[:12]}")
    print(f"frequency corpus: {total/1e6:.0f}M tokens of {language}  pool={args.pool}\n")

    for template in ("none", "variant_a", "variant_b", "fewshot"):
        if template not in runs:
            continue
        run_dir, meta = runs[template]
        # Two passes: the first only to learn sd(log p) and sd(log f) for this
        # run, the second at the raw alphas those imply for each target share.
        _, diag = score_run(run_dir, gold, counts, total, [0.0], args.pool)
        ratio = diag["sd_logp"] / diag["sd_logfreq"]
        alphas = [a * ratio for a in STD_ALPHAS]
        scored, diag = score_run(run_dir, gold, counts, total, alphas, args.pool)
        base_auc, (base_f, _) = None, (None, None)
        print(f"--- {template}  ({run_dir.name}) ---")
        print(f"    {diag['n_subtokens']:,} sub-tokens, {diag['n_oov']} OOV "
              f"({diag['n_oov']/diag['n_subtokens']:.3%});  sd(log p)={diag['sd_logp']:.2f}  "
              f"sd(log f)={diag['sd_logfreq']:.2f}  r={diag['pearson_r']:+.3f}")
        print(f"    {'share':>7s} {'raw a':>7s} {'AUC':>8s} {'dAUC':>8s} "
              f"{'bestF0.5':>9s} {'dF0.5':>8s}")
        for a_std, alpha in zip(STD_ALPHAS, alphas):
            pos, neg = scored[alpha]
            a = auc(pos, neg)
            f, _thr = best_f05(pos, neg)
            if base_auc is None:
                base_auc, base_f = a, f
            print(f"    {a_std*100:6.1f}% {alpha:7.4f} {a:8.4f} {a-base_auc:+8.4f} "
                  f"{f*100:8.2f}% {(f-base_f)*100:+7.2f}")
        print()


if __name__ == "__main__":
    main()
