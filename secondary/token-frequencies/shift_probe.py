"""What the correction actually does to the scores, independent of whether it helps.

Two questions, neither about accuracy:

1. Does it move rare tokens up and common tokens down? In absolute terms every
   token moves UP, because -alpha*log p_uni is positive for every token. But the
   threshold is fitted, so a constant offset is invisible; what the detector sees
   is movement RELATIVE to the average token. Reported here re-centred on the
   token-weighted mean shift, which is what the decision boundary actually feels.

2. How many words does it push across the boundary, and were they words that
   should have moved? Each condition is given its own best-F0.5 threshold before
   and after, so this is the correction's effect at its own best operating point
   rather than at a boundary tuned for a different score.
"""

import gzip, json, math, statistics, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "standard-eval"))
import paths                              # noqa: E402
from pooling import flatten               # noqa: E402
from slor_probe import best_f05           # noqa: E402

SHARE = 0.125

CASES = [("english/en_fce_dev", "fewshot"), ("english/en_fce_dev", "none"),
         ("german/de_falko-merlin_dev", "variant_a")]
MODEL = "gemma-4-12B"

for dataset, template in CASES:
    language = paths.language(dataset); stem = paths.stem(dataset)
    run = next(d.parent for d in sorted(
        (paths.APPROACH_DIRS["decoder-thresh"] / "outputs").glob(f"{stem}__{MODEL}__*/meta.json"))
        if json.loads(d.read_text()).get("template") == template
        and not json.loads(d.read_text()).get("adapter"))
    with gzip.open(paths.FREQ_TABLES / f"{MODEL}__{language}.freqs.json.gz", "rt") as f:
        tbl = json.load(f)
    counts, log_total = tbl["counts"], math.log(tbl["total_tokens"])
    gold = {u["unit_id"]: u for u in json.loads(paths.gold(dataset).read_text())["units"]}

    rows = []
    for uid, sent in enumerate(json.loads((run / "data.json").read_text())):
        n_words = len(gold[uid]["tokens"])
        for sub in flatten(sent):
            wid, tid = sub.get("word_id"), sub.get("token_id")
            if wid is None or tid is None or not (0 <= wid < n_words): continue
            c = counts[tid] if tid < len(counts) else 0
            rows.append([uid, wid, math.log(max(sub["probability"], 1e-300)),
                         math.log(c) - log_total if c else None])
    mean_lf = statistics.fmean(r[3] for r in rows if r[3] is not None)
    for r in rows:
        if r[3] is None: r[3] = mean_lf
    sd_p = statistics.pstdev([r[2] for r in rows])
    sd_f = statistics.pstdev([r[3] for r in rows])
    alpha = SHARE * sd_p / sd_f

    print(f"\n=== {stem} / {template}   share={SHARE:.1%}  raw alpha={alpha:.4f} "
          f"(sd log p={sd_p:.2f}, sd log f={sd_f:.2f}) ===")

    # 1. relative shift by unigram-frequency decile
    lfs = sorted(r[3] for r in rows)
    cuts = [lfs[int(k / 10 * (len(lfs) - 1))] for k in range(1, 10)]
    mean_shift = alpha * statistics.fmean(r[3] for r in rows)   # what re-centring removes
    buckets = {}
    for _, _, _, lf in rows:
        d = sum(lf > c for c in cuts)
        buckets.setdefault(d, []).append(-alpha * lf - (-mean_shift))
    print("   decile   unigram freq       relative shift (nats)   as % of sd(log p)")
    for d in sorted(buckets):
        lo = -math.inf if d == 0 else cuts[d - 1]
        med = statistics.median(buckets[d])
        tag = "rarest" if d == 0 else ("commonest" if d == 9 else "")
        print(f"   {d+1:>4}   {math.exp(lo):>9.2e}+  {med:+9.3f}            "
              f"{med/sd_p*100:+6.1f}%   {tag}")

    # 2. label flips at each score's own best threshold
    def pooled(a):
        best = {}
        for uid, wid, lp, lf in rows:
            s = lp - a * lf
            k = (uid, wid)
            if k not in best or s < best[k]: best[k] = s
        return best
    p0, p1 = pooled(0.0), pooled(alpha)
    keys = sorted(p0)
    labs = [gold[u]["labels"][w] for u, w in keys]
    def thr(p):
        pos = [p[k] for k, l in zip(keys, labs) if l == "i"]
        neg = [p[k] for k, l in zip(keys, labs) if l == "c"]
        return best_f05(pos, neg)[1]
    t0, t1 = thr(p0), thr(p1)
    flips = {"c->i": [0, 0], "i->c": [0, 0]}     # [gold i, gold c]
    for k, lab in zip(keys, labs):
        a0, a1 = p0[k] < t0, p1[k] < t1
        if a0 == a1: continue
        flips["c->i" if a1 else "i->c"][0 if lab == "i" else 1] += 1
    n = len(keys)
    tot = sum(sum(v) for v in flips.values())
    good = flips["c->i"][0] + flips["i->c"][1]
    print(f"   words {n:,};  {tot} change prediction ({tot/n:.2%}) — "
          f"{good} toward gold, {tot-good} away  (net {good-(tot-good):+d})")
    for name, (gi, gc) in flips.items():
        print(f"      {name}: {gi+gc:>4}   gold i {gi:>4}  gold c {gc:>4}")
