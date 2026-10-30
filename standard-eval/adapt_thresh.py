"""Adapter: probability-threshold runs -> canonical predictions.

Shared by llm-probability-thresh (encoder) and decoder-probability-thresh
(all four prompt templates) -- both write the same per-token data.json with a
`word_id` and a `probability`, and both call a token "incorrect" when its
probability is low.

Pipeline (matches the approaches' own evaluate-probs.py):
  1. pool sub-token probabilities to word level by taking the MIN probability
     across the sub-tokens of each word;
  2. a word is predicted "i" when its pooled probability < threshold.

Threshold selection is kept separate from application, so the SAME code turns a
data.json into predictions whether we run it on an archived file today or on a
fresh GPU run later:
  --threshold 0.23   use a fixed value
  --threshold auto   pick the value that maximises incorrect-class F0.5. By
                     default that is fitted on the set being scored, which is an
                     optimistic operating point; with --tune-on it is fitted on
                     a different dataset and merely applied here. Either way the
                     choice and its provenance are recorded in meta.

`--tune-on` is the honest configuration: fit the threshold on synthetic C4_200M
and apply it to the real learner dev set, and no human-labelled data has been
used to pick the operating point. It requires a run of the SAME model (and, for
the decoder, the same prompt template) over the tuning dataset, because a
threshold is only meaningful against the probability distribution that produced
it -- standard-eval/run.py matches those up automatically.

A gold word can receive no sub-token of its own when the model's tokenizer
merged it into a neighbour (gold "!" "!" -> one "!!" sub-token; "cannot" kept
whole where gold split "can" "not"). Such a word is recovered rather than
dropped -- it inherits the merged piece's pooled probability from the preceding
word (see pool_word_probs). Only a genuinely out-of-range word_id, which cannot
be recovered this way, is skipped and counted in meta.
"""

import argparse
import sys
from pathlib import Path

from common import REPO_ROOT, load_json, save_json, prf

sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402

from pooling import flatten  # noqa: E402


def pool_word_probs(paragraph, n_tokens):
    """Min sub-token probability per word, recovering merged-away words.

    Accepts either data.json shape (see pooling.py); flatten() normalises it.

    The forward-fill below only ever fires for LEGACY flat runs. There, the
    aligner anchored a merged sub-token to the FIRST gold word it overlaps, so a
    swallowed word is always a later index covered by an earlier sub-token and
    inherits the nearest preceding word's pooled probability -- the merged
    piece's probability, shared across every gold word it spans (Lever A).
    Nested runs list a straddling sub-token under every word it overlaps, so no
    word is ever left empty and the recovery is dead code for them.

    Returns None only on a genuine misalignment (a word_id outside range), which
    cannot be recovered this way.
    """
    word_probs = [None] * n_tokens
    for token in flatten(paragraph):
        wid = token["word_id"]
        if wid is None:
            continue
        if wid < 0 or wid >= n_tokens:
            return None
        prob = token["probability"]
        word_probs[wid] = prob if word_probs[wid] is None else min(word_probs[wid], prob)
    # Forward-fill words the tokenizer merged away with the preceding word's prob.
    last = 1.0
    for i in range(n_tokens):
        if word_probs[i] is None:
            word_probs[i] = last
        else:
            last = word_probs[i]
    return word_probs


def collect_aligned(data_path, gold):
    """Return (aligned, skipped): aligned = [(unit_id, tokens, labels, word_probs)].

    data.json is a flat list of sentences (one per gold unit), each a list of
    per-token records.
    """
    gold_units = {u["unit_id"]: u for u in gold["units"]}
    sentences = load_json(data_path)
    if len(sentences) != len(gold_units):
        raise ValueError(
            f"{data_path}: {len(sentences)} sentences but gold has "
            f"{len(gold_units)} units"
        )

    aligned = []
    skipped = []
    for unit_id, sentence in enumerate(sentences):
        gu = gold_units[unit_id]
        word_probs = pool_word_probs(sentence, len(gu["tokens"]))
        if word_probs is None:
            skipped.append(unit_id)
            continue
        aligned.append((unit_id, gu["tokens"], gu["labels"], word_probs))
    return aligned, skipped


def threshold_grid():
    """Fixed candidate thresholds: log-spaced below 0.01, fine-linear above.

    The original grid was 0.01, 0.02, ... 0.99, and 6 of 11 archived runs
    selected exactly 0.01 -- the first value tried. An optimum sitting on the
    first grid point means the true optimum is *below* it and the sweep was
    clipping rather than tuning. Encoder MLM probabilities in particular sit
    orders of magnitude under 0.01 for most tokens, so the region that decides
    the operating point is decades wide, not one step wide; it needs log
    spacing, and 25 points per decade over 1e-6..1e-2 resolves it. Above 0.01 a
    0.001 linear step is already finer than the scoring data can distinguish.

    Still a fixed grid, computed once and identical for every run -- nothing
    here adapts to the data.
    """
    grid = [10.0 ** (-6.0 + i / 25.0) for i in range(100)]   # 1e-6 .. ~9.1e-3
    grid += [i / 1000.0 for i in range(10, 1000)]            # 0.010 .. 0.999
    return grid


def select_threshold(aligned, grid=None):
    """Pick the threshold maximising incorrect-class F0.5 over aligned units.

    Counts come from two sorted probability lists and a binary search per
    candidate, rather than rescanning every token per candidate. That is what
    makes a 1090-point grid affordable where the old 99-point one was already
    the expensive part of scoring: cost goes from O(grid x tokens) to
    O(tokens log tokens + grid log tokens). The selected value is identical to
    what the exhaustive scan would return -- `bisect_left` counts exactly the
    probabilities strictly below the threshold, which is the same `prob < thr`
    test applied when the labels are assigned.
    """
    from bisect import bisect_left

    if grid is None:
        grid = threshold_grid()

    probs_i, probs_c = [], []
    for _, _, labels, word_probs in aligned:
        for label, prob in zip(labels, word_probs):
            (probs_i if label == "i" else probs_c).append(prob)
    probs_i.sort()
    probs_c.sort()
    n_i = len(probs_i)

    best_thr, best_f05 = grid[0], -1.0
    for thr in grid:
        tp = bisect_left(probs_i, thr)          # gold i predicted i
        fp = bisect_left(probs_c, thr)          # gold c predicted i
        _, _, _, f05 = prf(tp, fp, n_i - tp)
        if f05 > best_f05:                      # strict: ties keep the lowest
            best_f05, best_thr = f05, thr
    return best_thr, best_f05


def adapt(data_path, tag, gold, threshold, out_path, tune=None):
    """Turn one run's data.json into canonical predictions.

    `tune` is an optional `(aligned, label)` pair from a DIFFERENT dataset, as
    produced by `collect_aligned` against that dataset's own gold. When given
    and the threshold is "auto", the operating point is fitted there and only
    applied here -- so the scored set contributes nothing to its own threshold.
    """
    aligned, skipped = collect_aligned(data_path, gold)
    tuned_on = None

    if threshold == "auto":
        if tune is not None:
            tune_aligned, tuned_on = tune
            threshold, sel_f05 = select_threshold(tune_aligned)
            selection = (f"auto (max incorrect-class F0.5 = {sel_f05*100:.2f}% on "
                         f"{tuned_on}, held out from this scoring set)")
        else:
            threshold, sel_f05 = select_threshold(aligned)
            selection = (f"auto (max incorrect-class F0.5 on dev = {sel_f05*100:.2f}%; "
                         f"selected on the set it is scored on)")
    else:
        threshold = float(threshold)
        selection = "fixed"

    units = []
    for unit_id, tokens, _labels, word_probs in aligned:
        predicted = ["i" if prob < threshold else "c" for prob in word_probs]
        units.append(
            {"unit_id": unit_id, "tokens": tokens, "predicted_labels": predicted}
        )

    pred = {
        "approach": "prob-threshold",
        "tag": tag,
        "meta": {
            "dataset": gold.get("dataset"),
            "source": str(data_path),
            "threshold": threshold,
            "threshold_selection": selection,
            "tuned_on": tuned_on,
            "pooling": "min sub-token probability per word",
            "skipped_units": len(skipped),
            "skipped_unit_ids": skipped,
        },
        "units": units,
    }
    save_json(out_path, pred)
    print(f"wrote {out_path}  units={len(units)}  threshold={threshold:.6g}  "
          f"skipped={len(skipped)}"
          + (f"  tuned-on={tuned_on}" if tuned_on else ""))
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="a run's data.json.")
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. english/en_fce_dev.")
    parser.add_argument("--threshold", default="auto", help="float, or 'auto'.")
    parser.add_argument("--tag", default=None, help="label; default: the run folder name.")
    parser.add_argument("--out", type=Path, default=None,
                        help="prediction json; default: <run>/predictions.json.")
    parser.add_argument("--tune-on", default=None,
                        help="dataset id to fit the threshold on instead of --dataset "
                             "(e.g. english/en_c4200_eval5k). Needs --tune-data.")
    parser.add_argument("--tune-data", type=Path, default=None,
                        help="data.json of the SAME model/template over --tune-on.")
    args = parser.parse_args()

    gold = load_json(paths.gold(args.dataset))
    tag = args.tag or args.data.parent.name
    out = args.out or (args.data.parent / "predictions.json")

    tune = None
    if args.tune_on or args.tune_data:
        if not (args.tune_on and args.tune_data):
            parser.error("--tune-on and --tune-data must be given together")
        tune_gold = load_json(paths.gold(args.tune_on))
        tune_aligned, _ = collect_aligned(args.tune_data, tune_gold)
        tune = (tune_aligned, paths.stem(args.tune_on))

    adapt(args.data, tag, gold, args.threshold, out, tune=tune)


if __name__ == "__main__":
    main()
