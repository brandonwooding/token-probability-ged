"""MLP error-word detector with attention + fragmentation features.

The original train-mlp.py used three LM-confidence features [probability,
log_rank, entropy]. This adds two more and reports the new model against that
exact three-feature baseline on the *same* held-out split, so any change is
attributable to the features rather than to a different draw:

  k          sub-tokens the tokenizer split the word into. An EXPLICIT
             fragmentation control: fragmentation correlates with rarity and
             misspelling (hence with the error class), and it also correlates
             with attention mass. Giving the MLP k directly stops it from
             reconstructing "is this word rare" out of the attention feature and
             calling it attention signal.

  self_attn  self_share from extract-attention.py: the fraction of a word's own
             outgoing attention that lands back on itself, on the head-averaged
             final-layer matrix. The word-level "measure of self-attention".

Feature table is built the current-pipeline way -- gold.json (flat per-word c/i
labels) joined to the flat per-token data.json and the per-word attention.json,
all three aligned by unit index and by word_id from the same get_custom_word_ids
call. Sub-token -> word pooling for the confidence features matches
confidence_predictions.pool_word_features exactly (min probability, max rank,
max entropy; merged-away words forward-filled). k and self_attn are already
per-word, so they are only placed into gold slots and forward-filled the same
way.

Protocol matches train-mlp.py: stratified held-out split, scale on train only,
tune the decision threshold for F0.5 on the held-out set (class is ~10% errors
and F0.5 is the project metric). This is a within-dev held-out estimate, not
train-on-train; it answers "do these features help", not "what is the final
operating point".

Run (after extract-attention.py has produced an attention run):
    python train-mlp-attention.py \
        --data   outputs/en_fce_dev__mmBERT-base__20260722-163541/data.json \
        --attn   outputs/en_fce_dev__mmBERT-base__attention__<ts> \
        --dataset english/en_fce_dev
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "standard-eval"))
from pooling import flatten  # noqa: E402

ALL_FEATURES = ["probability", "log_rank", "entropy", "k", "self_attn"]


def load_json(path):
    import json
    with open(path) as f:
        return json.load(f)


def build_table(gold, prob_data, attn_data):
    """Per-word feature rows + labels, joining the three files by unit and word.

    Returns (X, y, skipped_unit_ids) with X columns in ALL_FEATURES order.

    A word the tokenizer merged into a neighbour owns no sub-token, so it is
    absent from BOTH data.json and attention.json (both key off the same
    word_ids); it is forward-filled from the preceding word, exactly as the
    probability pipeline does. A genuinely out-of-range word_id -- unrecoverable
    -- skips the whole unit, matching the other adapters' guard.
    """
    gold_units = {u["unit_id"]: u for u in gold["units"]}
    n_units = len(gold_units)
    if not (len(prob_data) == len(attn_data) == n_units):
        raise ValueError(
            f"length mismatch: data={len(prob_data)} attn={len(attn_data)} "
            f"gold={n_units}"
        )

    rows, labels, skipped = [], [], []
    for uid in range(n_units):
        gu = gold_units[uid]
        n = len(gu["tokens"])

        # -- confidence features: least-confident sub-token per word.
        probs = [None] * n
        ranks = [None] * n
        ents = [None] * n
        bad = False
        for tok in flatten(prob_data[uid]):
            wid = tok["word_id"]
            if wid is None:
                continue
            if wid < 0 or wid >= n:
                bad = True
                break
            p, r, e = tok["probability"], tok["rank"], tok["entropy"]
            probs[wid] = p if probs[wid] is None else min(probs[wid], p)
            ranks[wid] = r if ranks[wid] is None else max(ranks[wid], r)
            ents[wid] = e if ents[wid] is None else max(ents[wid], e)
        if bad:
            skipped.append(uid)
            continue

        # -- attention features: already per word, keyed by present word_id.
        aw = attn_data[uid]
        k_by_word = dict(zip(aw["word_ids"], aw["k"]))
        self_by_word = dict(zip(aw["word_ids"], aw["self_share"]))
        ks = [k_by_word.get(i) for i in range(n)]
        selfs = [self_by_word.get(i) for i in range(n)]

        # -- forward-fill every feature over words the tokenizer merged away.
        last_p, last_r, last_e, last_k, last_s = 1.0, 1.0, 0.0, 1, 0.0
        for i in range(n):
            if probs[i] is None:
                probs[i], ranks[i], ents[i] = last_p, last_r, last_e
            else:
                last_p, last_r, last_e = probs[i], ranks[i], ents[i]
            if ks[i] is None:
                ks[i] = last_k
            else:
                last_k = ks[i]
            if selfs[i] is None:
                selfs[i] = last_s
            else:
                last_s = selfs[i]

        for i in range(n):
            rows.append([probs[i], math.log1p(ranks[i]), ents[i], ks[i], selfs[i]])
            labels.append(1 if gu["labels"][i] == "i" else 0)

    return np.asarray(rows, dtype=float), np.asarray(labels), skipped


def f05_from_pr(precision, recall):
    beta2 = 0.25
    denom = beta2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (1 + beta2) * precision * recall / denom, 0.0)


def best_f05_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f05 = f05_from_pr(precision[:-1], recall[:-1])
    best = int(np.argmax(f05))
    return float(thresholds[best]), float(f05[best])


def fit_and_report(name, cols, X_train, X_test, y_train, y_test, args):
    """Train one MLP on the given feature columns; return its held-out summary."""
    idx = [ALL_FEATURES.index(c) for c in cols]
    Xtr, Xte = X_train[:, idx], X_test[:, idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(int(h) for h in args.hidden.split(",") if h.strip()),
        activation="relu",
        alpha=1e-4,
        max_iter=args.max_iter,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=args.seed,
    )
    clf.fit(Xtr_s, y_train)
    scores = clf.predict_proba(Xte_s)[:, 1]

    roc = roc_auc_score(y_test, scores)
    pr = average_precision_score(y_test, scores)
    thr, f05 = best_f05_threshold(y_test, scores)

    print(f"\n=== {name}: {cols}")
    print(f"held-out roc_auc={roc:.4f}  pr_auc={pr:.4f}  "
          f"best F0.5={f05:.4f} @ thr={thr:.3f}")
    print(classification_report(y_test, (scores >= thr).astype(int),
                                digits=3, zero_division=0))
    return {"cols": cols, "roc": roc, "pr": pr, "f05": f05, "thr": thr,
            "clf": clf, "scaler": scaler, "idx": idx}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="probability run's data.json.")
    parser.add_argument("--attn", required=True,
                        help="attention run folder (or its attention.json).")
    parser.add_argument("--dataset", required=True, help="dataset id for gold.")
    parser.add_argument("--hidden", default="64,32")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    attn_path = Path(args.attn)
    if attn_path.is_dir():
        attn_path = attn_path / "attention.json"

    gold = load_json(paths.gold(args.dataset))
    prob_data = load_json(args.data)
    attn_data = load_json(attn_path)

    X, y, skipped = build_table(gold, prob_data, attn_data)
    print(f"loaded {len(y)} words | features={ALL_FEATURES} | "
          f"errors={int(y.sum())} ({y.mean():.1%}) | skipped units={len(skipped)}")
    # Quick univariate sanity: mean of each feature by class.
    for j, name in enumerate(ALL_FEATURES):
        m0, m1 = X[y == 0, j].mean(), X[y == 1, j].mean()
        print(f"  {name:12s} mean  correct={m0:.4f}  error={m1:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    baseline = fit_and_report("baseline (3 confidence features)",
                              ["probability", "log_rank", "entropy"],
                              X_train, X_test, y_train, y_test, args)
    full = fit_and_report("full (+ k + self_attn)",
                          ALL_FEATURES,
                          X_train, X_test, y_train, y_test, args)

    print("\n--- delta (full - baseline), same split ---")
    print(f"  roc_auc  {full['roc'] - baseline['roc']:+.4f}")
    print(f"  pr_auc   {full['pr'] - baseline['pr']:+.4f}")
    print(f"  F0.5     {full['f05'] - baseline['f05']:+.4f}")

    # Which features the full model actually leans on (held-out, PR-AUC scoring).
    imp = permutation_importance(
        full["clf"], full["scaler"].transform(X_test), y_test,
        scoring="average_precision", n_repeats=20, random_state=args.seed,
    )
    print("\n--- permutation importance (drop in held-out PR-AUC) ---")
    order = np.argsort(imp.importances_mean)[::-1]
    for j in order:
        print(f"  {ALL_FEATURES[j]:12s} {imp.importances_mean[j]:+.4f} "
              f"± {imp.importances_std[j]:.4f}")


if __name__ == "__main__":
    main()
