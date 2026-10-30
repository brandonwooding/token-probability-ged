"""Train one confidence classifier (MLP or logistic regression) and write its
dev predictions in the standard-eval canonical format.

Method: a supervised error-word detector over encoder confidence features. Each
classifier sees probability + log_rank + entropy + exactly ONE attention metric
(token_mean / word_vote / word_vote_renorm / word_vote_zeroed), so running the
2 classifier types x 4 attention metrics grid gives 8 comparable systems.

HONEST split (unlike the retired encoder-probability-thresh/train-mlp.py, which
trained on 80% of dev and tested on the other 20%):
  * fit on the TRAIN split (en_fce_train),
  * pick the F0.5-optimal decision threshold on a held-out slice of TRAIN,
  * predict on the FULL dev split -- the same 2191 units every other approach is
    scored on, so this drops straight into standard-eval as a comparable row.

Features come from two encoder runs of the SAME model (so their word alignment
matches): the probability/rank/entropy data.json and the attention.json. Both
are auto-discovered as the latest matching runs under
encoder-probability-thresh/outputs/, or given explicitly.

This is CPU-only sklearn -- no GPU. The GPU cost is generating the feature runs
(the encoder + attention passes over each split), done separately.

    python train_confidence_clf.py --attn token_mean --clf mlp
    python train_confidence_clf.py --attn word_vote --clf logreg \
        --train english/en_fce_train --eval english/en_fce_dev
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "standard-eval"))
import paths  # noqa: E402
from common import load_json, write_predictions  # noqa: E402
import features as feat  # noqa: E402

DEFAULT_MODEL_ID = "mmBERT-base"  # multilingual, so the same setup extends to cs/de/it/sv


def resolve_runs(stem, model_id):
    """Latest (prob data.json, attention.json) pair for one split+model.

    Run names are '<stem>__<model>__<ts>' (prob, 3 '__'-parts because the stem
    uses single underscores) and '<stem>__<model>__attention__<ts>' (attention,
    4 parts). That difference is what tells the two runs apart.
    """
    outs = paths.APPROACH_DIRS["encoder-thresh"] / "outputs"
    runs = sorted(outs.glob(f"{stem}__{model_id}__*"))
    prob = [r for r in runs if len(r.name.split("__")) == 3]
    attn = [r for r in runs if "attention" in r.name.split("__")]
    if not prob:
        raise FileNotFoundError(
            f"no prob run '{stem}__{model_id}__<ts>/data.json' under {outs} "
            f"(generate the encoder features for this split first)"
        )
    if not attn:
        raise FileNotFoundError(
            f"no attention run '{stem}__{model_id}__attention__<ts>/attention.json' "
            f"under {outs} (generate the attention features for this split first)"
        )
    return prob[-1] / "data.json", attn[-1] / "attention.json"


def _f05_curve(precision, recall):
    b2 = 0.25  # beta=0.5 weights precision 4:1
    denom = b2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (1 + b2) * precision * recall / denom, 0.0)


def best_threshold(y_true, scores):
    """Threshold maximising incorrect-class F0.5 on the given (held-out) set."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one fewer threshold than precision/recall;
    # the final point (recall=0) has no threshold, so drop it.
    f05 = _f05_curve(precision[:-1], recall[:-1])
    best = int(np.argmax(f05))
    return float(thresholds[best]), float(f05[best])


def make_classifier(kind, hidden, seed, max_iter):
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=1e-4,               # L2 regularisation
            max_iter=max_iter,
            early_stopping=True,      # internal val slice for convergence
            n_iter_no_change=15,
            random_state=seed,
        )
    if kind == "logreg":
        return LogisticRegression(max_iter=max_iter, random_state=seed)
    raise ValueError(f"unknown classifier {kind!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="english/en_fce_train",
                    help="dataset id to FIT on (default: english/en_fce_train).")
    ap.add_argument("--eval", dest="eval_ds", default="english/en_fce_dev",
                    help="dataset id to PREDICT on (default: english/en_fce_dev).")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID,
                    help="feature-source encoder model id (default: mmBERT-base).")
    ap.add_argument("--attn", required=True, choices=feat.ATTENTION_METRICS,
                    help="the single attention metric to include as the 4th feature.")
    ap.add_argument("--clf", required=True, choices=["mlp", "logreg"])
    ap.add_argument("--hidden", default="64,32", help="MLP hidden sizes, e.g. '64,32'.")
    ap.add_argument("--val-size", type=float, default=0.15,
                    help="fraction of TRAIN held out for threshold selection.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=500)
    # Explicit feature paths override auto-discovery (e.g. before the run naming
    # settles, or to pin a specific run).
    ap.add_argument("--train-data")
    ap.add_argument("--train-attn")
    ap.add_argument("--eval-data")
    ap.add_argument("--eval-attn")
    args = ap.parse_args()

    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())

    if args.train_data or args.train_attn:
        tr_data, tr_attn = Path(args.train_data), Path(args.train_attn)
    else:
        tr_data, tr_attn = resolve_runs(paths.stem(args.train), args.model)
    if args.eval_data or args.eval_attn:
        ev_data, ev_attn = Path(args.eval_data), Path(args.eval_attn)
    else:
        ev_data, ev_attn = resolve_runs(paths.stem(args.eval_ds), args.model)

    train_gold = load_json(paths.gold(args.train))
    eval_gold = load_json(paths.gold(args.eval_ds))

    _, X_tr, y_tr, tr_skip = feat.build(tr_data, tr_attn, train_gold, args.attn, want_labels=True)
    ev_units, _, _, ev_skip = feat.build(ev_data, ev_attn, eval_gold, args.attn, want_labels=False)
    X_tr = np.asarray(X_tr, dtype=float)
    y_tr = np.asarray(y_tr, dtype=int)
    names = feat.feature_names(args.attn)
    print(f"[train {args.train}] words={len(y_tr)} errors={int(y_tr.sum())} "
          f"({y_tr.mean():.1%}) skipped_units={len(tr_skip)} | features={names}")
    print(f"[eval {args.eval_ds}] units={len(ev_units)} skipped_units={len(ev_skip)}")

    # Fit slice / threshold-selection slice, both from TRAIN (dev is never seen).
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=args.val_size, random_state=args.seed, stratify=y_tr
    )
    scaler = StandardScaler().fit(X_fit)
    clf = make_classifier(args.clf, hidden, args.seed, args.max_iter)
    clf.fit(scaler.transform(X_fit), y_fit)

    val_scores = clf.predict_proba(scaler.transform(X_val))[:, 1]
    threshold, val_f05 = best_threshold(y_val, val_scores)
    val_ap = average_precision_score(y_val, val_scores)
    val_roc = roc_auc_score(y_val, val_scores)
    print(f"[val] threshold={threshold:.3f} F0.5={val_f05:.4f} "
          f"AP={val_ap:.4f} ROC_AUC={val_roc:.4f}")

    # Predict the FULL eval split, per unit, in gold order.
    units = []
    for unit_id, tokens, rows in ev_units:
        scores = clf.predict_proba(scaler.transform(np.asarray(rows, dtype=float)))[:, 1]
        units.append({
            "unit_id": unit_id,
            "tokens": tokens,
            "predicted_labels": ["i" if s >= threshold else "c" for s in scores],
        })

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{args.attn}-{args.clf}"
    run_dir = paths.run_dir("confidence-clf", args.eval_ds, args.model, tag=tag, timestamp=timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "approach": "confidence-clf",
        "classifier": args.clf,
        "hidden": list(hidden) if args.clf == "mlp" else None,
        "attention_metric": args.attn,
        "feature_names": names,
        "feature_model": args.model,
        "train_dataset": args.train,
        "eval_dataset": args.eval_ds,
        "train_data": str(tr_data),
        "train_attn": str(tr_attn),
        "eval_data": str(ev_data),
        "eval_attn": str(ev_attn),
        "train_words": int(len(y_tr)),
        "train_error_rate": float(y_tr.mean()),
        "val_size": args.val_size,
        "threshold": threshold,
        "threshold_selection": f"F0.5-optimal on {args.val_size:.0%} held-out TRAIN slice",
        "val_f05": val_f05,
        "val_ap": val_ap,
        "val_roc_auc": val_roc,
        "train_skipped_units": tr_skip,
        "eval_skipped_units": ev_skip,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    write_predictions(run_dir / "predictions.json", approach="confidence-clf",
                      tag=run_dir.name, units=units, meta=meta)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "threshold": threshold,
                     "feature_names": names}, f)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
