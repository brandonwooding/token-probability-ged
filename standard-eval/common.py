"""Shared helpers for the standardised GED evaluation.

Everything here is pure standard library so the scoring pipeline runs anywhere
(no numpy / torch / sklearn needed). Heavy approaches that need torch do their
own imports in their own adapter.

Two canonical file shapes flow through the pipeline:

  gold        {dataset, convention, units: [{unit_id, tokens, labels, types}]}
  predictions {approach, tag, meta, units: [{unit_id, tokens, predicted_labels}]}

`labels` / `predicted_labels` are per-token "c"/"i". `types` is per-token and
only meaningful on "i" tokens (None elsewhere). Predictions never carry types --
none of these approaches predict an error type, only detection.
"""

import json
from pathlib import Path

# repo root = parent of standard-eval/
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent

POSITIVE = "i"  # the "incorrect" class is what we score detection on


def strip_suffix(text, suffix):
    """Portable str.removesuffix (available only on Python 3.9+)."""
    return text[: -len(suffix)] if suffix and text.endswith(suffix) else text


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_predictions(out_path, approach, tag, units, meta=None):
    """Write a canonical predictions file that score.py can read.

    units: list of {unit_id, tokens, predicted_labels} where predicted_labels
    are per-token "c"/"i" aligned to tokens. Any approach can call this.
    """
    save_json(out_path, {
        "approach": approach,
        "tag": tag,
        "meta": meta or {},
        "units": units,
    })
    return out_path


def prf(tp, fp, fn):
    """Precision, recall, F1, F0.5 for a single positive class."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    def f_beta(beta):
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    return precision, recall, f_beta(1.0), f_beta(0.5)
