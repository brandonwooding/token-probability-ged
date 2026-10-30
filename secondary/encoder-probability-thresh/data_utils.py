"""Shared loader for the encoder probability-threshold experiment.

Turns a run's per-token `data.json` plus the gold paragraph labels into a flat
`(X, y)` design matrix at the *word* level, which is the format the downstream
predictors (logistic regression, MLP, ...) need.

Aggregation mirrors `evaluate-probs.py`: a word can span several sub-word
tokens, so we take the model's *least confident* token as that word's score
(min probability, min percentile, max rank, max entropy). Label "c" -> 0
(correct), "i" -> 1 (incorrect / error to be detected).
"""

import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "data.json"
DEFAULT_LABELS = SCRIPT_DIR / "fce-dev-paragraph-labels.json"

# Order matters: it is the column order of the returned design matrix.
FEATURE_NAMES = ["probability", "log_rank", "entropy"]


def load_word_scores(data_path=DEFAULT_DATA, label_path=DEFAULT_LABELS):
    """Read per-word scores split by class.

    Returns a dict with keys 'c' and 'i', each mapping to a dict of parallel
    lists: probs, percs, ranks, ents. Also returns bookkeeping counts.
    """
    with open(label_path, "r") as file:
        label_data = json.load(file)
    with open(data_path, "r") as file:
        prob_data = json.load(file)

    prob_data_flat = [paragraph for essay in prob_data for paragraph in essay]
    label_data_flat = [paragraph for essay in label_data for paragraph in essay]
    assert len(prob_data_flat) == len(label_data_flat), "data / label length mismatch"

    scores = {
        "c": {"probs": [], "percs": [], "ranks": [], "ents": []},
        "i": {"probs": [], "percs": [], "ranks": [], "ents": []},
    }
    skipped_paragraphs = 0
    skipped_label_tokens = 0

    for paragraph_idx, prob_paragraph in enumerate(prob_data_flat):
        labels = label_data_flat[paragraph_idx]["labels"]
        word_ids = [t["word_id"] for t in prob_paragraph if t["word_id"] is not None]

        if not word_ids:
            skipped_paragraphs += 1
            skipped_label_tokens += len(labels)
            continue

        # Every labelled word must be covered exactly once by the model tokens.
        if set(word_ids) != set(range(len(labels))):
            skipped_paragraphs += 1
            skipped_label_tokens += len(labels)
            continue

        n = len(labels)
        word_probs = [1.0] * n
        word_percs = [1.0] * n
        word_ranks = [1] * n
        word_ents = [0.0] * n

        for token in prob_paragraph:
            word_id = token["word_id"]
            if word_id is None:
                continue
            word_probs[word_id] = min(word_probs[word_id], token["probability"])
            word_percs[word_id] = min(word_percs[word_id], token["percentile"])
            word_ranks[word_id] = max(word_ranks[word_id], token["rank"])
            word_ents[word_id] = max(word_ents[word_id], token["entropy"])

        for label, prob, perc, rank, ent in zip(labels, word_probs, word_percs, word_ranks, word_ents):
            if label not in ("c", "i"):
                raise ValueError(f"unexpected label: {label}")
            bucket = scores[label]
            bucket["probs"].append(prob)
            bucket["percs"].append(perc)
            bucket["ranks"].append(rank)
            bucket["ents"].append(ent)

    scores["skipped_paragraphs"] = skipped_paragraphs
    scores["skipped_label_tokens"] = skipped_label_tokens
    scores["used_paragraphs"] = len(prob_data_flat) - skipped_paragraphs
    return scores


def load_features(data_path=DEFAULT_DATA, label_path=DEFAULT_LABELS):
    """Return (X, y, feature_names) ready for sklearn.

    X columns follow FEATURE_NAMES: [probability, log_rank, entropy].
    Rank is compressed with log1p (heavy right tail). Correct words come first
    (y=0), then incorrect (y=1) -- same ordering the LR script used.
    """
    scores = load_word_scores(data_path, label_path)
    c, i = scores["c"], scores["i"]

    probs = np.asarray(c["probs"] + i["probs"], dtype=float)
    ranks = np.asarray(c["ranks"] + i["ranks"], dtype=float)
    ents = np.asarray(c["ents"] + i["ents"], dtype=float)

    X = np.column_stack([probs, np.log1p(ranks), ents])
    y = np.array([0] * len(c["probs"]) + [1] * len(i["probs"]))
    return X, y, list(FEATURE_NAMES)
