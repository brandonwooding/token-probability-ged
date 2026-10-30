"""Join encoder confidence features (probability, rank, entropy) with ONE
attention metric, per word, for the confidence-classifier method.

Two encoder runs feed one classifier, and both MUST come from the same model so
their word segmentation lines up:

    data.json       token-level probability / rank / entropy   (pooled to word here)
    attention.json  word-level attention metrics               (one column selected)

Everything is aligned to the GOLD units: for unit i we build exactly
len(gold.tokens) word rows, recovering tokenizer-merged words the same way the
thresh adapters do (a merged-away word inherits its predecessor's features), so
the feature matrix, the gold labels and the canonical prediction units stay in
lockstep -- no drift between what the model scored and what the scorer reads.

The four attention metrics are alternatives, not extra columns: each classifier
uses probability + log_rank + entropy + exactly ONE of them (see ATTENTION_METRICS),
so the attention method is a controlled variable rather than a bag of features.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# pool_word_features (prob/log_rank/entropy per word, with merged-word recovery)
# is the shared thresh pooling -- reuse it so this method pools identically.
sys.path.insert(0, str(REPO_ROOT / "standard-eval"))
from confidence_predictions import pool_word_features  # noqa: E402

# The four attention metrics live side by side in attention.json; a classifier
# picks exactly one. Order is informational only.
ATTENTION_METRICS = ["token_mean", "word_vote", "word_vote_renorm", "word_vote_zeroed"]
BASE_FEATURES = ["probability", "log_rank", "entropy"]


def _load(path):
    return json.loads(Path(path).read_text())


def pool_word_attention(attn_sentence, n_tokens, metric):
    """One attention value per gold word, forward-filling merged-away words.

    attention.json is already word-level (parallel `word_ids` + per-metric
    arrays), so a gold word_id normally appears exactly once. Any duplicates
    (should not occur) are averaged; a gold word with no attention row of its own
    was merged into a neighbour by the tokenizer and inherits the previous word's
    value -- the same Lever A recovery pool_word_features applies to the
    probability side. Returns None only on a genuinely out-of-range word_id.
    """
    acc = [None] * n_tokens
    cnt = [0] * n_tokens
    for wid, val in zip(attn_sentence["word_ids"], attn_sentence[metric]):
        if wid is None:
            continue
        if wid < 0 or wid >= n_tokens:
            return None
        acc[wid] = val if acc[wid] is None else acc[wid] + val
        cnt[wid] += 1
    vals = [None if a is None else a / c for a, c in zip(acc, cnt)]
    last = 0.0
    for i in range(n_tokens):
        if vals[i] is None:
            vals[i] = last
        else:
            last = vals[i]
    return vals


def feature_names(metric):
    return BASE_FEATURES + [metric]


def build(data_path, attn_path, gold, metric, want_labels):
    """Assemble the per-word feature matrix for one (split, model, metric).

    Returns (per_unit, X, y, skipped):
      per_unit  list of (unit_id, tokens, feat_rows) -- feat_rows is
                len(tokens) x 4, aligned to gold, for writing predictions.
      X         flat list of 4-dim rows (all units concatenated, gold order).
      y         flat list of 0/1 error labels aligned to X, or None.
      skipped   unit_ids dropped on a genuine misalignment (out-of-range word_id).
    """
    if metric not in ATTENTION_METRICS:
        raise ValueError(f"unknown attention metric {metric!r}; have {ATTENTION_METRICS}")
    sentences = _load(data_path)
    attn = _load(attn_path)
    gold_units = gold["units"]
    if not (len(sentences) == len(attn) == len(gold_units)):
        raise ValueError(
            f"length mismatch: data={len(sentences)} attn={len(attn)} "
            f"gold={len(gold_units)} -- all three must share sentences.json order"
        )

    per_unit, X, y, skipped = [], [], [], []
    for gu, sent, asent in zip(gold_units, sentences, attn):
        n = len(gu["tokens"])
        base = pool_word_features(sent, n)              # n x 3, or None
        att = pool_word_attention(asent, n, metric)     # n,   or None
        if base is None or att is None:
            skipped.append(gu["unit_id"])
            continue
        rows = [b + [a] for b, a in zip(base, att)]     # n x 4
        per_unit.append((gu["unit_id"], gu["tokens"], rows))
        X.extend(rows)
        if want_labels:
            y.extend(1 if lab == "i" else 0 for lab in gu["labels"])

    return per_unit, X, (y if want_labels else None), skipped
