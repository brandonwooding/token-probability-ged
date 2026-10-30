"""Write confidence-model dev predictions into the canonical eval format.

You bring the trained model (logistic regression, MLP, ...) and a threshold;
this just handles the per-word pooling, the word->unit alignment, and writing
the {approach, tag, meta, units:[...]} file that score.py consumes. No training
happens here.

Typical use from your own training script, once a model is fit on FCE-train and
a threshold chosen on a train-val slice:

    from confidence_predictions import build_units, pool_word_features
    from common import load_json, write_predictions, EVAL_DIR

    gold = load_json(EVAL_DIR / "gold/fce-dev.json")
    units, skipped = build_units(
        dev_data="llm-probability-thresh/data.json",
        gold=gold,
        score_words=lambda rows: clf.predict_proba(scaler.transform(rows))[:, 1],
        threshold=thr,
    )
    write_predictions(
        EVAL_DIR / "predictions/confidence-mlp__modernbert-encoder.json",
        approach="confidence-mlp", tag="modernbert-encoder",
        units=units, meta={"threshold": thr, "skipped_units": len(skipped)},
    )

Then: python score.py --pred predictions/confidence-mlp__modernbert-encoder.json

`score_words` takes a list of per-word feature rows [probability, log_rank,
entropy] and returns one error-probability per word, in the same order.
"""

import math

from common import load_json
from pooling import flatten


def pool_word_features(paragraph, n_tokens):
    """Per-word [probability, log_rank, entropy], recovering merged-away words.

    Least-confident sub-token per word (min probability, max rank, max entropy) --
    the same pooling as adapt_thresh. Accepts either data.json shape; flatten()
    normalises it. As in adapt_thresh, the forward-fill recovery below only fires
    for legacy flat runs, where a word merged into a neighbour by the tokenizer
    inherits the nearest preceding word's pooled features (Lever A).
    Returns None only on a genuine out-of-range word_id.
    """
    probs = [None] * n_tokens
    ranks = [None] * n_tokens
    ents = [None] * n_tokens
    for token in flatten(paragraph):
        wid = token["word_id"]
        if wid is None:
            continue
        if wid < 0 or wid >= n_tokens:
            return None
        p, r, e = token["probability"], token["rank"], token["entropy"]
        probs[wid] = p if probs[wid] is None else min(probs[wid], p)
        ranks[wid] = r if ranks[wid] is None else max(ranks[wid], r)
        ents[wid] = e if ents[wid] is None else max(ents[wid], e)
    # Forward-fill words the tokenizer merged away with the preceding word's features.
    last = (1.0, 1.0, 0.0)
    for i in range(n_tokens):
        if probs[i] is None:
            probs[i], ranks[i], ents[i] = last
        else:
            last = (probs[i], ranks[i], ents[i])
    return [[p, math.log1p(r), e] for p, r, e in zip(probs, ranks, ents)]


def build_units(dev_data, gold, score_words, threshold):
    """Turn a trained word-scorer into canonical prediction units over dev.

    dev_data:    path to the encoder dev data.json (per-token probabilities).
    gold:        loaded canonical gold (for unit_id order + tokens).
    score_words: callable(feature_rows) -> per-word error-probabilities.
    threshold:   a word is "i" when its error-probability >= threshold.

    Returns (units, skipped_unit_ids). Words the tokenizer merged away are
    recovered by pool_word_features; only a genuine out-of-range word_id skips a
    unit (same guard the other adapters use).
    """
    gold_units = {u["unit_id"]: u for u in gold["units"]}
    sentences = load_json(dev_data)
    if len(sentences) != len(gold_units):
        raise ValueError(
            f"dev data has {len(sentences)} sentences but gold has "
            f"{len(gold_units)} units"
        )

    units, skipped = [], []
    for unit_id, sentence in enumerate(sentences):
        gu = gold_units[unit_id]
        feats = pool_word_features(sentence, len(gu["tokens"]))
        if feats is None:
            skipped.append(unit_id)
            continue
        probs = score_words(feats)
        predicted = ["i" if p >= threshold else "c" for p in probs]
        units.append(
            {"unit_id": unit_id, "tokens": gu["tokens"], "predicted_labels": predicted}
        )
    return units, skipped
