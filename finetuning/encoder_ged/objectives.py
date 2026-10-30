"""Family-E masked-LM probability objectives.

Three error-aware terms over the corrupt sentence, plus an `mlm` anchor on the
clean sentence supplied by the caller. Normalisation follows the decoder
family: sub-tokens->word (min), words->edit, edits->example, examples->batch,
so an example with many edits does not outweigh one with a single edit.
"""
from __future__ import annotations

import torch


def _mean(values):
    values = [value for value in values if value is not None]
    return torch.stack(values).mean() if values else None


def slot_loss(log_probs, batch, margin=1.0):
    """Divergence-point contrast, read from one distribution at one position.

    For a replacement edit the wrong word and the right word occupy the same
    slot, so masking that single position yields ONE softmax containing both
    candidates -- unlike the decoder's cross-context contrast, whose reference
    came from a different sentence at a different position with a different
    span length. Both numbers are looked up in the same row here, which is why
    a uniform flattening of the distribution cannot reduce this loss.

    The compared position is the first index at which the two tokenisations
    diverge, so the two candidates are never the same token; comparing at a
    fixed end instead would be degenerate on 39% of Russian and 24% of German
    multi-sub-token replacements (shared stem) or 17% and 15% (shared suffix).
    """
    per_example, counts = [], {"1:1": 0, "n:m": 0}
    for example_index, example in enumerate(batch["examples"]):
        edits = []
        for slot in example["slots"]:
            key = (example_index, slot["position"])
            if key not in batch["row_of"]:
                continue
            row = batch["row_of"][key]
            source = log_probs[row, slot["source_id"]]
            target = log_probs[row, slot["target_id"]]
            edits.append(torch.relu(margin + source - target))
            counts[slot["pair"]] += 1
        value = _mean(edits)
        if value is not None:
            per_example.append(value)
    return _mean(per_example), counts


def unlikelihood_loss(word_scores, batch):
    """-log(1 - P(word)) over every labelled erroneous word.

    Reads `label_indices`, so it covers all three edit shapes: the wrong word
    of a replacement, the spurious word of a deletion, and -- because the
    canonical policy attaches an insertion to the next source token -- the
    anchor word of an insertion. Sentence-final insertions mark no token and
    are excluded, as are edits outside the GED policy (noop, UNK, *:PUNCT).
    """
    per_example, skipped_final = [], 0
    for scores, example in zip(word_scores, batch["examples"]):
        edits = []
        for edit in example["record"]["edits"]:
            if not edit.get("included_in_ged", True):
                continue
            if edit.get("sentence_final_missing"):
                skipped_final += 1
                continue
            erroneous = [scores[i] for i in edit["label_indices"]
                         if i < len(scores) and scores[i] is not None]
            if erroneous:
                # clamp is numerical protection only, not a modelling choice
                edits.append(_mean([-torch.log1p(-value.exp().clamp(max=1 - 1e-7))
                                    for value in erroneous]))
        value = _mean(edits)
        if value is not None:
            per_example.append(value)
    return _mean(per_example), skipped_final


def keep_loss(word_scores, batch):
    """-log P(word) over every unchanged word, plain mean within the sentence.

    The decoder splits this at the first error and weights the two halves
    equally, because a causal model's left context is clean before the first
    error and contaminated after. An encoder sees the whole sentence, so every
    correct word is contaminated by every error regardless of position and the
    partition has no referent.

    A flat mean is not a dilution, but the reason is easy to state wrongly.
    With respect to the log-probability the gradient is a constant -1/N. It is
    at the LOGITS -- which is what the parameters actually move -- that the
    term self-concentrates: d(-log p_t)/d(logit_t) = p_t - 1, so a correct word
    still scoring .9 pulls with .10 while one dragged to .05 pulls with .95.
    The words nearest the damage, which are the ones that produce false
    positives, therefore dominate the update without any distance weighting.
    """
    per_example = []
    for scores, example in zip(word_scores, batch["examples"]):
        labels = example["record"]["labels"]
        kept = [-scores[i] for i, label in enumerate(labels)
                if label == "c" and i < len(scores) and scores[i] is not None]
        value = _mean(kept)
        if value is not None:
            per_example.append(value)
    return _mean(per_example)


def combine(components, weights, reference):
    """Weighted sum, with absent components contributing a real zero."""
    zero = reference * 0.
    total = zero
    resolved = {}
    for name, weight in weights.items():
        value = components.get(name)
        resolved[name] = value if value is not None else zero
        if weight:
            total = total + weight * resolved[name]
    return total, resolved
