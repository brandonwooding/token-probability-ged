"""Hierarchically normalised Family-A token probability objectives."""
from __future__ import annotations

import torch


def word_min_logprobs(logits, input_ids, word_positions):
    """Minimum teacher-forced subtoken log P for every word."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for batch_index, positions in enumerate(word_positions):
        words = []
        for token_positions in positions:
            scores = [log_probs[batch_index, pos - 1, input_ids[batch_index, pos]]
                      for pos in token_positions if pos > 0]
            words.append(torch.stack(scores).min() if scores else None)
        out.append(words)
    return out


def _mean(values):
    values = [value for value in values if value is not None]
    return torch.stack(values).mean() if values else None


def resolve_edit(edit, n_source, n_target, empty_policy):
    # Existing by-construction GED labels are the source-side authority.  This
    # matters for ambiguous multi-token alignments, where a minimum edit span
    # may contain unchanged words that must not be pushed down.
    source = list(edit["label_indices"])
    target = list(range(edit["target_start"], edit["target_end"]))
    structurally_empty_source = edit["source_start"] == edit["source_end"]
    if source and target and not structurally_empty_source:
        return source, target
    if empty_policy == "skip" or edit.get("sentence_final_missing"):
        return None
    if empty_policy != "anchor-next":
        raise ValueError(f"unknown empty edit policy: {empty_policy}")
    anchor_source, anchor_target = edit["source_end"], edit["target_end"]
    if anchor_source >= n_source or anchor_target >= n_target:
        return None
    return sorted(set(source + [anchor_source])), sorted(set(target + [anchor_target]))


def auxiliary_losses(source_scores, target_scores, records, *, margin=1.0,
                     empty_policy="skip", contrast_weight=0.0,
                     unlikelihood_weight=0.0, keep_weight=0.0):
    """Return total and components with edit→example→batch normalisation.

    Target scores are normally detached by the caller.  This keeps the second
    forward's graph small while correction CE from the clean pass supplies the
    positive pressure on correct tokens.
    """
    per_component = {"contrast": [], "unlikelihood": [], "keep": []}
    skipped_empty = skipped_final = 0
    for src, tgt, record in zip(source_scores, target_scores, records):
        edit_rank, edit_ul = [], []
        for edit in record["edits"]:
            # The complete rich record retains punctuation/UNK edits for M2
            # provenance, but they are intentionally outside the GED target.
            if not edit.get("included_in_ged", True):
                continue
            resolved = resolve_edit(edit, len(src), len(tgt), empty_policy)
            if resolved is None:
                if edit.get("sentence_final_missing"):
                    skipped_final += 1
                elif not (edit["source_start"] < edit["source_end"] and
                          edit["target_start"] < edit["target_end"]):
                    skipped_empty += 1
            else:
                source_indices, target_indices = resolved
                source_mean = _mean([src[i] for i in source_indices])
                target_mean = _mean([tgt[i] for i in target_indices])
                if source_mean is not None and target_mean is not None:
                    edit_rank.append(torch.relu(margin + source_mean - target_mean))
            erroneous = [src[i] for i in edit["label_indices"] if i < len(src)]
            if erroneous and not edit.get("sentence_final_missing"):
                # UL(word) = -log(1-P(word)); clamp is only numerical protection.
                edit_ul.append(_mean([-torch.log1p(-value.exp().clamp(max=1 - 1e-7))
                                      for value in erroneous]))
        rank_example, ul_example = _mean(edit_rank), _mean(edit_ul)
        if rank_example is not None:
            per_component["contrast"].append(rank_example)
        if ul_example is not None:
            per_component["unlikelihood"].append(ul_example)

        error_indices = [i for i, label in enumerate(record["labels"]) if label == "i"]
        if error_indices:
            first = min(error_indices)
            before = _mean([-src[i] for i in range(first) if record["labels"][i] == "c"])
            after = _mean([-src[i] for i in range(first + 1, len(src))
                           if record["labels"][i] == "c"])
            keep_example = _mean([before, after])
            if keep_example is not None:
                per_component["keep"].append(keep_example)

    zero = next((score for row in source_scores for score in row if score is not None), None)
    if zero is None:
        raise ValueError("batch has no scored source words")
    components = {key: (_mean(values) if values else zero * 0)
                  for key, values in per_component.items()}
    total = (contrast_weight * components["contrast"]
             + unlikelihood_weight * components["unlikelihood"]
             + keep_weight * components["keep"])
    return total, components, {"skipped_empty_edits": skipped_empty,
                               "skipped_sentence_final_missing": skipped_final}
