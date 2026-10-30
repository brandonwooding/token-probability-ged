"""Masked-copy construction, word alignment, and token-budget batching.

One sentence becomes one masked copy per *content* sub-token position. That is
not an arbitrary choice: the positions `ul` needs (error-word sub-tokens), the
positions `keep` needs (correct-word sub-tokens) and the positions `slot` needs
(divergence points) union to exactly the set of positions covered by some word.
Masking them all once therefore serves every objective from one set of forward
passes, and it is the same thing `llm-encoder-threshold.py` does at inference --
so training and scoring read the identical quantity.

Copies are separate ROWS of the batch, each with its own attention mask. No
sentence ever attends to another, and the copies of one sentence do not attend
to each other; they are only kept in the same forward so that the min over a
word's sub-tokens and the edits->example normalisation can be built in one
graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from reconstruct_from_tsv import positions_for_words  # noqa: E402


def ordered_unique(values):
    """De-duplicate preserving order.

    A sub-token that straddles two dataset words is listed under both, so
    concatenating a multi-word span's position lists can repeat one.
    """
    seen, out = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def common_prefix_length(a, b):
    index = 0
    while index < len(a) and index < len(b) and a[index] == b[index]:
        index += 1
    return index


def _side(tokenizer, text, words, max_length):
    """Tokenise one sentence and locate every word's sub-token positions."""
    encoded = tokenizer(text, add_special_tokens=True, truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
    spans = [(word["start"], word["end"]) for word in words]
    positions = positions_for_words(encoded["offset_mapping"], spans)
    if any(not row for row in positions):
        return None  # truncated away, or unalignable
    return {"input_ids": encoded["input_ids"], "word_positions": positions}


def _slot_targets(record, source, target):
    """One divergence-point comparison per replacement edit.

    Both sub-token sequences are read from their own sentence's tokenisation,
    never by tokenising the span text standalone -- boundary effects make
    `tokenize(" " + span)` disagree with the in-context slice, and the whole
    rule depends on the two sequences being the real ones.
    """
    out, skipped = [], {"empty_side": 0, "strict_prefix": 0}
    for edit in record["edits"]:
        if not edit.get("included_in_ged", True) or edit["operation"] != "R":
            continue
        source_positions = ordered_unique(
            p for w in range(edit["source_start"], edit["source_end"])
            for p in source["word_positions"][w])
        target_positions = ordered_unique(
            p for w in range(edit["target_start"], edit["target_end"])
            for p in target["word_positions"][w])
        if not source_positions or not target_positions:
            skipped["empty_side"] += 1  # insertion or deletion: no shared slot
            continue
        a = [source["input_ids"][p] for p in source_positions]
        b = [target["input_ids"][p] for p in target_positions]
        d = common_prefix_length(a, b)
        if d >= len(a) or d >= len(b):
            skipped["strict_prefix"] += 1
            continue
        out.append({
            "position": source_positions[d],
            "source_id": a[d],
            "target_id": b[d],
            "pair": "1:1" if (edit["source_end"] - edit["source_start"] == 1
                              and edit["target_end"] - edit["target_start"] == 1)
                    else "n:m",
        })
    return out, skipped


def encode_example(record, tokenizer, max_length):
    """-> a masked-copy plan for one rich record, or None if unusable."""
    source = _side(tokenizer, record["source"], record["source_words"], max_length)
    target = _side(tokenizer, record["target"], record["target_words"], max_length)
    if source is None or target is None:
        return None
    masked = ordered_unique(p for row in source["word_positions"] for p in row)
    if not masked:
        return None
    slots, skipped = _slot_targets(record, source, target)
    return {
        "record": record,
        "input_ids": source["input_ids"],
        "word_positions": source["word_positions"],
        "masked_positions": sorted(masked),
        "clean_input_ids": target["input_ids"],
        "slots": slots,
        "slot_skips": skipped,
        "n_copies": len(masked),
        "width": len(source["input_ids"]),
    }


def iter_batches(examples, token_budget, shuffle=False, rng=None):
    """Group examples so each batch holds at most `token_budget` token-positions.

    The budget is rows x padded width, NOT a count of masked copies. A sentence
    of n sub-tokens expands to n copies each of width n, so it contributes n^2
    token-positions -- and activation memory tracks that product, not the row
    count. Budgeting copies alone would let a batch of long sentences use ~6x
    the memory of a short-sentence batch with the same nominal size, which is
    exactly the variance a budget exists to remove.

    A single sentence larger than the budget still forms its own batch rather
    than being dropped; with gradient checkpointing on, the worst case at
    --max-length 192 is one sentence at 192^2 positions and fits.

    Greedy and order-preserving, so the language interleaving from
    ft_data._mix survives -- sorting by length would quietly undo it, since
    length correlates with language.
    """
    order = list(range(len(examples)))
    if shuffle:
        (rng or __import__("random")).shuffle(order)
    batch, rows, width = [], 0, 0
    for index in order:
        example = examples[index]
        grown_width = max(width, example["width"])
        grown_rows = rows + example["n_copies"]
        if batch and grown_rows * grown_width > token_budget:
            yield batch
            batch, rows, width = [], 0, 0
            grown_rows, grown_width = example["n_copies"], example["width"]
        batch.append(example)
        rows, width = grown_rows, grown_width
    if batch:
        yield batch


def collate(examples, mask_id, pad_id):
    """Expand examples into one padded row per masked position."""
    rows, row_example, row_position, row_observed = [], [], [], []
    for example_index, example in enumerate(examples):
        ids = example["input_ids"]
        for position in example["masked_positions"]:
            copy = list(ids)
            row_observed.append(copy[position])
            copy[position] = mask_id
            rows.append(copy)
            row_example.append(example_index)
            row_position.append(position)
    width = max(len(row) for row in rows)
    input_ids = [row + [pad_id] * (width - len(row)) for row in rows]
    attention = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
    index = {(e, p): i for i, (e, p) in enumerate(zip(row_example, row_position))}
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "mask_index": torch.tensor(row_position, dtype=torch.long),
        "observed_id": torch.tensor(row_observed, dtype=torch.long),
        "row_of": index,
        "examples": examples,
    }


def apply_head(model, hidden):
    """hidden [R, H] -> logits [R, V], without ever materialising [R, L, V].

    With a 256k vocabulary, full-sequence logits for a 320-row batch at length
    192 would be ~63 GB in fp32. Only the masked position is ever read, so the
    hidden state is gathered *before* the head and the vocabulary tensor is
    [R, V] (~0.33 GB) instead. This is not an optimisation; the design does not
    fit in memory otherwise.
    """
    if hasattr(model, "lm_head"):                 # RoBERTa / XLM-R
        return model.lm_head(hidden)
    if hasattr(model, "head") and hasattr(model, "decoder"):   # ModernBERT / mmBERT
        return model.decoder(model.head(hidden))
    if hasattr(model, "cls"):                     # BERT
        return model.cls(hidden)
    raise TypeError(
        f"{type(model).__name__}: cannot locate the masked-LM head. Add it to "
        "apply_head() -- gathering before the head is required, not optional.")


def masked_log_probs(model, batch):
    """-> log-softmax over the vocabulary at each row's masked position, [R, V]."""
    body = model.base_model
    hidden = body(input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"]).last_hidden_state
    picked = hidden[torch.arange(hidden.size(0), device=hidden.device),
                    batch["mask_index"]]
    return torch.log_softmax(apply_head(model, picked).float(), dim=-1)


def word_min_logprobs(log_probs, batch):
    """Per example, per source word: min teacher-forced log P over sub-tokens.

    The same rule inference uses, so what training moves is what the threshold
    later reads.
    """
    observed = log_probs[torch.arange(log_probs.size(0), device=log_probs.device),
                         batch["observed_id"]]
    out = []
    for example_index, example in enumerate(batch["examples"]):
        words = []
        for positions in example["word_positions"]:
            scores = [observed[batch["row_of"][(example_index, p)]]
                      for p in positions if (example_index, p) in batch["row_of"]]
            words.append(torch.stack(scores).min() if scores else None)
        out.append(words)
    return out, observed


def build_mlm_batch(examples, tokenizer, rate, rng, pad_id, mask_id):
    """Standard multi-mask MLM over the CLEAN sentences of a batch.

    This is the one term that does not need one-mask fidelity. It runs on the
    corrected sentence, which never appears at inference at all, so nothing
    downstream reads the probabilities it shapes -- it is an anchor that stops
    the error-aware terms from being satisfied by degrading the distribution.
    `keep`, whose probabilities ARE read at scoring time, gets single masks.
    """
    special = set(tokenizer.all_special_ids)
    rows, row_of_target, position_of_target, targets = [], [], [], []
    for example in examples:
        ids = list(example["clean_input_ids"])
        candidates = [p for p, token in enumerate(ids) if token not in special]
        if not candidates:
            continue
        count = max(1, int(round(rate * len(candidates))))
        for position in rng.sample(candidates, min(count, len(candidates))):
            targets.append(ids[position])
            row_of_target.append(len(rows))
            position_of_target.append(position)
            ids[position] = mask_id
        rows.append(ids)
    if not rows:
        return None
    width = max(len(row) for row in rows)
    return {
        "input_ids": torch.tensor([r + [pad_id] * (width - len(r)) for r in rows],
                                  dtype=torch.long),
        "attention_mask": torch.tensor(
            [[1] * len(r) + [0] * (width - len(r)) for r in rows], dtype=torch.long),
        "row": torch.tensor(row_of_target, dtype=torch.long),
        "position": torch.tensor(position_of_target, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
    }


def mlm_loss(model, batch):
    """Cross-entropy at the masked clean-sentence positions only."""
    hidden = model.base_model(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"]).last_hidden_state
    picked = hidden[batch["row"], batch["position"]]
    logits = apply_head(model, picked).float()
    return torch.nn.functional.cross_entropy(logits, batch["target"])


def move(batch, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}
