"""Shared prompt/token/word alignment for Family A and Family D."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "decoder-probability-thresh"))
import prompts  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
from reconstruct_from_tsv import positions_for_words  # noqa: E402


def _body_encoding(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)


def encode_slot(record, tokenizer, field, max_length, include_eos=False):
    """Put source or target into Variant-B's perfect slot and locate its words."""
    prefix = prompts.variant_b(record["source"], record["language_folder"])
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    body = _body_encoding(tokenizer, record[field])
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    ids = [bos] + prefix_ids + body["input_ids"]
    if include_eos:
        ids.append(tokenizer.eos_token_id)
    if len(ids) > max_length:
        return None
    shift = 1 + len(prefix_ids)
    words = record["source_words" if field == "source" else "target_words"]
    word_spans = [(word["start"], word["end"]) for word in words]
    word_positions = positions_for_words(
        body["offset_mapping"], word_spans, shift)
    if any(not positions for positions in word_positions):
        return None
    labels = [-100] * shift + body["input_ids"]
    if include_eos:
        labels.append(tokenizer.eos_token_id)
    return {"input_ids": ids, "labels": labels, "word_positions": word_positions,
            "body_start": shift}


def build_a_example(record, tokenizer, max_length):
    clean = encode_slot(record, tokenizer, "target", max_length, include_eos=True)
    corrupt = encode_slot(record, tokenizer, "source", max_length, include_eos=False)
    return ({"record": record, "clean": clean, "corrupt": corrupt}
            if clean is not None and corrupt is not None else None)


def build_d_example(record, tokenizer, max_length):
    corrupt = encode_slot(record, tokenizer, "source", max_length, include_eos=False)
    return ({"record": record, "corrupt": corrupt} if corrupt is not None else None)


def collate(examples, pad_id, key):
    encoded = [example[key] for example in examples]
    width = max(len(item["input_ids"]) for item in encoded)
    ids, masks, labels = [], [], []
    for item in encoded:
        padding = width - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * padding)
        masks.append([1] * len(item["input_ids"]) + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "word_positions": [item["word_positions"] for item in encoded],
        "records": [example["record"] for example in examples],
    }


def move_batch(batch, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}
