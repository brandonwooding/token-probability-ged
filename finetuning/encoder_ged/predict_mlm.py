#!/usr/bin/env python3
"""Score a dataset with a Family-E encoder -> canonical data.json.

Emits P(observed sub-token) under a single mask, exactly the quantity
`secondary/encoder-probability-thresh/llm-encoder-threshold.py` emits for an untrained
encoder, so a Family-E run and an `encoder-thresh` baseline are directly
comparable and `standard-eval/adapt_thresh.py` needs no change: low probability
means suspicious, and pooling is MIN over a word's sub-tokens.

Unlike that script this batches masked copies across sentences rather than
running one forward per position at batch size 1, and gathers the hidden state
before the LM head so a 256k-row vocabulary tensor is never materialised for a
whole sequence.

    python predict_mlm.py --run models/<E-run> --dataset english/en_fce_dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402
from reconstruct_from_tsv import positions_for_words  # noqa: E402
sys.path.insert(0, str(HERE))
from encoding import apply_head  # noqa: E402


def plan(unit, tokenizer, max_length):
    encoded = tokenizer(unit["text"], add_special_tokens=True, truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
    word_positions = positions_for_words(encoded["offset_mapping"], unit["spans"])
    positions = sorted({p for row in word_positions for p in row})
    return encoded["input_ids"], word_positions, positions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="a Family-E model directory")
    ap.add_argument("--dataset", required=True, help="e.g. english/en_fce_dev")
    ap.add_argument("--model-id", default=None,
                    help="short name for the run folder; defaults to the run's "
                         "basename, which is long enough to be truncated in the "
                         "standard-eval comparison table")
    ap.add_argument("--max-sentences", type=int)
    ap.add_argument("--row-budget", type=int, default=256,
                    help="masked rows per inference forward (no backward pass "
                         "here, so rows rather than token-positions is enough)")
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()

    run = Path(args.run)
    meta_path = run / "train_meta.json"
    train_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model_id = args.model_id or train_meta.get("variant") or run.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(run)
    model = AutoModelForMaskedLM.from_pretrained(run, dtype=torch.float32)
    model.to(device).eval()
    mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id

    with open(paths.sentences(args.dataset)) as f:
        units = json.load(f)
    if args.max_sentences:
        units = units[:args.max_sentences]

    plans = [plan(unit, tokenizer, args.max_length) for unit in units]
    # (unit index, position) pairs, batched so memory is flat across lengths
    queue = [(u, p) for u, (_, _, positions) in enumerate(plans) for p in positions]
    scored = {}
    with torch.no_grad():
        for start in range(0, len(queue), args.row_budget):
            chunk = queue[start:start + args.row_budget]
            rows, observed, mask_index = [], [], []
            for unit_index, position in chunk:
                ids = list(plans[unit_index][0])
                observed.append(ids[position])
                ids[position] = mask_id
                rows.append(ids)
                mask_index.append(position)
            width = max(len(r) for r in rows)
            input_ids = torch.tensor(
                [r + [pad_id] * (width - len(r)) for r in rows], device=device)
            attention = torch.tensor(
                [[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device)
            hidden = model.base_model(
                input_ids=input_ids, attention_mask=attention).last_hidden_state
            picked = hidden[torch.arange(len(rows), device=device),
                            torch.tensor(mask_index, device=device)]
            log_probs = torch.log_softmax(apply_head(model, picked).float(), dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            top_prob, top_id = probs.max(dim=-1)
            target = torch.tensor(observed, device=device)
            own = probs[torch.arange(len(rows), device=device), target]
            rank = (probs > own.unsqueeze(1)).sum(dim=-1) + 1
            percentile = (probs < own.unsqueeze(1)).float().mean(dim=-1)
            for offset, (unit_index, position) in enumerate(chunk):
                scored[(unit_index, position)] = {
                    "position": position,
                    "token": tokenizer.convert_ids_to_tokens(observed[offset]),
                    "token_id": observed[offset],
                    "probability": own[offset].item(),
                    "max_probability": top_prob[offset].item(),
                    "top_token_id": int(top_id[offset]),
                    "entropy": entropy[offset].item(),
                    "rank": int(rank[offset]),
                    "percentile": percentile[offset].item(),
                }
            print("scored %d/%d masked positions" % (min(start + args.row_budget,
                  len(queue)), len(queue)), flush=True)

    all_sentences = []
    for unit_index, (unit, (_, word_positions, _)) in enumerate(zip(units, plans)):
        results = []
        for word_id, positions in enumerate(word_positions):
            subtokens = [scored[(unit_index, p)] for p in positions
                         if (unit_index, p) in scored]
            if not subtokens:
                raise ValueError(
                    f"sentence {unit_index}, word {word_id}: no scored model token")
            results.append({"word_id": word_id, "token": unit["tokens"][word_id],
                            "subtokens": subtokens})
        all_sentences.append(results)

    out = paths.run_dir("encoder-ged-mlm", args.dataset, model_id)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "data.json", "w") as f:
        json.dump(all_sentences, f, indent=4)
    with open(out / "meta.json", "w") as f:
        json.dump({"model_id": model_id, "schema": "subtoken-nested-v1",
                   "dataset": args.dataset, "num_sentences": len(all_sentences),
                   "run": str(run), "variant": train_meta.get("variant"),
                   "base_model": train_meta.get("base_model"),
                   "corpus": train_meta.get("corpus"),
                   "probability_is": "P(observed sub-token) under a single mask"},
                  f, indent=4)
    print("wrote %d sentences to %s" % (len(all_sentences), out), flush=True)


if __name__ == "__main__":
    main()
