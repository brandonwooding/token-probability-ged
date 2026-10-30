import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils import login_to_huggingface, choose_device

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths
from reconstruct_from_tsv import positions_for_words

parser = argparse.ArgumentParser(description="Encoder (MLM) probability GED over one dataset.")
parser.add_argument("--dataset", required=True, help="dataset id, e.g. english/en_fce_dev")
parser.add_argument("--model", default="answerdotai/ModernBERT-base")
parser.add_argument("--max-sentences", type=int, default=None,
                    help="Limit to the first N sentences (for smoke testing). Default: all.")
args = parser.parse_args()
model_name = args.model
model_id = model_name.split("/")[-1]

# MODEL
hf_token = login_to_huggingface(Path(".env"))
device = choose_device("auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

model.to(device)
model.eval()

# DATA
#
# Single source: reconstruct_from_tsv.py detokenises the MultiGED TSV into one
# natural sentence string per unit, each carrying its own tokens, c/i labels and
# by-construction char spans. We tokenise that text and align the model's
# sub-tokens back to those spans -- no second (JSONL) source to drift against.

with open(paths.sentences(args.dataset)) as f:
    sentences = json.load(f)

if args.max_sentences is not None:
    sentences = sentences[:args.max_sentences]

# LOOP
#
# data.json is a flat list of sentences (one per gold unit), each a list of
# per-token records. Essay/paragraph nesting is gone with sentence-level units.

all_sentences = []

for sentence_no, unit in enumerate(sentences):
    print(f"sentence {sentence_no + 1}/{len(sentences)}")
    text = unit["text"]
    tsv_spans = unit["spans"]

    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(device)
    offset_mapping = encoding["offset_mapping"][0]

    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    special_ids = set(tokenizer.all_special_ids)
    # Every model token overlapping each dataset word. Special tokens carry a
    # (0, 0) offset and so overlap nothing, which is what keeps them out.
    word_positions = positions_for_words(offset_mapping.tolist(), tsv_spans)

    subtoken_results = {}

    for idx, original_token_id in enumerate(input_ids):
        original_token_id = original_token_id.item()

        if original_token_id in special_ids:
            continue

        masked_input_ids = input_ids.clone()
        masked_input_ids[idx] = tokenizer.mask_token_id

        inputs = {
            "input_ids": masked_input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0)
        }

        with torch.no_grad():
            logits = model(**inputs).logits[0, idx]

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()

        target_prob = probs[original_token_id].item()

        rank = (probs > probs[original_token_id]).sum().item() + 1

        percentile = (probs < probs[original_token_id]).float().mean().item()

        # The peak of the distribution. Kept because it is the one quantity a
        # later token-level normaliser cannot reconstruct from the others: a low
        # probability only means "surprising" if the model was confident about
        # something else, and `rank` gives the observed token's position, not the
        # size of the gap.
        top_prob, top_id = probs.max(dim=-1)

        subtoken_results[idx] = {
            "position": idx,
            "token": tokens[idx],
            "token_id": original_token_id,
            "probability": target_prob,
            "max_probability": top_prob.item(),
            "top_token_id": top_id.item(),
            "entropy": entropy,
            "rank": rank,
            "percentile": percentile
        }

    # One row per DATASET word, holding every model sub-token that overlaps it.
    # Pooling is NOT done here -- it is min probability per word and lives in
    # standard-eval, so a token-level normaliser can be inserted before it
    # without re-running the GPU pass. See standard-eval/pooling.py.
    results = []
    for word_id, positions in enumerate(word_positions):
        subtokens = [subtoken_results[position] for position in positions
                     if position in subtoken_results]
        if not subtokens:
            raise ValueError(
                f"sentence {sentence_no}, word {word_id}: no scored model token"
            )
        results.append({
            "word_id": word_id,
            "token": unit["tokens"][word_id],
            "subtokens": subtokens,
        })

    all_sentences.append(results)

# OUTPUT: one run folder under this approach's outputs/, named by dataset+model+time.
run_dir = paths.run_dir("encoder-thresh", args.dataset, model_id)
run_dir.mkdir(parents=True, exist_ok=True)

with open(run_dir / "data.json", "w") as f:
    json.dump(all_sentences, f, indent=4)

with open(run_dir / "meta.json", "w") as f:
    json.dump({
        "model_id": model_id,
        "schema": "subtoken-nested-v1",
        "dataset": args.dataset,
        "num_sentences": len(all_sentences),
    }, f, indent=4)

print(f"Wrote {len(all_sentences)} sentences to {run_dir}.")