"""Final-layer attention features for the encoder GED approach.

Deliberately a SEPARATE pass from llm-encoder-threshold.py rather than a few
extra lines inside it. Two reasons:

  1. Getting attention weights out at all requires attn_implementation="eager";
     the default (sdpa / flash) never materialises them. Loading the same model
     eagerly changes the accumulation order and would perturb the existing
     probabilities in their last few decimals. Keeping this separate leaves
     every archived data.json byte-identical.
  2. That pass masks one token at a time (N forwards per sentence). Attention is
     wanted on the *unmasked* sentence -- what the model actually does with the
     text as written -- which is one clean forward, and cheap.

The two outputs join on word_id, which is produced here by the same
get_custom_word_ids call on the same sentences file, so the alignment is
identical by construction rather than by luck.

WHAT IS WRITTEN (into its own run folder, alongside the probability runs)

  attention.json  per sentence, per word: the four scores from
                  attention_features.SCORE_NAMES, plus k (sub-tokens in the
                  word) and self_share (the diagnostic). Word-level, because
                  that is the unit the gold labels and the hypothesis live at.

  attention.npz   the raw per-head (H, T, T) matrix per sentence, float16.
                  ~22 MB for FCE-dev. This is the reason none of the feature
                  decisions are locked in: pooling, head choice, zeroing,
                  renormalisation and the diagonal can all be reworked offline
                  from these, with no GPU. Some of them genuinely cannot be
                  recovered from a per-token scalar -- a column sum has already
                  summed over the rows, so the row-side per-word reweighting and
                  the within-word block are unrecoverable once it is taken.

  meta.json       model, layer, and the settings that fix what the numbers mean.

Layer: the last one by default. For both ModernBERT-base and mmBERT-base that is
index 21, which is a *global* attention layer (global_attn_every_n_layers=3, and
21 % 3 == 0), so the matrix is not a sliding-window band.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from attention_features import SCORE_NAMES, word_scores
from utils import login_to_huggingface, choose_device, get_custom_word_ids

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402

DEFAULT_MODEL_NAME = "jhu-clsp/mmBERT-base"


def run_attention(dataset, model_name, layer=-1, max_units=None):
    model_id = model_name.split("/")[-1]

    # MODEL
    login_to_huggingface(REPO_ROOT / ".env")
    device = choose_device("auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(model_name)
    load_kwargs = {"attn_implementation": "eager"}
    # ModernBERT torch.compiles its layers by default, which silently drops the
    # attention weights on the floor. Only set it where the option exists.
    if hasattr(config, "reference_compile"):
        load_kwargs["reference_compile"] = False

    # AutoModel, not AutoModelForMaskedLM: the LM head is dead weight here.
    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model.to(device)
    model.eval()

    # DATA
    #
    # Same single source as the probability pass: reconstruct_from_tsv.py's
    # detokenised text plus by-construction char spans.
    with open(paths.sentences(dataset)) as f:
        sentences = json.load(f)

    if max_units is not None:
        sentences = sentences[:max_units]

    special_ids = set(tokenizer.all_special_ids)

    # LOOP
    all_sentences = []
    matrices = {}
    empty_units = []

    for sentence_no, unit in enumerate(sentences):
        if sentence_no % 100 == 0:
            print(f"[{model_id}] sentence {sentence_no + 1}/{len(sentences)}")

        text = unit["text"]
        tsv_spans = unit["spans"]

        encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = encoding["offset_mapping"][0]
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        word_ids = get_custom_word_ids(input_ids, offset_mapping, tsv_spans, special_ids)

        # One clean forward over the sentence exactly as written -- nothing
        # masked, nothing prompted.
        inputs = {
            "input_ids": input_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
        }
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        if not outputs.attentions or outputs.attentions[layer] is None:
            raise RuntimeError(
                f"{model_name} returned no attention weights -- the model was not "
                "loaded with attn_implementation='eager'."
            )

        # (H, T, T) for the chosen layer.
        heads = outputs.attentions[layer][0].float().cpu().numpy()

        # Head-mean first, then every word-level operation on that single
        # matrix. For token_mean and word_vote the order is irrelevant (both are
        # linear), but renormalisation is not, so the order has to be fixed --
        # the per-head matrices below keep the alternative available.
        head_mean = heads.mean(axis=0)

        scores = word_scores(head_mean, word_ids)
        if not scores["word_ids"]:
            empty_units.append(sentence_no)

        all_sentences.append(scores)
        matrices[str(sentence_no)] = heads.astype(np.float16)

    # OUTPUT
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = paths.run_dir("encoder-thresh", dataset, model_id,
                            tag="attention", timestamp=timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "attention.json", "w") as f:
        json.dump(all_sentences, f, indent=4)

    np.savez_compressed(run_dir / "attention.npz", **matrices)

    n_layers = len(outputs.attentions)
    with open(run_dir / "meta.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "model_id": model_id,
            "dataset": dataset,
            "timestamp": timestamp,
            "num_sentences": len(all_sentences),
            "layer": layer,
            "layer_index": range(n_layers)[layer],
            "num_layers": n_layers,
            "num_heads": int(heads.shape[0]),
            "attn_implementation": "eager",
            "head_aggregation": "mean over heads, before word-level pooling",
            "scores": list(SCORE_NAMES),
            "pass": "clean unmasked forward",
            "units_with_no_aligned_word": empty_units,
        }, f, indent=4)

    print(f"Wrote {len(all_sentences)} sentences to {run_dir}.")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="dataset id, e.g. english/en_fce_dev.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"encoder to run. Default: {DEFAULT_MODEL_NAME}.")
    parser.add_argument("--layer", type=int, default=-1,
                        help="attention layer index; -1 (default) is the last.")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="limit to the first N sentences (smoke test).")
    args = parser.parse_args()

    run_attention(args.dataset, args.model, layer=args.layer,
                  max_units=args.max_sentences)


if __name__ == "__main__":
    main()
