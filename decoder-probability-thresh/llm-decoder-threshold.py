import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import login_to_huggingface, choose_device
from prompts import TEMPLATES

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_PATH = REPO_ROOT.parent / ".env"

sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402
from reconstruct_from_tsv import positions_for_words  # noqa: E402

DEFAULT_MODEL_NAME = "google/gemma-4-12B"


def run_decoder_threshold(dataset: str, model_name: str, template_name: str,
                          max_units: int | None = None, adapter: str | None = None):
    # A LoRA-adapted model is a *different system*, so it takes the adapter's
    # name in the run folder rather than the base model's. standard-eval pairs a
    # scoring run with its --tune-on counterpart by that name, and a fine-tuned
    # run tuned against a base-model threshold would be silently wrong.
    model_id = Path(adapter).name if adapter else model_name.split("/")[-1]
    build_prefix = TEMPLATES[template_name]
    # Prompt wording is per-language (see prompts.py); the dataset id's language
    # folder picks the block. Resolved up front so an unsupported language fails
    # before the model is loaded rather than on the first sentence.
    language = paths.language(dataset)
    build_prefix(".", language)

    # MODEL
    login_to_huggingface(ENV_PATH)
    device = choose_device("auto")

    # The tokenizer comes from the base model even when an adapter is loaded:
    # LoRA does not touch the vocabulary, and taking it from one place keeps the
    # offsets that tsv_spans are measured against identical across runs.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )

    if adapter:
        try:
            from peft import PeftModel
        except ImportError:
            raise SystemExit("--adapter needs peft installed: pip install peft")
        model = PeftModel.from_pretrained(model, adapter)
        print(f"loaded LoRA adapter {adapter}")

    model.to(device)
    model.eval()

    # DATA
    #
    # The sentences file is the single source of truth: each sentence carries its
    # own detokenised text, c/i labels and by-construction char spans (from
    # reconstruct_from_tsv.py). Driving the loop from it -- rather than zipping a
    # separate FCE JSON against it -- means there is no second source to drift.

    with open(paths.sentences(dataset)) as f:
        label_data = json.load(f)

    # Process only the first max_units sentences. Pass a small number (e.g. 5)
    # for a quick smoke test, or None to run the whole dataset.
    if max_units is not None:
        label_data = label_data[:max_units]

    special_ids = set(tokenizer.all_special_ids)

    # A causal model predicts a token only from what comes BEFORE it, so the very
    # first token has nothing to predict it from. We give it some left context by
    # sticking a beginning-of-sequence token on the front. It is a special token
    # like any other, so it is never itself scored.
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id

    # LOOP
    #
    # Flat list of sentences (one per gold unit), each a list of per-token
    # records. Essay/paragraph nesting is gone with sentence-level units.

    all_sentences = []

    for sentence_no, unit in enumerate(label_data):
        print(f"[{model_id}/{template_name}] sentence {sentence_no + 1}/{len(label_data)}")
        sentence = unit["text"]
        tsv_spans = unit["spans"]

        # The prompt is context only. We tokenize it separately from the
        # sentence so that (a) the sentence's char offsets stay relative to
        # the sentence itself -- which is what tsv_spans are measured against
        # -- and (b) we can score ONLY the sentence tokens, masking the
        # prompt out. Layout of the input we feed the model:
        #
        #   [BOS] [ ...prefix tokens... ] [ ...sentence tokens... ]
        #   never scored   never scored          scored
        prefix = build_prefix(sentence, language)
        prefix_ids = (
            tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        )

        encoding = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
        sentence_ids = encoding["input_ids"]
        word_positions = positions_for_words(encoding["offset_mapping"], tsv_spans)
        if any(not positions for positions in word_positions):
            raise ValueError(f"sentence {sentence_no}: a dataset word has no model token")

        input_ids = torch.tensor([bos_id] + prefix_ids + sentence_ids)
        attention_mask = torch.ones_like(input_ids)

        # Position at which the scored sentence tokens begin.
        sentence_start = 1 + len(prefix_ids)

        inputs = {
            "input_ids": input_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
        }

        # One forward pass gives a next-token prediction for every position
        # at once, because causal attention already only lets position idx
        # see positions 0..idx.
        with torch.no_grad():
            logits = model(**inputs).logits[0]

        # Sub-token strings for the scored region, so a stored record can be
        # read back without reloading the tokenizer.
        sentence_tokens = tokenizer.convert_ids_to_tokens(sentence_ids)

        subtoken_results = {}

        for sentence_idx, idx in enumerate(range(sentence_start, len(input_ids))):
            original_token_id = input_ids[idx].item()

            if original_token_id in special_ids:
                continue

            # logits[idx - 1] is the model's predicted distribution for the
            # token at position idx, based only on tokens before it (BOS +
            # prompt + earlier sentence tokens).
            log_probs = torch.log_softmax(logits[idx - 1], dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum().item()

            target_prob = probs[original_token_id].item()

            rank = (probs > probs[original_token_id]).sum().item() + 1

            percentile = (probs < probs[original_token_id]).float().mean().item()

            # The peak of the distribution. Kept because it is the one quantity
            # a later token-level normaliser cannot reconstruct from the others:
            # a low probability only means "surprising" if the model was
            # confident about something else, and `rank` gives the observed
            # token's position, not the size of the gap.
            top_prob, top_id = probs.max(dim=-1)

            subtoken_results[sentence_idx] = {
                "position": idx,
                "token": sentence_tokens[sentence_idx],
                "token_id": original_token_id,
                "probability": target_prob,
                "max_probability": top_prob.item(),
                "top_token_id": top_id.item(),
                "entropy": entropy,
                "rank": rank,
                "percentile": percentile
            }

        # One row per DATASET word, holding every model sub-token that overlaps
        # it. Pooling is NOT done here: it is min probability per word and lives
        # in standard-eval, so a token-level normaliser can be inserted before it
        # without re-running the GPU pass. See standard-eval/pooling.py.
        #
        # A sub-token straddling two words (gemma's 'wasn' over gold 'was' and
        # "n't") is listed under both. Its `position` is what identifies it as
        # shared, so no extra flag is needed to detect that after the fact.
        results = []
        for word_id, (dataset_token, positions) in enumerate(
            zip(unit["tokens"], word_positions)
        ):
            subtokens = [subtoken_results[position] for position in positions
                         if position in subtoken_results]
            if not subtokens:
                raise ValueError(
                    f"sentence {sentence_no}, word {word_id}: no scored model token"
                )
            results.append({
                "word_id": word_id,
                "token": dataset_token,
                "subtokens": subtokens,
            })

        all_sentences.append(results)

    # OUTPUT
    #
    # Each run gets its own self-describing folder under runs/, named by model,
    # template and timestamp, holding the scores plus a meta.json. evaluate-probs
    # then writes its metrics and plots back into the same folder.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = paths.run_dir("decoder-thresh", dataset, model_id,
                            tag=template_name, timestamp=timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "data.json", "w") as f:
        json.dump(all_sentences, f, indent=4)

    with open(run_dir / "meta.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "model_id": model_id,
            "adapter": adapter,
            "dataset": dataset,
            "language": language,
            "template": template_name,
            "schema": "subtoken-nested-v1",
            "timestamp": timestamp,
            "num_sentences": len(all_sentences),
        }, f, indent=4)

    print(f"Wrote {len(all_sentences)} sentences to {run_dir}.")

    del model
    del tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="dataset id, e.g. english/en_fce_dev (or bare stem).",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Model name/path to run. Repeat to queue multiple models.",
    )
    parser.add_argument(
        "--template",
        action="append",
        dest="templates",
        default=None,
        choices=list(TEMPLATES),
        help="Prompt template to use. Repeat to queue multiple. Default: none.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Limit to the first N sentences (for smoke testing). Default: all.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to a LoRA adapter from finetuning/train_decoder.py. The run "
             "is named after the adapter, not the base model.",
    )
    args = parser.parse_args()

    models = args.models or [DEFAULT_MODEL_NAME]
    templates = args.templates or ["none"]
    for model_name in models:
        for template_name in templates:
            run_decoder_threshold(args.dataset, model_name, template_name,
                                  max_units=args.max_sentences, adapter=args.adapter)


if __name__ == "__main__":
    main()
