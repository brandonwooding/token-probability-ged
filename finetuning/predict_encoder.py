#!/usr/bin/env python3
"""Run a fine-tuned encoder classifier over a dataset -> canonical data.json.

**The emitted `probability` is P(`c`), not P(`i`).** That is what lets this
approach reuse `standard-eval/adapt_thresh.py` untouched: that adapter pools
sub-token probabilities by taking the MIN across a word and calls the word
incorrect when the pooled value falls below a threshold. Under P(`c`) both
halves of that are exactly right -- a word is suspicious if *any* of its pieces
looks wrong, and low confidence in correctness means predict `i`. Emitting
P(`i`) would invert the comparison and silently produce nonsense.

Tokenisation is `is_split_into_words=True` over the gold tokens, the same call
training makes, so `word_ids()` maps subtokens to gold words directly. The
char-span alignment the probability-threshold approaches need (they tokenise
detokenised text) is unnecessary here and would be a second way to get the same
answer -- which is a second way to be wrong.

    python predict_encoder.py --model models/<run> --dataset english/en_fce_dev
    python predict_encoder.py --model models/<run> --dataset english/en_c4200_eval5k

Writes finetuning/outputs/<stem>__<model>__<ts>/{data.json,meta.json}, which
standard-eval/run.py discovers on its own.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

import ft_data

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO_ROOT)
import paths  # noqa: E402

CORRECT_ID = ft_data.LABEL_TO_ID["c"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a train_encoder.py run dir")
    ap.add_argument("--dataset", required=True, help="dataset id, e.g. english/en_fce_dev")
    ap.add_argument("--model-id", default=None,
                    help="identifier used in the run name (default: model dir basename)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model)
    model_id = args.model_id or os.path.basename(model_dir.rstrip("/"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    with open(paths.sentences(args.dataset)) as f:
        units = json.load(f)

    all_sentences = []
    # A sentence longer than --max-length loses its tail, and adapt_thresh's
    # merged-token recovery would forward-fill the missing words from their
    # predecessor -- quietly, and with a probability that means nothing. Count
    # it here so a truncated run is visible rather than merely wrong.
    n_truncated = 0
    for start in range(0, len(units), args.batch_size):
        batch = units[start:start + args.batch_size]
        encoding = tokenizer([u["tokens"] for u in batch], is_split_into_words=True,
                             truncation=True, max_length=args.max_length,
                             padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[:, :, CORRECT_ID]

        for row in range(len(batch)):
            word_ids = encoding.word_ids(row)
            input_ids = encoding["input_ids"][row]
            by_word = {}
            for index, word_id in enumerate(word_ids):
                if word_id is None:      # special tokens and padding
                    continue
                by_word.setdefault(word_id, []).append({
                    "position": index,
                    "token": tokenizer.convert_ids_to_tokens(input_ids[index].item()),
                    "token_id": input_ids[index].item(),
                    "probability": probabilities[row, index].item(),
                })
            if len(by_word) != len(batch[row]["tokens"]):
                n_truncated += 1
            # One row per dataset word, holding its model sub-tokens. Unlike the
            # threshold approaches this aligns via the tokenizer's own word_ids
            # over pre-split input, so a sub-token belongs to exactly one word and
            # the lists are disjoint. A word past the truncation point keeps an
            # EMPTY subtokens list rather than vanishing, so row indices stay in
            # step with gold and standard-eval recovers it exactly as before.
            records = [{
                "word_id": word_id,
                "token": batch[row]["tokens"][word_id],
                "subtokens": by_word.get(word_id, []),
            } for word_id in range(len(batch[row]["tokens"]))]
            all_sentences.append(records)
        print("[%s] %d/%d" % (model_id, min(start + args.batch_size, len(units)), len(units)))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = paths.run_dir("ft-encoder", args.dataset, model_id, timestamp=timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "data.json", "w") as f:
        json.dump(all_sentences, f, indent=4)

    train_meta_path = os.path.join(model_dir, "train_meta.json")
    train_meta = None
    if os.path.exists(train_meta_path):
        with open(train_meta_path) as f:
            train_meta = json.load(f)
        train_meta.pop("history", None)   # the run meta records what, not the curve

    with open(run_dir / "meta.json", "w") as f:
        json.dump({
            "model_dir": model_dir,
            "model_id": model_id,
            "schema": "subtoken-nested-v1",
            "dataset": args.dataset,
            "language": paths.language(args.dataset),
            "probability_is": "P(correct) -- see module docstring",
            "timestamp": timestamp,
            "num_sentences": len(all_sentences),
            "max_length": args.max_length,
            "n_sentences_incompletely_covered": n_truncated,
            "train_meta": train_meta,
        }, f, indent=2)
    if n_truncated:
        print("WARNING: %d/%d sentences did not get a sub-token for every gold "
              "word (raise --max-length, currently %d)."
              % (n_truncated, len(all_sentences), args.max_length))
    print("Wrote %d sentences to %s." % (len(all_sentences), run_dir))


if __name__ == "__main__":
    main()
