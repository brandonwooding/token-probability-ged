#!/usr/bin/env python3
"""Fine-tune a multilingual encoder as a token-level GED classifier.

Supervised token classification over the C4_200M synthetic labels: two classes,
`c` (0) and `i` (1), one label per *word*, carried by that word's first
subtoken. Swap in real MultiGED training TSVs later by pointing --name at a
split cut into the same layout -- nothing here knows the labels are synthetic.

**No class weighting, deliberately.** The `i` rate ranges from 7.9% (ru) to
22.3% (en) across the training languages, which is exactly the kind of prior
skew class weights are usually reached for. It is not needed here because the
operating point is not the argmax: inference emits a calibrated P(`c`) per word
and `standard-eval/adapt_thresh.py` sweeps a threshold over it downstream. A
class weight would move the same decision boundary twice, once here and once
there, and make the threshold that adapt_thresh selects uninterpretable.

Checkpoint selection is by F0.5 on the validation mix at the argmax boundary --
a proxy, not the reported metric, but a stable one for picking a step.

    python train_encoder.py --langs en,de,es,ro,ru --name train20k
    python train_encoder.py --langs de --name train10k --epochs 2

Writes models/<run>/ (weights + tokenizer + train_meta.json), ready for
predict_encoder.py.
"""
import argparse
import json
import os
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import ft_data

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_MODEL = "jhu-clsp/mmBERT-base"


def collate(batch, tokenizer, max_length):
    encoding = tokenizer([ex["tokens"] for ex in batch], is_split_into_words=True,
                         truncation=True, max_length=max_length, padding=True,
                         return_tensors="pt")
    labels = [ft_data.align_labels_to_subtokens(encoding, batch[i]["labels"], i)
              for i in range(len(batch))]
    encoding["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoding


def evaluate(model, loader, device):
    """Val loss plus word-level P/R/F0.5 at the argmax boundary (positive = i)."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            n_batches += 1
            predictions = out.logits.argmax(-1)
            gold = batch["labels"]
            scored = gold != -100
            tp += ((predictions == 1) & (gold == 1) & scored).sum().item()
            fp += ((predictions == 1) & (gold == 0) & scored).sum().item()
            fn += ((predictions == 0) & (gold == 1) & scored).sum().item()
    model.train()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 0.25 * precision + recall
    f05 = 1.25 * precision * recall / denominator if denominator else 0.0
    return total_loss / max(n_batches, 1), precision, recall, f05


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--langs", default="en,de,es,ro,ru")
    ap.add_argument("--corpus", default=ft_data.DEFAULT_CORPUS,
                    choices=list(ft_data.CORPORA),
                    help="synthetic corpus the split stems belong to")
    ap.add_argument("--name", default="train20k", help="training split name")
    ap.add_argument("--val-name", default="val5k", help="validation split name")
    ap.add_argument("--max-per-lang", type=int, default=None,
                    help="truncate each language to N sentences (scaling curve)")
    ap.add_argument("--max-val-per-lang", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--eval-every", type=int, default=500, help="steps between evals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="extra label in the run name")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    langs = args.langs.split(",")
    ft_data.hf_login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = args.model.split("/")[-1]

    print("loading training data:")
    train_rows = ft_data.load_encoder(langs, args.name, args.max_per_lang,
                                      args.seed, args.corpus)
    print("loading validation data:")
    val_rows = ft_data.load_encoder(langs, args.val_name, args.max_val_per_lang,
                                    args.seed, args.corpus)
    print("  total: %d train / %d val" % (len(train_rows), len(val_rows)))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=2,
        id2label={0: "c", 1: "i"}, label2id={"c": 0, "i": 1})
    model.to(device)
    model.train()

    def make_loader(rows, shuffle):
        return DataLoader(rows, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=lambda b: collate(b, tokenizer, args.max_length))

    train_loader = make_loader(train_rows, True)
    val_loader = make_loader(val_rows, False)

    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # corpus is in the name because c4200 and mlcheck write into the same
    # language folders and differ only by stem prefix -- without it the two
    # runs of one config are indistinguishable on disk.
    parts = [model_id, "-".join(langs), args.corpus, args.name,
             str(args.max_per_lang or "all")] + ([args.tag] if args.tag else []) + [timestamp]
    run_name = "__".join(parts)
    out_dir = os.path.join(MODELS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print("run: %s\ntotal steps: %d" % (run_name, total_steps))

    best_f05, best_step, step = -1.0, -1, 0
    history = []

    def checkpoint(step):
        """Save on improvement only -- a base encoder is ~1.2GB and the scaling
        curve runs many configs, so keeping every eval point would fill the
        quota for no benefit."""
        nonlocal best_f05, best_step
        loss, precision, recall, f05 = evaluate(model, val_loader, device)
        history.append({"step": step, "val_loss": loss, "precision": precision,
                        "recall": recall, "f0.5": f05})
        marker = ""
        if f05 > best_f05:
            best_f05, best_step = f05, step
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            marker = "  <- saved"
        print("  step %5d  val_loss=%.4f  P=%.3f R=%.3f F0.5=%.4f%s"
              % (step, loss, precision, recall, f05, marker))

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % args.eval_every == 0:
                checkpoint(step)
    if step != best_step:
        checkpoint(step)

    meta = {
        "kind": "ft-encoder",
        "base_model": args.model,
        "model_id": model_id,
        "run_name": run_name,
        "languages": langs,
        "corpus": args.corpus,
        "train_split": args.name,
        "val_split": args.val_name,
        "max_per_lang": args.max_per_lang,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "total_steps": total_steps,
        "best_step": best_step,
        "best_val_f0.5": best_f05,
        "history": history,
        "timestamp": timestamp,
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("best F0.5=%.4f at step %d -> %s" % (best_f05, best_step, out_dir))


if __name__ == "__main__":
    main()
