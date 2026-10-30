#!/usr/bin/env python3
"""Train Family-E encoder masked-LM objectives for probability-threshold GED.

Family E is the encoder counterpart of the decoder's Family A. It fine-tunes
the masked-LM head's distribution directly rather than training a classifier,
so inference stays the existing mask-one-position / min-pool / threshold path
and the numbers remain comparable with `encoder-thresh` and with the decoder
families.

    slot  divergence-point contrast, both candidates from ONE softmax
    ul    -log(1 - P(word)) on labelled erroneous words
    keep  -log P(word) on unchanged words in the corrupt sentence
    mlm   cross-entropy on the clean sentence -- the anchor, always on

The embedding matrix and the output projection are frozen (see runtime.py):
every checkpoint here ties them, and leaving them trainable gives `ul` a
shortcut the decoder's LoRA never had.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ft_data  # noqa: E402
sys.path.insert(0, str(HERE))
import encoding  # noqa: E402
import objectives  # noqa: E402
import runtime  # noqa: E402

VARIANTS = {
    "e0":  {"slot": 0.,  "ul": 0.,  "keep": 0.},
    "e1a": {"slot": .5,  "ul": 0.,  "keep": 0.},
    "e1b": {"slot": 1.,  "ul": 0.,  "keep": 0.},
    "e2a": {"slot": 0.,  "ul": .5,  "keep": .5},
    "e2b": {"slot": 0.,  "ul": 1.,  "keep": .5},
    "e3":  {"slot": .5,  "ul": .5,  "keep": .5},
}


def evaluate(model, examples, tokenizer, token_budget, device):
    """Pooled-micro GED metrics at the word level, plus the separation gap."""
    mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
    rows, distributions = [], defaultdict(list)
    model.eval()
    with torch.no_grad():
        for group in encoding.iter_batches(examples, token_budget):
            batch = encoding.move(encoding.collate(group, mask_id, pad_id), device)
            log_probs = encoding.masked_log_probs(model, batch)
            word_scores, _ = encoding.word_min_logprobs(log_probs, batch)
            for scores, example in zip(word_scores, batch["examples"]):
                record = example["record"]
                for index, (score, label) in enumerate(zip(scores, record["labels"])):
                    if score is None:
                        continue
                    probability = float(score.exp())
                    rows.append((record["lang"], probability, int(label == "i")))
                    distributions["error" if label == "i" else "correct"].append(
                        probability)
    model.train()
    result = runtime.multilingual_metrics(rows, error_when_low=True)
    means = {key: (sum(v) / len(v) if v else None)
             for key, v in distributions.items()}
    result["probability_means"] = means
    result["separation"] = (
        (means["correct"] - means["error"])
        if means.get("correct") is not None and means.get("error") is not None
        else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--model", default=runtime.DEFAULT_MODEL)
    parser.add_argument("--langs", default="en,de,es,ro,ru")
    parser.add_argument("--corpus", default=ft_data.DEFAULT_CORPUS,
                        choices=list(ft_data.CORPORA))
    parser.add_argument("--name", default="train20k")
    parser.add_argument("--val-name", default="val5k")
    parser.add_argument("--max-per-lang", type=int)
    parser.add_argument("--max-val-per-lang", type=int, default=200)
    parser.add_argument("--token-budget", type=int, default=16384,
                        help="rows x padded width per forward, the memory knob; "
                             "a single oversized sentence still forms its own batch")
    parser.add_argument("--grad-accum", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=.06)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--margin", type=float, default=1.)
    parser.add_argument("--mlm-weight", type=float, default=1.)
    parser.add_argument("--mlm-rate", type=float, default=.15)
    parser.add_argument("--train-output-head", action="store_true",
                        help="unfreeze the tied embedding/projection (ablation only)")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs:
        parser.error("require 1 <= --min-epochs <= --max-epochs")
    weights = VARIANTS[args.variant]

    runtime.seed_everything(args.seed)
    rng = random.Random(args.seed)
    ft_data.hf_login()
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    print("loading canonical training data:", flush=True)
    train_rows = runtime.mixed(runtime.load_records(
        langs, args.name, args.max_per_lang, args.corpus), args.seed)
    print("loading canonical validation data:", flush=True)
    val_rows = runtime.mixed(runtime.load_records(
        langs, args.val_name, args.max_val_per_lang, args.corpus), args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.mask_token_id is None:
        raise SystemExit(f"{args.model} has no mask token; Family E needs an MLM")

    def encode(records, label):
        built = [encoding.encode_example(r, tokenizer, args.max_length)
                 for r in records]
        kept = [e for e in built if e is not None]
        copies = sum(e["n_copies"] for e in kept)
        print("  %s: %d kept, %d too long/unalignable, %d masked copies (%.1f/sentence)"
              % (label, len(kept), len(records) - len(kept), copies,
                 copies / max(len(kept), 1)), flush=True)
        return kept

    train_examples, val_examples = encode(train_rows, "train"), encode(val_rows, "val")
    if not train_examples or not val_examples:
        raise SystemExit("no train or validation examples survived encoding")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frozen = runtime.configure_base(
        args.model, device, freeze_output_head=not args.train_output_head)
    parameters = runtime.trainable_summary(model)
    print("frozen: %s" % (", ".join(frozen) or "nothing"), flush=True)
    print("trainable %d / %d params (%.1f%%)"
          % (parameters["trainable"], parameters["total"],
             100 * parameters["trainable_fraction"]), flush=True)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]

    batches_per_epoch = sum(1 for _ in encoding.iter_batches(
        train_examples, args.token_budget))
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.max_epochs
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    tag = args.tag or args.variant
    out_dir, run_name, timestamp = runtime.run_directory(
        args.model, langs, args.corpus, args.name, args.max_per_lang, tag)
    print("run: %s\n%d batches/epoch, %d optimiser steps/epoch"
          % (run_name, batches_per_epoch, steps_per_epoch), flush=True)

    mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
    history, tallies = [], defaultdict(int)
    best_score, best_step, step = -1., -1, 0

    def checkpoint(epoch):
        nonlocal best_score, best_step
        report = evaluate(model, val_examples, tokenizer, args.token_budget, device)
        report.update({"step": step, "epoch": epoch})
        history.append(report)
        score = report["pooled_micro"]["f0.5"]
        marker = ""
        if score > best_score:
            best_score, best_step = score, step
            runtime.save_best(model, tokenizer, out_dir)
            marker = " <- saved"
        separation = report["separation"]
        print("  step %5d  pooled_micro_F0.5=%.2f  separation=%s%s"
              % (step, 100 * score,
                 "n/a" if separation is None else "%.4f" % separation, marker),
              flush=True)

    checkpoint(0.)
    optimizer.zero_grad(set_to_none=True)
    completed = 0
    for epoch_index in range(args.max_epochs):
        epoch_start_step, epoch_previous_best = step, best_score
        batches = list(encoding.iter_batches(
            train_examples, args.token_budget, shuffle=True, rng=rng))
        for batch_index, group in enumerate(batches):
            components = {}
            reference = None
            if any(weights[name] for name in ("slot", "ul", "keep")):
                batch = encoding.move(
                    encoding.collate(group, mask_id, pad_id), device)
                log_probs = encoding.masked_log_probs(model, batch)
                word_scores, observed = encoding.word_min_logprobs(log_probs, batch)
                reference = observed.sum() * 0.
                if weights["slot"]:
                    value, counts = objectives.slot_loss(
                        log_probs, batch, margin=args.margin)
                    components["slot"] = value
                    for key, count in counts.items():
                        tallies["slot_pairs_" + key] += count
                if weights["ul"]:
                    value, skipped = objectives.unlikelihood_loss(word_scores, batch)
                    components["ul"] = value
                    tallies["ul_skipped_sentence_final"] += skipped
                if weights["keep"]:
                    components["keep"] = objectives.keep_loss(word_scores, batch)
                for example in group:
                    for key, count in example["slot_skips"].items():
                        tallies["slot_skipped_" + key] += count
            mlm_batch = encoding.build_mlm_batch(
                group, tokenizer, args.mlm_rate, rng, pad_id, mask_id)
            mlm_value = (encoding.mlm_loss(model, encoding.move(mlm_batch, device))
                         if mlm_batch is not None else None)
            if reference is None:
                reference = (mlm_value * 0. if mlm_value is not None
                             else torch.zeros((), device=device))
            total, _resolved = objectives.combine(components, weights, reference)
            if mlm_value is not None and args.mlm_weight:
                total = total + args.mlm_weight * mlm_value
            (total / args.grad_accum).backward()
            del components, mlm_value, total

            boundary = ((batch_index + 1) % args.grad_accum == 0
                        or batch_index + 1 == len(batches))
            if boundary:
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.eval_every == 0:
                    checkpoint(epoch_index + (batch_index + 1) / len(batches))
        if not history or history[-1]["step"] != step:
            checkpoint(epoch_index + 1.)
        completed = epoch_index + 1
        if args.min_epochs <= completed < args.max_epochs:
            epoch_steps = max(step - epoch_start_step, 1)
            late = best_step >= epoch_start_step + .9 * epoch_steps
            improving = best_score > epoch_previous_best
            if not (late and improving):
                print("adaptive stop: best checkpoint is not both improving "
                      "and in the final 10%", flush=True)
                break

    meta = {
        "kind": "encoder-ged-family-e", "variant": args.variant,
        "base_model": args.model, "run_name": run_name,
        "languages": langs, "corpus": args.corpus,
        "train_split": args.name, "val_split": args.val_name,
        "max_per_lang": args.max_per_lang,
        "n_train": len(train_examples), "n_val": len(val_examples),
        "weights": {**weights, "mlm": args.mlm_weight}, "margin": args.margin,
        "mlm_rate": args.mlm_rate,
        "frozen_parameters": frozen, "parameters": parameters,
        "slot_rule": ("first divergence index of the source and target sub-token "
                      "sequences; single-position mask; replacement edits only"),
        "word_probability": "minimum probability across sub-tokens",
        "normalisation": ("sub-tokens->word; words->edit; edits->example; "
                          "examples->batch; keep is a flat mean over c words"),
        "keep_rule": "plain mean over all c words (no before/after split)",
        "batching": ("sentence is the unit; greedy budget over rows x padded "
                     "width; every content sub-token masked once per sentence"),
        "gradient_checkpointing": True,
        "selection": ("per-language thresholds, maximum pooled micro "
                      "incorrect-class F0.5"),
        "primary_metric": "held-out multilingual token F0.5",
        "best_step": best_step, "best_pooled_micro_f0.5": best_score,
        "steps": step, "epochs_completed": completed, "tallies": dict(tallies),
        "optimizer": {"lr": args.lr, "token_budget": args.token_budget,
                      "grad_accum": args.grad_accum,
                      "warmup_ratio": args.warmup_ratio,
                      "max_length": args.max_length},
        "seed": args.seed, "history": history, "timestamp": timestamp,
    }
    runtime.save_meta(out_dir, meta)
    print("best pooled micro F0.5=%.2f at step %d -> %s"
          % (100 * best_score, best_step, out_dir), flush=True)


if __name__ == "__main__":
    main()
