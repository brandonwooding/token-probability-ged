#!/usr/bin/env python3
"""Train isolated Family-A decoder objectives for probability-threshold GED.

Family A updates use two teacher-forced passes through the same Variant-B
prompt: the clean target pass learns correction CE; the corrupt-source pass
learns low probability on erroneous words and/or preservation of correct
words.  The correction-only A0 baseline uses the same training and validation
pipeline but omits the corrupt-source training pass.  Clean word scores are
detached before auxiliary passes to keep 12B training within one accelerator;
correction CE still provides their positive training signal.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ft_data  # noqa: E402
sys.path.insert(0, str(HERE))
from encoding import build_a_example, collate, move_batch  # noqa: E402
from metrics import multilingual_metrics  # noqa: E402
from objectives import auxiliary_losses, word_min_logprobs  # noqa: E402
import runtime  # noqa: E402


VARIANTS = {
    "a0": {"contrast": 0., "unlikelihood": 0., "keep": 0.},
    "a1a": {"contrast": .5, "unlikelihood": 0., "keep": 0.},
    "a1b": {"contrast": 1., "unlikelihood": 0., "keep": 0.},
    "a1-anchor": {"contrast": .5, "unlikelihood": 0., "keep": 0.},
    "a2-ul": {"contrast": 0., "unlikelihood": .5, "keep": 0.},
    "a2-keep": {"contrast": 0., "unlikelihood": 0., "keep": .5},
    "a2a": {"contrast": 0., "unlikelihood": .5, "keep": .5},
    "a2b": {"contrast": 0., "unlikelihood": 1., "keep": .5},
    "a3a": {"contrast": .5, "unlikelihood": .5, "keep": .5},
    "a3b": {"contrast": .5, "unlikelihood": 1., "keep": .5},
    "a3-anchor": {"contrast": .5, "unlikelihood": .5, "keep": .5},
}


def evaluate(model, examples, tokenizer, batch_size, device):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: batch)
    rows, distributions = [], defaultdict(list)
    correction_loss, batches = 0., 0
    model.eval()
    with torch.no_grad():
        for examples_batch in loader:
            clean = move_batch(collate(examples_batch, pad_id, "clean"), device)
            corrupt = move_batch(collate(examples_batch, pad_id, "corrupt"), device)
            clean_out = model(input_ids=clean["input_ids"], attention_mask=clean["attention_mask"],
                              labels=clean["labels"])
            corrupt_out = model(input_ids=corrupt["input_ids"],
                                attention_mask=corrupt["attention_mask"])
            target_scores = word_min_logprobs(clean_out.logits, clean["input_ids"],
                                               clean["word_positions"])
            source_scores = word_min_logprobs(corrupt_out.logits, corrupt["input_ids"],
                                               corrupt["word_positions"])
            correction_loss += float(clean_out.loss)
            batches += 1
            for src, tgt, record in zip(source_scores, target_scores, clean["records"]):
                first_error = next((i for i, y in enumerate(record["labels"]) if y == "i"), None)
                for index, (score, label) in enumerate(zip(src, record["labels"])):
                    probability = float(score.exp())
                    rows.append((record["lang"], probability, int(label == "i")))
                    key = ("error" if label == "i" else
                           "correct_before" if first_error is not None and index < first_error
                           else "correct_after")
                    distributions[key].append(probability)
                distributions["target"].extend(float(score.exp()) for score in tgt if score is not None)
    model.train()
    result = multilingual_metrics(rows, error_when_low=True)
    result["correction_loss"] = correction_loss / max(batches, 1)
    result["probability_means"] = {
        key: sum(values) / len(values) if values else None for key, values in distributions.items()}
    result["global_separation"] = (
        result["probability_means"].get("target", 0.)
        - sum(distributions.get("error", [])) / max(len(distributions.get("error", [])), 1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--model", default=runtime.DEFAULT_MODEL)
    parser.add_argument("--langs", default="en,de,es,ro,ru")
    parser.add_argument("--corpus", default=ft_data.DEFAULT_CORPUS,
                        choices=list(ft_data.CORPORA),
                        help="synthetic corpus the split stems belong to")
    parser.add_argument("--name", default="train10k")
    parser.add_argument("--val-name", default="val5k")
    parser.add_argument("--max-per-lang", type=int)
    parser.add_argument("--max-val-per-lang", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=.03)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--margin", type=float, default=1.)
    parser.add_argument("--empty-edit-contrast", choices=("skip", "anchor-next"), default=None)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs:
        parser.error("require 1 <= --min-epochs <= --max-epochs")
    weights = VARIANTS[args.variant]
    has_auxiliary_loss = any(weight != 0. for weight in weights.values())
    empty_policy = args.empty_edit_contrast or (
        "anchor-next" if args.variant.endswith("-anchor") else "skip")

    runtime.seed_everything(args.seed)
    ft_data.hf_login()
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    print("loading prepared training data:")
    train_rows = runtime.mixed(runtime.load_records(
        langs, args.name, args.max_per_lang, args.corpus), args.seed)
    print("loading prepared validation data:")
    val_rows = runtime.mixed(runtime.load_records(
        langs, args.val_name, args.max_val_per_lang, args.corpus), args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def encode(rows, label):
        built = [build_a_example(row, tokenizer, args.max_length) for row in rows]
        kept = [example for example in built if example is not None]
        print(f"  {label}: {len(kept)} kept, {len(rows) - len(kept)} too long/unalignable")
        return kept

    train_examples, val_examples = encode(train_rows, "train"), encode(val_rows, "val")
    if not train_examples or not val_examples:
        raise SystemExit("no train or validation examples survived encoding")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = runtime.fresh_lora(runtime.configure_base(args.model, device),
                               args.lora_r, args.lora_alpha, args.lora_dropout).to(device)
    model.print_trainable_parameters()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda batch: batch)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.max_epochs
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    tag = args.tag or f"{args.variant}-{empty_policy}"
    out_dir, run_name, timestamp = runtime.run_directory(
        args.model, langs, args.name, args.max_per_lang, tag, args.corpus)
    history, aggregate_skips = [], defaultdict(int)
    best_score, best_step, step = -1., -1, 0

    def checkpoint(epoch):
        nonlocal best_score, best_step
        report = evaluate(model, val_examples, tokenizer, args.batch_size, device)
        report.update({"step": step, "epoch": epoch})
        history.append(report)
        score = report["pooled_micro"]["f0.5"]
        marker = ""
        if score > best_score:
            best_score, best_step = score, step
            runtime.save_best_adapter(model, tokenizer, out_dir)
            marker = " <- saved"
        print(f"  step {step:5d} pooled_micro_F0.5={100*score:.2f} "
              f"correction_loss={report['correction_loss']:.4f}{marker}")

    checkpoint(0.)
    optimizer.zero_grad(set_to_none=True)
    for epoch_index in range(args.max_epochs):
        epoch_start_step = step
        epoch_previous_best = best_score
        for batch_index, examples_batch in enumerate(loader):
            group_size = min(args.grad_accum,
                             len(loader) - (batch_index // args.grad_accum) * args.grad_accum)
            clean = move_batch(collate(examples_batch, pad_id, "clean"), device)
            clean_out = model(input_ids=clean["input_ids"], attention_mask=clean["attention_mask"],
                              labels=clean["labels"])
            if has_auxiliary_loss:
                target_scores = [[value.detach() if value is not None else None for value in row]
                                 for row in word_min_logprobs(
                                     clean_out.logits, clean["input_ids"],
                                     clean["word_positions"])]
            (clean_out.loss / group_size).backward()
            del clean_out

            if has_auxiliary_loss:
                corrupt = move_batch(collate(examples_batch, pad_id, "corrupt"), device)
                corrupt_out = model(input_ids=corrupt["input_ids"],
                                    attention_mask=corrupt["attention_mask"])
                source_scores = word_min_logprobs(corrupt_out.logits, corrupt["input_ids"],
                                                   corrupt["word_positions"])
                aux, _components, skips = auxiliary_losses(
                    source_scores, target_scores, corrupt["records"], margin=args.margin,
                    empty_policy=empty_policy, contrast_weight=weights["contrast"],
                    unlikelihood_weight=weights["unlikelihood"], keep_weight=weights["keep"])
                (aux / group_size).backward()
                del corrupt_out, source_scores, aux
                for key, value in skips.items():
                    aggregate_skips[key] += value
            boundary = (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(loader)
            if boundary:
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.eval_every == 0:
                    checkpoint(epoch_index + (batch_index + 1) / len(loader))
        if history[-1]["step"] != step:
            checkpoint(epoch_index + 1.)
        completed = epoch_index + 1
        if completed >= args.min_epochs and completed < args.max_epochs:
            epoch_steps = max(step - epoch_start_step, 1)
            best_is_late = best_step >= epoch_start_step + .9 * epoch_steps
            still_improving = best_score > epoch_previous_best
            if not (best_is_late and still_improving):
                print("adaptive stop: best checkpoint is not both improving and in final 10%")
                break

    meta = {"kind": "decoder-ged-family-a", "variant": args.variant,
            "base_model": args.model, "run_name": run_name, "template": "variant_b",
            "languages": langs, "corpus": args.corpus,
            "train_split": args.name, "val_split": args.val_name,
            "max_per_lang": args.max_per_lang, "n_train": len(train_examples),
            "n_val": len(val_examples), "weights": weights, "margin": args.margin,
            "empty_edit_contrast": empty_policy,
            "word_probability": "minimum probability across subtokens",
            "normalisation": "subtokens->word; words->edit; edits->example; examples->batch",
            "clean_target_aux_gradient": (
                "detached; correction CE supplies positive pressure"
                if has_auxiliary_loss else "not applicable; correction-only training"),
            "selection": "per-language thresholds, maximum pooled micro incorrect-class F0.5",
            "primary_metric": "held-out multilingual token F0.5",
            "correction_metric_role": "diagnostic only", "best_step": best_step,
            "best_pooled_micro_f0.5": best_score, "steps": step,
            "epochs_completed": completed, "skips": dict(aggregate_skips),
            "optimizer": {"lr": args.lr, "batch_size": args.batch_size,
                          "grad_accum": args.grad_accum, "warmup_ratio": args.warmup_ratio},
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                     "dropout": args.lora_dropout}, "seed": args.seed,
            "history": history, "timestamp": timestamp}
    runtime.save_meta(out_dir, meta)
    print(f"best pooled micro F0.5={100*best_score:.2f} at step {best_step} -> {out_dir}")


if __name__ == "__main__":
    main()
