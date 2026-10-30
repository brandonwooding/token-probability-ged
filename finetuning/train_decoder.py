#!/usr/bin/env python3
"""LoRA fine-tune a decoder as a GEC model, in the variant_b prompt structure.

Training examples are exactly the shape the detector uses at inference:

    <original_header><corrupted source><perfect_header><clean target>
    |------------------ context, masked -------------|--- loss here ---|

**The wording is imported from decoder-probability-thresh/prompts.py, never
copied.** If training framing and inference framing drift apart by even a
newline, the probabilities the detector reads are conditioned on a prefix the
model was never tuned for, and the whole method quietly degrades. One source of
truth, both sides.

Full fine-tuning a 12B model needs ~150GB of optimizer state, so this is LoRA
with gradient checkpointing -- ~24GB of bf16 weights plus activations fits one
48GB L40S. `peft` must be installed; the import fails loudly rather than
silently falling back to something that would not fit.

**Checkpoint selection is by separation, not validation loss.** The dominant
behaviour in C4_200M is copying (most tokens are unchanged), so a model can
drive LM loss down by learning to copy *everything* -- including errors -- which
destroys the very signal the detector reads. Validation loss cannot see that
happening; it goes down either way. So each eval measures the mean probability
the model assigns to clean target tokens versus corrupted source tokens placed
in the same 'perfect' slot, and keeps the checkpoint that separates them most.
Separation collapsing toward zero while loss keeps falling is the copy-collapse
failure, visible directly in the printed history.

    python train_decoder.py --langs en,de,es,ro,ru --name train10k --max-per-lang 4000
    python train_decoder.py --langs de --max-per-lang 1000 --epochs 1    # smoke test

Writes models/<run>/ (LoRA adapter + train_meta.json). Point the detector at it:
    llm-decoder-threshold.py --adapter finetuning/models/<run> --template variant_b
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import ft_data

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_MODEL = "google/gemma-4-12B"

# The prompt wording lives with the detector, and is imported rather than
# duplicated -- see the module docstring.
sys.path.insert(0, os.path.join(REPO_ROOT, "decoder-probability-thresh"))
import prompts  # noqa: E402


def build_example(row, tokenizer, max_length):
    """One pair -> (input_ids, labels), or None if it does not fit.

    Truncation is not an option here: cutting the prefix would corrupt the
    schema the model is being taught, and cutting the target would teach it to
    stop mid-sentence. Over-long pairs are dropped and counted instead.
    """
    prefix = prompts.variant_b(row["source"], row["language_folder"])
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]

    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    eos = tokenizer.eos_token_id

    input_ids = [bos] + prefix_ids + target_ids + [eos]
    if len(input_ids) > max_length:
        return None
    # -100 everywhere the model is only reading context. The target (and the
    # EOS that ends it) is the only thing it is scored on.
    labels = [-100] * (1 + len(prefix_ids)) + target_ids + [eos]
    return input_ids, labels


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attention = [], [], []
    for ids, labs in batch:
        padding = width - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        labels.append(labs + [-100] * padding)
        attention.append([1] * len(ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def mean_slot_probability(model, tokenizer, rows, device, max_length, which):
    """Mean probability of `which` ('target'|'source') teacher-forced into the
    perfect slot, plus the fraction of its tokens above 0.9.

    Scoring the *source* here reproduces exactly what the detector does at
    inference -- it forces the erroneous sentence into the slot the model was
    trained to fill with clean text. The gap between the two is the signal.
    """
    model.eval()
    total_prob, total_tokens, high = 0.0, 0, 0
    with torch.no_grad():
        for row in rows:
            prefix = prompts.variant_b(row["source"], row["language_folder"])
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            scored_ids = tokenizer(row[which], add_special_tokens=False)["input_ids"]
            bos = tokenizer.bos_token_id or tokenizer.eos_token_id
            ids = [bos] + prefix_ids + scored_ids
            if len(ids) > max_length or not scored_ids:
                continue
            tensor = torch.tensor([ids], device=device)
            logits = model(input_ids=tensor,
                           attention_mask=torch.ones_like(tensor)).logits[0]
            start = 1 + len(prefix_ids)
            for offset, token_id in enumerate(scored_ids):
                probs = torch.log_softmax(logits[start + offset - 1], dim=-1).exp()
                p = probs[token_id].item()
                total_prob += p
                total_tokens += 1
                high += 1 if p > 0.9 else 0
    model.train()
    if not total_tokens:
        return 0.0, 0.0
    return total_prob / total_tokens, high / total_tokens


def evaluate(model, loader, device):
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
            batches += 1
    model.train()
    return total / max(batches, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--langs", default="en,de,es,ro,ru")
    ap.add_argument("--name", default="train10k")
    ap.add_argument("--val-name", default="val5k")
    ap.add_argument("--max-per-lang", type=int, default=4000,
                    help="truncate each language to N pairs (scaling curve)")
    ap.add_argument("--max-val-per-lang", type=int, default=200)
    ap.add_argument("--probe-size", type=int, default=200,
                    help="val pairs used for the separation probe (it is O(n) forwards)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=250, help="optimizer steps between evals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise SystemExit(
            "peft is required for LoRA and is not installed.\n"
            "  pip install peft\n"
            "Full fine-tuning is not an alternative here: a 12B model needs "
            "~150GB of optimizer state, well beyond one L40S.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    langs = args.langs.split(",")
    ft_data.hf_login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = args.model.split("/")[-1]

    print("loading training data:")
    train_rows = ft_data.load_decoder(langs, args.name, args.max_per_lang, args.seed)
    print("loading validation data:")
    val_rows = ft_data.load_decoder(langs, args.val_name, args.max_val_per_lang, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def encode(rows, label):
        built = [build_example(r, tokenizer, args.max_length) for r in rows]
        kept = [b for b in built if b is not None]
        dropped = len(built) - len(kept)
        print("  %s: %d examples (%d dropped as longer than --max-length %d)"
              % (label, len(kept), dropped, args.max_length))
        return kept, dropped

    train_examples, n_dropped = encode(train_rows, "train")
    val_examples, _ = encode(val_rows, "val")
    if not train_examples:
        raise SystemExit("no training examples survived --max-length")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # checkpointing needs a grad-carrying input
    model.config.use_cache = False       # incompatible with gradient checkpointing

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    def make_loader(examples, shuffle):
        return DataLoader(examples, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=lambda b: collate(b, pad_id))

    train_loader = make_loader(train_examples, True)
    val_loader = make_loader(val_examples, False)

    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [model_id, "-".join(langs), args.name,
             str(args.max_per_lang or "all")] + ([args.tag] if args.tag else []) + [timestamp]
    run_name = "__".join(parts)
    out_dir = os.path.join(MODELS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print("run: %s\noptimizer steps: %d" % (run_name, total_steps))

    probe_rows = val_rows[:args.probe_size]
    best_separation, best_step, step = -1.0, -1, 0
    history = []

    def checkpoint(step):
        nonlocal best_separation, best_step
        loss = evaluate(model, val_loader, device)
        p_target, high_target = mean_slot_probability(
            model, tokenizer, probe_rows, device, args.max_length, "target")
        p_source, high_source = mean_slot_probability(
            model, tokenizer, probe_rows, device, args.max_length, "source")
        separation = p_target - p_source
        history.append({"step": step, "val_loss": loss,
                        "p_target": p_target, "p_source": p_source,
                        "separation": separation,
                        "frac_above_0.9_target": high_target,
                        "frac_above_0.9_source": high_source})
        marker = ""
        if separation > best_separation:
            best_separation, best_step = separation, step
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            marker = "  <- saved"
        print("  step %5d  loss=%.4f  p_tgt=%.3f p_src=%.3f  sep=%.4f  "
              "hi_tgt=%.2f hi_src=%.2f%s"
              % (step, loss, p_target, p_source, separation,
                 high_target, high_source, marker))

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for index, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if (index + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % args.eval_every == 0:
                    checkpoint(step)
    if step != best_step:
        checkpoint(step)

    meta = {
        "kind": "ft-decoder",
        "base_model": args.model,
        "model_id": model_id,
        "run_name": run_name,
        "template": "variant_b",
        "languages": langs,
        "train_split": args.name,
        "val_split": args.val_name,
        "max_per_lang": args.max_per_lang,
        "n_train": len(train_examples),
        "n_train_dropped_too_long": n_dropped,
        "n_val": len(val_examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "seed": args.seed,
        "total_steps": total_steps,
        "best_step": best_step,
        "best_separation": best_separation,
        "selection": "max mean P(clean target) - mean P(corrupted source) in the perfect slot",
        "history": history,
        "timestamp": timestamp,
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("best separation=%.4f at step %d -> %s" % (best_separation, best_step, out_dir))


if __name__ == "__main__":
    main()
