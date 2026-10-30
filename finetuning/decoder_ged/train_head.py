#!/usr/bin/env python3
"""Train the separate Family-D linear GED head over decoder hidden states.

D1 freezes the base decoder. D2 trains a fresh LoRA with the head. D3a freezes
a correction LoRA; D3b continues it.  All modes use Variant B, repeat the
erroneous source in the perfect slot, and classify the hidden state after each
word's final subtoken.  GED CE is unweighted and is the only training loss.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ft_data  # noqa: E402
sys.path.insert(0, str(HERE))
from encoding import build_a_example, collate, move_batch  # noqa: E402
from head import LinearGEDHead, decoder_last_hidden, final_word_states  # noqa: E402
from metrics import multilingual_metrics  # noqa: E402
import runtime  # noqa: E402


def correction_adapter_meta(adapter):
    candidates = [Path(adapter) / "train_meta.json", Path(adapter).parent / "train_meta.json"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise ValueError("D3 requires train_meta.json beside the correction adapter")
    meta = json.loads(path.read_text())
    selection = str(meta.get("selection", "")).lower()
    if meta.get("kind") != "decoder-ged-family-a" or "pooled micro" not in selection:
        raise ValueError(
            "D3 initializer must be a Family-A adapter selected by strict synthetic "
            "multilingual pooled-micro F0.5, not an oracle authentic/English checkpoint")
    return meta, path


def make_model(args, device, langs):
    base = runtime.configure_base(args.model, device)
    provenance = None
    if args.mode == "d1":
        for parameter in base.parameters():
            parameter.requires_grad = False
        return base, provenance
    if args.mode == "d2":
        return runtime.fresh_lora(base, args.lora_r, args.lora_alpha, args.lora_dropout), provenance
    if not args.correction_adapter:
        raise ValueError(f"--correction-adapter is required for {args.mode}")
    meta, meta_path = correction_adapter_meta(args.correction_adapter)
    if not set(langs).issubset(meta.get("languages", [])):
        raise ValueError("D3 initializer does not cover every requested training language")
    from peft import PeftModel
    trainable = args.mode == "d3b"
    model = PeftModel.from_pretrained(base, args.correction_adapter, is_trainable=trainable)
    if not trainable:
        for parameter in model.parameters():
            parameter.requires_grad = False
    provenance = {"adapter": str(Path(args.correction_adapter).resolve()),
                  "meta": str(meta_path.resolve()), "run_name": meta.get("run_name"),
                  "selection": meta.get("selection"), "best_step": meta.get("best_step")}
    return model, provenance


def evaluate(model, head, examples, tokenizer, batch_size, device):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: batch)
    rows, loss_sum, correction_loss, batches = [], 0., 0., 0
    model.eval(); head.eval()
    with torch.no_grad():
        for examples_batch in loader:
            batch = move_batch(collate(examples_batch, pad_id, "corrupt"), device)
            last_hidden = decoder_last_hidden(model, batch["input_ids"], batch["attention_mask"])
            states, labels, languages, _, _ = final_word_states(
                last_hidden, batch["word_positions"], batch["records"])
            logits = head(states)
            loss_sum += float(F.cross_entropy(logits, labels))
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            rows.extend((lang, float(prob), int(label))
                        for lang, prob, label in zip(languages, probabilities, labels))
            clean = move_batch(collate(examples_batch, pad_id, "clean"), device)
            clean_output = model(input_ids=clean["input_ids"],
                                 attention_mask=clean["attention_mask"], labels=clean["labels"])
            correction_loss += float(clean_output.loss)
            batches += 1
    model.train(); head.train()
    report = multilingual_metrics(rows, error_when_low=False)
    report["ged_loss"] = loss_sum / max(batches, 1)
    report["correction_loss_diagnostic"] = correction_loss / max(batches, 1)
    return report


def save_head(head, out_dir, hidden_size, mode):
    path = Path(out_dir) / "ged_head.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"state_dict": head.state_dict(), "hidden_size": hidden_size,
                "classes": ["c", "i"], "mode": mode}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("d1", "d2", "d3a", "d3b"), required=True)
    parser.add_argument("--model", default=runtime.DEFAULT_MODEL)
    parser.add_argument("--correction-adapter")
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
    parser.add_argument("--patience", type=int, default=2,
                        help="epoch patience for frozen-decoder D1/D3a")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs:
        parser.error("require 1 <= --min-epochs <= --max-epochs")
    runtime.seed_everything(args.seed)
    ft_data.hf_login()
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    train_rows = runtime.mixed(runtime.load_records(
        langs, args.name, args.max_per_lang, args.corpus), args.seed)
    val_rows = runtime.mixed(runtime.load_records(
        langs, args.val_name, args.max_val_per_lang, args.corpus), args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def encode(rows, label):
        built = [build_a_example(row, tokenizer, args.max_length) for row in rows]
        kept = [example for example in built if example is not None]
        print(f"  {label}: {len(kept)} kept, {len(rows) - len(kept)} too long/unalignable")
        return kept

    train_examples, val_examples = encode(train_rows, "train"), encode(val_rows, "val")
    if not train_examples or not val_examples:
        raise SystemExit("no train or validation examples survived encoding")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, provenance = make_model(args, device, langs)
    model.to(device)
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None and hasattr(model.config, "text_config"):
        hidden_size = model.config.text_config.hidden_size
    head = LinearGEDHead(hidden_size).to(device)
    trainable = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda batch: batch)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.max_epochs
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    out_dir, run_name, timestamp = runtime.run_directory(
        args.model, langs, args.name, args.max_per_lang,
        args.tag or args.mode, args.corpus)
    history, best_score, best_step, step, stale_epochs = [], -1., -1, 0, 0

    def checkpoint(epoch):
        nonlocal best_score, best_step
        report = evaluate(model, head, val_examples, tokenizer, args.batch_size, device)
        report.update({"step": step, "epoch": epoch})
        history.append(report)
        score = report["pooled_micro"]["f0.5"]
        marker = ""
        if score > best_score:
            best_score, best_step = score, step
            save_head(head, out_dir, hidden_size, args.mode)
            if args.mode in ("d2", "d3b"):
                runtime.save_best_adapter(model, tokenizer, out_dir)
            marker = " <- saved"
        print(f"  step {step:5d} pooled_micro_F0.5={100*score:.2f} "
              f"GED_loss={report['ged_loss']:.4f}{marker}")

    checkpoint(0.)
    optimizer.zero_grad(set_to_none=True)
    for epoch_index in range(args.max_epochs):
        epoch_start_step, epoch_previous_best = step, best_score
        for batch_index, examples_batch in enumerate(loader):
            batch = move_batch(collate(examples_batch, pad_id, "corrupt"), device)
            if args.mode in ("d1", "d3a"):
                with torch.no_grad():
                    last_hidden = decoder_last_hidden(
                        model, batch["input_ids"], batch["attention_mask"])
            else:
                last_hidden = decoder_last_hidden(
                    model, batch["input_ids"], batch["attention_mask"])
            states, labels, _, _, _ = final_word_states(
                last_hidden, batch["word_positions"], batch["records"])
            loss = F.cross_entropy(head(states), labels)
            group_size = min(args.grad_accum,
                             len(loader) - (batch_index // args.grad_accum) * args.grad_accum)
            (loss / group_size).backward()
            boundary = (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(loader)
            if boundary:
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.eval_every == 0:
                    checkpoint(epoch_index + (batch_index + 1) / len(loader))
        if history[-1]["step"] != step:
            checkpoint(epoch_index + 1.)
        completed = epoch_index + 1
        stale_epochs = 0 if best_score > epoch_previous_best else stale_epochs + 1
        if completed >= args.min_epochs and completed < args.max_epochs:
            if args.mode in ("d1", "d3a"):
                if stale_epochs >= args.patience:
                    print("early stop: frozen-decoder head validation F0.5 is stale")
                    break
            else:
                epoch_steps = max(step - epoch_start_step, 1)
                if not (best_step >= epoch_start_step + .9 * epoch_steps
                        and best_score > epoch_previous_best):
                    print("adaptive stop: best checkpoint is not both improving and in final 10%")
                    break

    meta = {"kind": "decoder-ged-family-d", "mode": args.mode,
            "base_model": args.model, "run_name": run_name, "template": "variant_b",
            "representation": "hidden state after final subtoken of each source word",
            "head": "linear, two classes [correct, incorrect]", "class_weighting": "none",
            "loss": "GED cross entropy only", "languages": langs,
            "corpus": args.corpus,
            "correction_loss_role": "validation diagnostic only; never optimised or selected",
            "train_split": args.name, "val_split": args.val_name,
            "max_per_lang": args.max_per_lang, "n_train": len(train_examples),
            "n_val": len(val_examples), "correction_adapter_provenance": provenance,
            "selection": "per-language thresholds, maximum pooled micro incorrect-class F0.5",
            "primary_metric": "held-out multilingual token F0.5", "best_step": best_step,
            "best_pooled_micro_f0.5": best_score, "steps": step,
            "epochs_completed": completed,
            "optimizer": {"lr": args.lr, "batch_size": args.batch_size,
                          "grad_accum": args.grad_accum, "warmup_ratio": args.warmup_ratio},
            "lora": ({"r": args.lora_r, "alpha": args.lora_alpha,
                      "dropout": args.lora_dropout} if args.mode == "d2" else None),
            "seed": args.seed, "history": history, "timestamp": timestamp}
    runtime.save_meta(out_dir, meta)
    print(f"best pooled micro F0.5={100*best_score:.2f} at step {best_step} -> {out_dir}")


if __name__ == "__main__":
    main()
