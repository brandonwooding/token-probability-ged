"""Run naming, canonical rich-data loading, model setup, and safe saves."""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
DEFAULT_MODEL = "google/gemma-4-12B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(langs, name, limit, corpus=None):
    """Read the rich JSON emitted in the same mERRANT run as TSV/M2/pairs."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    import ft_data

    corpus = corpus or ft_data.DEFAULT_CORPUS
    per_lang = {}
    for lang in langs:
        path = Path(ft_data.rich_jsonl(lang, name, corpus))
        if not path.exists():
            raise SystemExit(
                f"missing canonical rich data: {path}\n"
                f"run datasets/synthetic-data/build/merrant/build.py "
                f"--corpus {corpus} --langs {lang}")
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit is not None and len(rows) >= limit:
                        break
        per_lang[lang] = rows
        print(f"  [{lang}] {len(rows):6d} canonical examples from {path.name}")
    return per_lang


def mixed(per_lang, seed):
    rows = [row for lang in sorted(per_lang) for row in per_lang[lang]]
    random.Random(seed).shuffle(rows)
    return rows


def run_directory(model, langs, split, size, tag, corpus=None):
    model_id = model.rstrip("/").split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [model_id, "-".join(langs)]
    if corpus:
        parts.append(corpus)
    parts += [split, str(size or "all"), tag, timestamp]
    name = "__".join(parts)
    path = MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=False)
    return path, name, timestamp


def configure_base(model_name, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def fresh_lora(model, rank, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                        task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    return get_peft_model(model, config)


def save_meta(out_dir, meta):
    path = Path(out_dir) / "train_meta.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(meta, indent=2))
    temporary.replace(path)


def save_best_adapter(model, tokenizer, out_dir):
    """Overwrite only files within this brand-new run's best checkpoint."""
    # Keep the adapter at the run root, as the existing decoder inference names
    # a system after Path(adapter).name. A generic best_adapter/ child would
    # collapse every A variant to the same signature during cross-tuning.
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
