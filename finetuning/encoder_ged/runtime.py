"""Run naming, model setup, output-head freezing, and safe saves.

Canonical rich-data loading and the selection metric are imported from
``decoder_ged`` rather than copied: both families must read the same mERRANT
JSONL and be selected by the same pooled-micro rule, and one definition is the
only way to guarantee that stays true.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
DEFAULT_MODEL = "jhu-clsp/mmBERT-base"



def _borrow(filename, *names):
    """Load a decoder_ged module by path, under a name that cannot collide.

    `import runtime` from here resolves to *this* file, and adding
    decoder_ged/ to sys.path would make which one you get depend on import
    order. Loading by explicit path keeps both files named runtime.py without
    either shadowing the other.
    """
    import importlib.util
    path = HERE.parent / "decoder_ged" / filename
    spec = importlib.util.spec_from_file_location(
        f"decoder_ged__{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [getattr(module, name) for name in names]


(multilingual_metrics,) = _borrow("metrics.py", "multilingual_metrics")
load_records, mixed, seed_everything, save_meta = _borrow(
    "runtime.py", "load_records", "mixed", "seed_everything", "save_meta")


def configure_base(model_name, device, freeze_output_head=True):
    """Load the MLM and freeze the tied embedding/decoder projection.

    Every one of these checkpoints ties its MLM decoder to the input embedding
    matrix (mmBERT and ModernBERT say so explicitly; XLM-R inherits HF's
    ``tie_word_embeddings=True`` default by omitting the key). That matrix is
    ~2/3 of the parameters, and leaving it trainable hands the unlikelihood
    term a shortcut: it can suppress a frequently-erroneous token globally by
    editing that token's row -- which is also its *input* representation
    everywhere it occurs -- instead of learning anything contextual. The
    decoder's Family A never had this exposure, because its LoRA touched only
    the q/k/v/o/gate/up/down projections.

    The head's output bias (mmBERT sets ``decoder_bias: true``) is the same
    shortcut in its purest form -- one free parameter per vocabulary entry --
    so it is frozen with the projection. Any dense/norm transform *before* the
    projection stays trainable: that is part of how the body shapes the
    distribution, which is exactly what these objectives are meant to move.
    """
    from transformers import AutoModelForMaskedLM
    # fp32 deliberately: RTX6000 is Turing (sm_75) with no native bf16, so a
    # bf16 request there is emulated -- tolerable for inference, not for
    # training.
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, dtype=torch.float32)
    # Not optional at this scale. One sentence expands to one masked copy per
    # sub-token, so a single 192-token sentence is a 192x192 activation block;
    # without checkpointing that alone does not fit a 24 GB card, let alone a
    # batch of them.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    frozen = []
    if freeze_output_head:
        embeddings = model.get_input_embeddings()
        output = model.get_output_embeddings()
        for module, label in ((embeddings, "input_embeddings"),
                              (output, "output_projection")):
            if module is None:
                continue
            for name, parameter in module.named_parameters():
                parameter.requires_grad = False
                frozen.append(f"{label}.{name}")
    model.to(device)
    return model, frozen


def trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total,
            "trainable_fraction": trainable / total if total else 0.}


def run_directory(model, langs, corpus, split, size, tag):
    model_id = model.rstrip("/").split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [model_id, "-".join(langs), corpus, split, str(size or "all"),
             tag, timestamp]
    name = "__".join(parts)
    path = MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=False)
    return path, name, timestamp


def save_best(model, tokenizer, out_dir):
    """Overwrite only files inside this brand-new run's directory."""
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
