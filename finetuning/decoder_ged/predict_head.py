#!/usr/bin/env python3
"""Run a trained Family-D head and emit standard-eval P(correct) data.json."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402
sys.path.insert(0, str(HERE))
from encoding import build_d_example  # noqa: E402
from head import LinearGEDHead, decoder_last_hidden, final_word_states  # noqa: E402
import runtime  # noqa: E402


def load_system(run_dir, device):
    meta = json.loads((run_dir / "train_meta.json").read_text())
    model = runtime.configure_base(meta["base_model"], device)
    adapter = None
    if meta["mode"] in ("d2", "d3b"):
        adapter = run_dir
    elif meta["mode"] == "d3a":
        adapter = Path(meta["correction_adapter_provenance"]["adapter"])
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    checkpoint = torch.load(run_dir / "ged_head.pt", map_location="cpu", weights_only=True)
    head = LinearGEDHead(checkpoint["hidden_size"])
    head.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), head.to(device).eval(), meta


def predict_dataset(run_dir, dataset, model, head, meta, tokenizer, device,
                    max_units=None, max_length=256):
    """Generate one dataset while reusing the already-loaded D system."""
    units = json.loads(paths.sentences(dataset).read_text())
    if max_units is not None:
        units = units[:max_units]
    language_folder = paths.language(dataset)
    all_sentences = []
    with torch.no_grad():
        for unit_index, unit in enumerate(units):
            record = {"source": unit["text"], "target": unit["text"],
                      "lang": language_folder, "language_folder": language_folder,
                      "source_words": [{"text": token, "start": span[0], "end": span[1]}
                                       for token, span in zip(unit["tokens"], unit["spans"])],
                      "labels": unit.get("labels", ["c"] * len(unit["tokens"]))}
            example = build_d_example(record, tokenizer, max_length)
            if example is None:
                raise ValueError(
                    f"unit {unit_index} is longer than --max-length or cannot be aligned; "
                    "refusing to emit an apparently-correct empty prediction")
            encoded = example["corrupt"]
            ids = torch.tensor([encoded["input_ids"]], device=device)
            last_hidden = decoder_last_hidden(model, ids, torch.ones_like(ids))
            states, _, _, _, word_indices = final_word_states(
                last_hidden, [encoded["word_positions"]], [record])
            probabilities_correct = torch.softmax(head(states), dim=-1)[:, 0]
            all_sentences.append([
                {"word_id": word_index, "probability": float(probability)}
                for word_index, probability in zip(word_indices, probabilities_correct)])
            print(f"[{run_dir.name}] unit {unit_index + 1}/{len(units)}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = paths.run_dir("decoder-ged-head", dataset, run_dir.name, timestamp=timestamp)
    output.mkdir(parents=True, exist_ok=False)
    (output / "data.json").write_text(json.dumps(all_sentences, separators=(",", ":")))
    (output / "meta.json").write_text(json.dumps({
        "training_run": str(run_dir), "training_run_name": run_dir.name,
        "dataset": dataset, "mode": meta["mode"], "template": "variant_b",
        # Exempt from the nested sub-token schema: the head classifies an
        # already-pooled per-word state, so there are no sub-tokens to record.
        "schema": "word-level",
        "probability": "P(correct)", "num_units": len(units),
        "skipped_too_long_or_unalignable": 0, "timestamp": timestamp}, indent=2))
    print(f"wrote {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Family-D training run directory")
    parser.add_argument(
        "--dataset", nargs="+", required=True,
        help="one or more dataset ids; the model is loaded once and reused",
    )
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    run_dir = args.run.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head, meta = load_system(run_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(meta["base_model"])
    for dataset in args.dataset:
        predict_dataset(
            run_dir, dataset, model, head, meta, tokenizer, device,
            max_units=args.max_units, max_length=args.max_length,
        )


if __name__ == "__main__":
    main()
