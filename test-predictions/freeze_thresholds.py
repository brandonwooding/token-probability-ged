#!/usr/bin/env python3
"""Freeze per-corpus dev thresholds for a configured submission system."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "standard-eval"))
from adapt_thresh import collect_aligned, select_threshold  # noqa: E402
from common import load_json, prf  # noqa: E402


def discover_dev_data(dev_dataset: str, config: dict, entry: dict):
    approach = config["approach"]
    checkpoint = config["checkpoint"]
    tag = entry.get("tag", config.get("tag"))
    stem = paths.stem(dev_dataset)
    prefix = f"{stem}__{checkpoint}"
    if tag:
        prefix += f"__{tag}"
    root = paths.APPROACH_DIRS[approach] / "outputs"
    candidates = sorted(
        run / "data.json" for run in root.glob(f"{prefix}__*")
        if (run / "data.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no dev data for {dev_dataset}, checkpoint={checkpoint}, tag={tag}"
        )
    return candidates[-1]


def metrics_at_threshold(aligned, threshold):
    tp = fp = fn = 0
    for _, _, labels, probabilities in aligned:
        for gold, probability in zip(labels, probabilities):
            predicted = "i" if probability < threshold else "c"
            if gold == "i" and predicted == "i":
                tp += 1
            elif gold == "c" and predicted == "i":
                fp += 1
            elif gold == "i":
                fn += 1
    precision, recall, f1, f05 = prf(tp, fp, fn)
    return {
        "precision": precision, "recall": recall, "f1": f1, "f0.5": f05,
        "tp": tp, "fp": fp, "fn": fn,
    }


def freeze(manifest_path: Path, system: str):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        config = manifest["systems"][system]
    except KeyError as exc:
        raise ValueError(f"unknown system {system!r} in {manifest_path}") from exc

    for test_dataset, entry in config["datasets"].items():
        dev_dataset = entry["dev_dataset"]
        data_path = discover_dev_data(dev_dataset, config, entry)
        gold = load_json(paths.gold(dev_dataset))
        aligned, skipped = collect_aligned(data_path, gold)
        if skipped:
            raise ValueError(f"{data_path}: refusing threshold with skipped units {skipped}")
        threshold, selected_f05 = select_threshold(aligned)
        metrics = metrics_at_threshold(aligned, threshold)
        if abs(metrics["f0.5"] - selected_f05) > 1e-12:
            raise AssertionError("threshold metric disagrees with standard-eval selector")
        result_path = paths.RESULTS_DIR / f"{config['approach']}__{data_path.parent.name}.json"
        entry.update({
            "threshold": threshold,
            "dev_metrics": metrics,
            "source_data": str(data_path.relative_to(REPO_ROOT)),
            "source_result": (
                str(result_path.relative_to(REPO_ROOT)) if result_path.exists() else None
            ),
        })
        print(
            f"{system:5s} {dev_dataset:32s} threshold={threshold:.6g} "
            f"F0.5={metrics['f0.5']*100:.2f}% -> {test_dataset}"
        )

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"updated {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "thresholds.json")
    parser.add_argument("--system", required=True)
    args = parser.parse_args()
    freeze(args.manifest, args.system)


if __name__ == "__main__":
    main()
