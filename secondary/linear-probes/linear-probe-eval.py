import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import choose_device, compute_metrics
import probe_heads


PROBE_PATTERN = re.compile(r"probe_layer_(\d+)\.pt$")


def parse_probe_layer(probe_path):
    match = PROBE_PATTERN.match(probe_path.name)
    if not match:
        raise ValueError(f"Could not parse layer from probe filename: {probe_path}")
    return int(match.group(1))


def find_probe_paths(probe_dir):
    probe_paths = sorted(
        probe_dir.glob("probe_layer_*.pt"),
        key=parse_probe_layer,
    )
    if not probe_paths:
        raise FileNotFoundError(f"No probe_layer_*.pt files found in {probe_dir}")
    return probe_paths


def find_feature_dirs(features_dir):
    if (features_dir / "metadata.pt").exists():
        return [features_dir]
    feature_dirs = sorted(
        path
        for path in features_dir.iterdir()
        if path.is_dir() and (path / "metadata.pt").exists()
    )
    if not feature_dirs:
        raise FileNotFoundError(
            f"No feature subdirectories with metadata.pt found in {features_dir}"
        )
    return feature_dirs


def load_hidden_size(feature_dirs):
    metadata = torch.load(feature_dirs[0] / "metadata.pt", map_location="cpu")
    return metadata["hidden_size"]


def build_probe(hidden_size, device, head_spec=None):
    """Rebuild the trained module. `head_spec` comes from the experiment's
    config.json (see probe_heads); omitted means the linear default, which is
    what every probe trained before the mlp head existed is."""
    spec = head_spec or probe_heads.spec_from_config({})
    return probe_heads.build_probe(hidden_size, device=device, **spec)


def get_preds_and_labels(probe, loader, device):
    all_preds = []
    all_labels = []

    probe.eval()

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.float().to(device)

            logits = probe(batch_X)
            preds = torch.argmax(logits, dim=-1).cpu()

            all_preds.append(preds)
            all_labels.append(batch_y.cpu())

    return torch.cat(all_preds), torch.cat(all_labels)


def evaluate_probe_on_features(
    probe_path,
    feature_dirs,
    hidden_size,
    device,
    batch_size,
    head_spec=None,
):
    layer_idx = parse_probe_layer(probe_path)
    probe = build_probe(hidden_size=hidden_size, device=device, head_spec=head_spec)
    probe.load_state_dict(torch.load(probe_path, map_location=device))

    layer_preds = []
    layer_labels = []
    total_examples = 0

    for feature_dir in feature_dirs:
        layer_path = feature_dir / f"layer_{layer_idx:02d}.npy"
        labels_path = feature_dir / "labels.npy"

        if not layer_path.exists():
            raise FileNotFoundError(f"Missing layer file: {layer_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        X = torch.from_numpy(np.array(np.load(layer_path, mmap_mode="r"))).float()
        y = torch.from_numpy(np.array(np.load(labels_path, mmap_mode="r"))).long()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size)
        preds, labels = get_preds_and_labels(probe, loader, device)

        layer_preds.append(preds)
        layer_labels.append(labels)
        total_examples += len(labels)

        del X
        del y
        del dataset
        del loader

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    preds = torch.cat(layer_preds)
    labels = torch.cat(layer_labels)
    metrics = compute_metrics(y_true=labels, y_pred=preds)

    return {
        "layer": layer_idx,
        "decision_rule": "argmax",
        "num_examples": total_examples,
        **metrics,
    }


def save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return

    fieldnames = [
        "layer",
        "decision_rule",
        "num_examples",
        "accuracy",
        "recall",
        "precision",
        "f1",
        "f0.5",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-dir",
        type=Path,
        required=True,
        help="Directory containing probe_layer_XX.pt files.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="One feature directory, or a directory containing feature subdirectories.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to write aggregate metrics, as .csv or .json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = choose_device(args.device)
    probe_paths = find_probe_paths(args.probe_dir)
    feature_dirs = find_feature_dirs(args.features_dir)
    hidden_size = load_hidden_size(feature_dirs)
    head_spec = probe_heads.read_spec(args.probe_dir)

    print(f"device={device}")
    print(f"probe_dir={args.probe_dir}")
    print(f"features_dir={args.features_dir}")
    print(f"feature_sets={len(feature_dirs)}")
    print(f"probes={len(probe_paths)}")
    print(f"hidden_size={hidden_size}")
    print(f"head={probe_heads.describe(head_spec)}")
    print("decision_rule=argmax")

    results = []
    for probe_path in probe_paths:
        layer_idx = parse_probe_layer(probe_path)
        print(f"evaluating layer {layer_idx} decision_rule=argmax")
        result = evaluate_probe_on_features(
            probe_path=probe_path,
            feature_dirs=feature_dirs,
            hidden_size=hidden_size,
            device=device,
            batch_size=args.batch_size,
            head_spec=head_spec,
        )
        results.append(result)
        print(
            f"layer {layer_idx}: "
            f"f1={result['f1']:.4f} "
            f"precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f}"
        )

    save_results(results, args.output_path)
    print(f"saved results={args.output_path}")


if __name__ == "__main__":
    main()
