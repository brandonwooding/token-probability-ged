"""Adapter: a trained linear probe -> canonical predictions.

Unlike the other approaches, linear-probes never saved per-token predictions to
disk -- only aggregate metrics. So this adapter reloads a saved probe and the
saved dev features and re-runs the (tiny, CPU-cheap) forward pass to produce
per-token labels. It reuses the approach's own artifacts without touching its
code.

Needs numpy + torch, so it runs in the GPU/training environment, not the bare
scoring environment. The rest of the pipeline is pure stdlib and runs anywhere.

Operating point: one layer must be chosen (the approach sweeps all layers). By
default we take the layer with the best dev F0.5 from the training run's
best_layers.json; override with --layer.

Alignment: the flat feature array is in the same token order as the gold, and
the co-saved labels.npy holds the gold binary labels. We assert labels.npy
matches the gold, token for token, before trusting the predictions -- so a
misaligned feature dump fails loudly.
"""

import argparse
import json
import sys
from pathlib import Path

from common import REPO_ROOT, load_json, save_json

sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402

# Appended, not inserted: secondary/linear-probes/ carries its own utils.py, and putting
# it ahead of standard-eval/ on the path would shadow modules by accident.
sys.path.append(str(REPO_ROOT / "secondary" / "linear-probes"))
import probe_heads  # noqa: E402

DEFAULT_PROBE_DIR = REPO_ROOT / "secondary/linear-probes/experiments/fce_binary"
DEFAULT_FEATURE_DIR = REPO_ROOT / "secondary/linear-probes/features/fce_eval/en_fce_dev"
LABEL2ID = {"c": 0, "i": 1}
ID2LABEL = {0: "c", 1: "i"}


def pick_best_layer(probe_dir):
    best = load_json(Path(probe_dir) / "best_layers.json")
    layer = max(best.values(), key=lambda r: r["metrics"]["f0.5"])
    return int(layer["layer"]), layer["metrics"]["f0.5"]


def adapt(probe_dir, feature_dir, gold, layer, out_path, tag):
    import numpy as np
    import torch

    probe_dir = Path(probe_dir)
    feature_dir = Path(feature_dir)

    if layer is None:
        layer, sel_f05 = pick_best_layer(probe_dir)
        selection = f"best dev F0.5 = {sel_f05*100:.2f}% (from best_layers.json)"
    else:
        selection = "fixed"

    metadata = torch.load(feature_dir / "metadata.pt", map_location="cpu")
    hidden_size = metadata["hidden_size"]

    # gold binary labels flattened in unit order -- the alignment reference
    gold_labels = [lbl for u in gold["units"] for lbl in u["labels"]]
    saved_labels = np.load(feature_dir / "labels.npy")
    if len(saved_labels) != len(gold_labels):
        raise ValueError(
            f"feature labels ({len(saved_labels)}) != gold tokens ({len(gold_labels)})"
        )
    gold_ids = np.array([LABEL2ID[l] for l in gold_labels])
    if not np.array_equal(saved_labels, gold_ids):
        raise ValueError("labels.npy does not match gold labels -- feature/gold misalignment")

    # The architecture comes from the experiment that trained the checkpoint, not
    # from an assumption here -- otherwise an mlp probe would fail to load (or a
    # changed dropout would load silently and score something that was never
    # trained). Experiments with no recorded head are linear, as they were.
    head_spec = probe_heads.read_spec(probe_dir)
    probe = probe_heads.build_probe(hidden_size, **head_spec)
    probe.load_state_dict(torch.load(probe_dir / f"probe_layer_{layer:02d}.pt", map_location="cpu"))
    probe.eval()

    X = torch.from_numpy(np.array(np.load(feature_dir / f"layer_{layer:02d}.npy"))).float()
    with torch.no_grad():
        flat_preds = probe(X).argmax(dim=-1).tolist()

    # slice the flat predictions back into gold units by token count
    units = []
    pos = 0
    for u in gold["units"]:
        n = len(u["tokens"])
        chunk = flat_preds[pos:pos + n]
        pos += n
        units.append(
            {
                "unit_id": u["unit_id"],
                "tokens": u["tokens"],
                "predicted_labels": [ID2LABEL[p] for p in chunk],
            }
        )
    assert pos == len(flat_preds)

    pred = {
        # An mlp head is not a linear probe and should not be read as one in the
        # comparison table, so the approach label follows the architecture that
        # was actually trained.
        "approach": ("mlp-probe" if head_spec["head"] == probe_heads.HEAD_MLP
                     else "linear-probe"),
        "tag": tag,
        "meta": {
            "probe_dir": str(probe_dir),
            "feature_dir": str(feature_dir),
            "layer": layer,
            "layer_selection": selection,
            "head": probe_heads.describe(head_spec),
            "model_id": metadata.get("model_id"),
        },
        "units": units,
    }
    save_json(out_path, pred)
    print(f"wrote {out_path}  units={len(units)}  layer={layer}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="english/en_fce_dev", help="dataset id (for gold).")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--layer", type=int, default=None, help="Default: best dev F0.5.")
    parser.add_argument("--tag", default="fce_binary")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gold = load_json(paths.gold(args.dataset))
    out = args.out or (paths.run_dir("linear-probe", args.dataset, args.tag) / "predictions.json")
    adapt(args.probe_dir, args.feature_dir, gold, args.layer, out, args.tag)


if __name__ == "__main__":
    main()
