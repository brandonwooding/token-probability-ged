"""Adapter: GED-by-instruction outputs -> canonical predictions.

The instruction model already emits per-token "c"/"i" in predicted_labels, so
this is almost a straight copy. The one convention we preserve from the
approach's own evaluate_ged.py: a paragraph whose predicted_labels is null
(the model produced invalid JSON or a wrong-length array) is scored as all
"c" -- i.e. "detected no errors" -- rather than being dropped. That keeps
parse failures counting against recall, exactly as the original eval did.
"""

import argparse
from pathlib import Path

from common import load_json, save_json


def adapt(data_path, out_path, tag=None, dataset=None):
    """GED-by-instruction raw output (flat list of sentences with predicted_labels)
    -> canonical predictions. A unit whose predicted_labels is null (invalid JSON
    or wrong length) is scored as all-"c", matching the approach's evaluate_ged.py."""
    sentences = load_json(data_path)

    units = []
    parse_failures = 0
    for unit_id, sent in enumerate(sentences):
        predicted = sent.get("predicted_labels")
        if predicted is None:
            parse_failures += 1
            predicted = ["c"] * len(sent["tokens"])
        units.append(
            {
                "unit_id": unit_id,
                "tokens": sent["tokens"],
                "predicted_labels": predicted,
            }
        )

    pred = {
        "approach": "ged-by-instruction",
        "tag": tag or Path(data_path).parent.name,
        "meta": {
            "dataset": dataset,
            "source": str(data_path),
            "parse_failures_as_all_c": parse_failures,
        },
        "units": units,
    }
    save_json(out_path, pred)
    print(f"wrote {out_path}  units={len(units)}  parse_failures={parse_failures}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="a run's data.json.")
    parser.add_argument("--dataset", default=None, help="dataset id (for meta).")
    parser.add_argument("--tag", default=None, help="label; default: the run folder name.")
    parser.add_argument("--out", type=Path, default=None,
                        help="prediction json; default: <run>/predictions.json.")
    args = parser.parse_args()

    out = args.out or (args.data.parent / "predictions.json")
    adapt(args.data, out, tag=args.tag, dataset=args.dataset)


if __name__ == "__main__":
    main()
