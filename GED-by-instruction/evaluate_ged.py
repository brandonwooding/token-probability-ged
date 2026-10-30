import argparse
import json
from pathlib import Path

POSITIVE = "i"
# Build the path to the outputs folder, relative to this script's location
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def flatten(docs: list) -> list[dict]:
    return [para for doc in docs for para in doc]


def evaluate_file(path: Path) -> dict:
    paragraphs = flatten(json.loads(path.read_text()))

    tp = fp = fn = tn = 0
    parse_failures = 0

    for para in paragraphs:
        gold = para["labels"]
        pred = para.get("predicted_labels")
        if pred is None:
            # Failed-to-parse predictions are scored as all "c" (no errors
            # detected) so they still count against recall/precision rather
            # than being silently excluded.
            parse_failures += 1
            pred = ["c"] * len(gold)
        for g, p in zip(gold, pred):
            g_pos, p_pos = g == POSITIVE, p == POSITIVE
            if g_pos and p_pos:
                tp += 1
            elif p_pos:
                fp += 1
            elif g_pos:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    def f_beta(beta: float) -> float:
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    return {
        "paragraphs": len(paragraphs),
        "tokens": tp + fp + fn + tn,
        "parse_failures": parse_failures,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f_beta(1.0),
        "f0.5": f_beta(0.5),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GED predictions against gold labels (token-level P/R/F1/F0.5)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Prediction JSON file(s). Defaults to outputs/*-fce-dev-predictions.json.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths] if args.paths else sorted(
        OUTPUTS_DIR.glob("*-fce-dev-predictions.json")
    )
    if not paths:
        parser.error("No prediction files found.")

    results = []
    for path in paths:
        result = evaluate_file(path)
        result["model"] = path.name.removesuffix("-fce-dev-predictions.json")
        results.append(result)

    header = f"{'Model':30s} {'P':>7s} {'R':>7s} {'F1':>7s} {'F0.5':>7s} {'Fail':>6s} {'Tokens':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:30s} {r['precision'] * 100:6.2f}% {r['recall'] * 100:6.2f}% "
            f"{r['f1'] * 100:6.2f}% {r['f0.5'] * 100:6.2f}% {r['parse_failures']:6d} {r['tokens']:8d}"
        )


if __name__ == "__main__":
    main()
