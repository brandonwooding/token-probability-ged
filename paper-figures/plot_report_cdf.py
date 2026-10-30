"""Generate the paired report CDFs for Gemma on FCE-dev."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "standard-eval"))
from pooling import flatten  # noqa: E402

RUNS = [
    ("No prompt", "en_fce_dev__gemma-4-12B__none__20260828-162553"),
    ("Few-shot", "en_fce_dev__gemma-4-12B__fewshot__20260828-164421"),
]
GOLD_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "multiged-2023"
    / "english"
    / "en_fce_dev.gold.json"
)
OUTPUTS_DIR = PROJECT_ROOT / "decoder-probability-thresh" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "standard-eval" / "results"
OUTPUT_PATH = (
    SCRIPT_DIR
    / "graphs"
    / "probability_cdf_gemma_fce_contexts.png"
)


def load_word_probabilities(run_tag, gold_units):
    probability_path = OUTPUTS_DIR / run_tag / "data.json"
    with probability_path.open() as probability_file:
        probability_units = json.load(probability_file)

    if len(probability_units) != len(gold_units):
        raise ValueError("Probability and gold files contain different numbers of units")

    correct = []
    erroneous = []
    labels = []
    scores = []
    for probability_unit, gold_unit in zip(probability_units, gold_units):
        unit_labels = gold_unit["labels"]
        word_probabilities = [1.0] * len(unit_labels)

        for subtoken in flatten(probability_unit):
            word_id = subtoken["word_id"]
            if word_id is not None:
                word_probabilities[word_id] = min(
                    word_probabilities[word_id], subtoken["probability"]
                )

        for label, probability in zip(unit_labels, word_probabilities):
            target = correct if label == "c" else erroneous
            target.append(probability)
            labels.append(1 if label == "i" else 0)
            scores.append(-probability)

    return (
        np.sort(np.clip(correct, 1e-9, 1.0)),
        np.sort(np.clip(erroneous, 1e-9, 1.0)),
        roc_auc_score(labels, scores),
    )


def cumulative_proportions(values):
    return np.arange(1, len(values) + 1) / len(values)


def main():
    with GOLD_PATH.open() as gold_file:
        gold_units = json.load(gold_file)["units"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, (title, run_tag) in zip(axes, RUNS):
        correct, erroneous, auc = load_word_probabilities(run_tag, gold_units)
        result_path = RESULTS_DIR / f"decoder-thresh__{run_tag}.json"
        with result_path.open() as result_file:
            result = json.load(result_file)
        threshold = result["meta"]["threshold"]
        f05 = result["global"]["f0.5"]

        ax.plot(
            correct,
            cumulative_proportions(correct),
            color="#356da8",
            linewidth=2.2,
            label=f"Correct (n={len(correct):,})",
        )
        ax.plot(
            erroneous,
            cumulative_proportions(erroneous),
            color="#c95055",
            linewidth=2.2,
            label=f"Erroneous (n={len(erroneous):,})",
        )
        ax.axvline(
            threshold,
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label=rf"Oracle $\tau={threshold:.3g}$",
        )
        ax.set_xscale("log")
        ax.set_xlim(1e-9, 1.0)
        ax.set_ylim(0, 1.01)
        ax.set_title(rf"{title}: AUC={auc:.3f}, $F_{{0.5}}$={100*f05:.2f}")
        ax.set_xlabel("Minimum sub-token probability per dataset token")
        ax.grid(alpha=0.22, which="both")
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("Cumulative proportion")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
