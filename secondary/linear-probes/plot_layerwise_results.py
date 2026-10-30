"""Plot learner-corpus performance for the synthetic-trained linear probes."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = (
    PROBE_ROOT
    / "graphs"
    / "probe_layerwise_synthetic_transfer.png"
)

PANELS = (
    (
        "English: FCE development set",
        (
            (
                "XLM-R-base",
                PROBE_ROOT
                / "output"
                / "en_c4200__FacebookAI__xlm-roberta-base_en_fce_dev.csv",
                9,
            ),
            (
                "mmBERT-base",
                PROBE_ROOT
                / "output"
                / "en_c4200__jhu-clsp__mmBERT-base_en_fce_dev.csv",
                17,
            ),
        ),
    ),
    (
        "German: Falko-MERLIN development set",
        (
            (
                "XLM-R-base",
                PROBE_ROOT
                / "output"
                / "de_c4200__FacebookAI__xlm-roberta-base_de_falko-merlin_dev.csv",
                10,
            ),
            (
                "mmBERT-base",
                PROBE_ROOT
                / "output"
                / "de_c4200__jhu-clsp__mmBERT-base_de_falko-merlin_dev.csv",
                17,
            ),
        ),
    ),
)


def read_curve(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["layer"]) for row in rows], [float(row["f0.5"]) * 100 for row in rows]


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colours = ("#3266a8", "#c24b3a")

    for axis, (title, curves) in zip(axes, PANELS):
        for colour, (label, path, selected_layer) in zip(colours, curves):
            layers, scores = read_curve(path)
            axis.plot(layers, scores, marker="o", markersize=3.5, linewidth=1.8,
                      color=colour, label=label)
            selected_index = layers.index(selected_layer)
            axis.scatter(
                [selected_layer],
                [scores[selected_index]],
                marker="*",
                s=150,
                color=colour,
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
            )

        axis.set_title(title)
        axis.set_xlabel("Hidden-state index")
        axis.set_xticks(range(0, max(layers) + 1, 2))
        axis.set_ylim(0, 65)
        axis.legend(frameon=True)

    axes[0].set_ylabel(r"Erroneous-class $F_{0.5}$ (\%)")
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
