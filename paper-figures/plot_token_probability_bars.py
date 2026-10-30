"""Per-word probability bar chart for one FCE-dev sentence.

Bar height = the per-word minimum sub-token probability (as in
standard-eval/adapt_thresh.py). The word sits in a horizontal row above its bar
and the value is printed just over the bar. Errors (gold label "i") are drawn in
the erroneous colour used by plot_fce_probability_dists.py; fills are 50% opaque with a solid outline, the legend is boxed, and a
dashed line marks the threshold (0.4). Written to ./graphs.

    python plot_token_probability_bars.py [--unit 299] [--words-below] [--legend-above]

--words-below puts the word row under the x axis (legend below that) instead of above the bars;
the PNG gets a "__words-below" suffix so both layouts can sit side by side.
--legend-above (with --words-below) moves the legend between the title and the plot ("__legend-above").
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
UGL = SCRIPT_DIR.parent
OUTPUTS_DIR = UGL / "decoder-probability-thresh" / "outputs"
GOLD_PATH = UGL / "datasets" / "multiged-2023" / "english" / "en_fce_dev.gold.json"
GRAPHS_DIR = SCRIPT_DIR / "graphs"

RUN_TAG = "en_fce_dev__gemma-4-12B__variant_b__20260828-163806"  # Gemma 4 Base (Variant B)
COLOURS = {"c": "#356da8", "i": "#c95055"}
THRESHOLD = 0.4
FILL_ALPHA = 0.5
AX_H = 1.175  # inches; the plot area is fixed, fig_h / ax_bottom below make room for the title, word row and legend
FONT_BOOST = 4  # points added to every text element (base sizes: values 10, words 12, y title 11, title 12, legend 10)
AX_SCALE = 0.94 / AX_H  # offsets outside the plot are tuned in axes fractions at the old 0.94in height
EDGE_WIDTH = 1.8  # solid outline in the full colour, as in plot_fce_probability_dists.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, default=299)
    parser.add_argument("--words-below", action="store_true")
    parser.add_argument("--legend-above", action="store_true")
    args = parser.parse_args()
    legend_above = args.legend_above and args.words_below
    below = args.words_below
    fig_h = (2.74 if legend_above else 2.71) if below else 2.73
    fig_h += AX_H - 0.94 + (0.12 if legend_above else 0)  # legend-above: extra gap between legend and plot
    ax_bottom = (0.58 if legend_above else 1.05) if below else 0.66  # inches under the axis: word row + legend, or legend only

    gold = json.load(open(GOLD_PATH))["units"][args.unit]
    assert gold["unit_id"] == args.unit
    words = json.load(open(OUTPUTS_DIR / RUN_TAG / "data.json"))[args.unit]
    assert [w["token"] for w in words] == gold["tokens"]
    probs = [min(s["probability"] for s in w["subtokens"]) for w in words]

    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    xs = range(len(words))
    ax.bar(xs, probs, width=0.7,
           color=[to_rgba(COLOURS[l], FILL_ALPHA) for l in gold["labels"]],
           edgecolor=[COLOURS[l] for l in gold["labels"]], linewidth=EDGE_WIDTH)
    ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1.2, zorder=3)
    for x, p, label in zip(xs, probs, gold["labels"]):
        # legend-above layout: value labels take the bar outline colour (full opacity)
        ax.text(x, p + 0.015, f"{p:.3f}", ha="center", va="bottom", fontsize=10 + FONT_BOOST,
                color=COLOURS[label] if legend_above else "black")
        # words in one straight row, above the tallest possible bar + value
        if below:
            ax.text(x, -0.12 * AX_SCALE, gold["tokens"][x], ha="center", va="top", fontsize=12 + FONT_BOOST)
        else:
            ax.text(x, 1 + 0.32 * AX_SCALE, gold["tokens"][x], ha="center", va="bottom", fontsize=12 + FONT_BOOST)

    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, len(words) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Token prob.", fontsize=11 + FONT_BOOST)
    ax.set_title("Threshold on token probabilities", fontsize=12 + FONT_BOOST,
                 pad=(60 + 0.12 * 72 if legend_above else 26) if below else 52)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(facecolor=to_rgba(COLOURS["c"], FILL_ALPHA), edgecolor=COLOURS["c"],
                       linewidth=EDGE_WIDTH, label="Correct"),
                 Patch(facecolor=to_rgba(COLOURS["i"], FILL_ALPHA), edgecolor=COLOURS["i"],
                       linewidth=EDGE_WIDTH, label="Erroneous"),
                 Line2D([], [], color="black", linestyle="--", linewidth=1.2,
                        label=rf"Threshold $\tau = {THRESHOLD}$")],
        loc="lower center" if legend_above else "upper center",
        bbox_to_anchor=(0.5, 1 + 0.27 * AX_SCALE + 0.12 / AX_H) if legend_above
        else (0.5, (-0.46 if below else -0.06) * AX_SCALE), ncol=3, frameon=True,
        fontsize=10 + FONT_BOOST)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=ax_bottom / fig_h,
                        top=(ax_bottom + AX_H) / fig_h)

    GRAPHS_DIR.mkdir(exist_ok=True)
    path = GRAPHS_DIR / f"token_probability_bars__unit{args.unit}__gemma4_base_variant_b{"__words-below" if below else ""}{"__legend-above" if legend_above else ""}.png"
    fig.savefig(path, dpi=180)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
