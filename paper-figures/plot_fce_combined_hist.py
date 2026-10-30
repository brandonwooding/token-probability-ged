"""Three FCE-dev probability histograms side by side (sqrt y axis, shared axes).

Base No Prompt | Base Few-shot | Adapted Few-shot. Same data and styling as
plot_fce_probability_dists.py's scalar-sqrt mode; written to
./graphs/scalar-sqrt/fce_hist__combined.png.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_fce_probability_dists import (
    COLOURS, GOLD_PATH, GRAPHS_DIR, NAMES, RESULTS_DIR, RUNS,
    auc_erroneous, word_probabilities,
)

BIN_WIDTH = 0.02
TICK = 14          # axis tick labels (the numbers)
FONT = 18          # axis titles, legend, tau labels
TITLE = 20         # run name; the AUC/F0.5 line beneath is TITLE - 2


def main():
    plt.rcdefaults()
    plt.rcParams.update({"axes.labelsize": FONT, "xtick.labelsize": TICK,
                         "ytick.labelsize": TICK, "legend.fontsize": FONT})
    gold_units = json.load(open(GOLD_PATH))["units"]
    edges = np.linspace(0, 1, round(1 / BIN_WIDTH) + 1)

    panels, top = [], 0.0
    for slug, name, tag, _ in RUNS:
        probs = word_probabilities(tag, gold_units)
        result = json.load(open(RESULTS_DIR / f"decoder-thresh__{tag}.json"))
        shares = {k: np.histogram(probs[k], bins=edges)[0] / len(probs[k]) for k in "ci"}
        top = max(top, *(s.max() for s in shares.values()))
        panels.append((name, result["meta"]["threshold"], 100 * result["global"]["f0.5"],
                       auc_erroneous(probs["c"], probs["i"]), shares,
                       {k: len(probs[k]) for k in "ci"}))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), sharex=True, sharey=True)
    axes[0].set_yscale("function", functions=(np.sqrt, np.square))
    for letter, ax, (name, thr, f05, auc, shares, n) in zip("abc", axes, panels):
        for k in "ci":
            ax.stairs(shares[k], edges, color=COLOURS[k], linewidth=1.8, fill=False)
            ax.stairs(shares[k], edges, color=COLOURS[k], alpha=0.5, fill=True)
        ax.axvline(thr, color="#333333", linestyle="--", linewidth=1.5)
        ax.set_title(f"({letter}) " + name.removeprefix("Gemma 4 "), fontsize=TITLE, pad=40)
        ax.text(0.5, 1.015, rf"AUC={auc:.3f}, $F_{{0.5}}$={f05:.2f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=TITLE - 2)
        tau = rf"{thr:.3g}" if thr >= 1e-3 else rf"{thr / 10 ** np.floor(np.log10(thr)):.3g}\times10^{{{int(np.floor(np.log10(thr)))}}}"
        # beside the dashed line: halfway up, just to its right (clear of the first 0.02-wide
        # bin, which the No Prompt line sits inside)
        ax.text(max(thr, 0.02) + 0.03, 0.5, rf"$\tau = {tau}$", transform=ax.get_xaxis_transform(),
                ha="left", va="center", fontsize=FONT,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
        ax.grid(alpha=0.22)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, top * 1.04)
    axes[0].set_yticks([t for t in (0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
                        if t <= top * 1.04])
    handles = [Line2D([], [], color=COLOURS["c"], linewidth=2), Line2D([], [], color=COLOURS["i"], linewidth=2),
               Line2D([], [], color="#333333", linestyle="--", linewidth=1.5)]
    fig.legend(handles, ["Correct", "Erroneous", "Oracle threshold"], loc="lower center",
               ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.0))  # one legend for all panels
    fig.supxlabel("Minimum token probability", fontsize=FONT, y=0.13)
    fig.supylabel("Proportion (sqrt scale)", fontsize=FONT, x=0.02)
    # explicit margins (tight_layout pushes the shared axis titles too far out)
    fig.subplots_adjust(left=0.082, right=0.98, top=0.79, bottom=0.25, wspace=0.12)
    fig.savefig(GRAPHS_DIR / "scalar-sqrt" / "fce_hist__combined.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
