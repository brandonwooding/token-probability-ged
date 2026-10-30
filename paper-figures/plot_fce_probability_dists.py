"""CDF and histogram of probability, correct vs erroneous tokens, FCE dev.

One pair of PNGs per run, written to ./graphs. Probability is linear (not log):
the per-word minimum sub-token probability used by the thresholding scorer.
Styling follows plot_report_cdf.py (colours, boxed legend, oracle-threshold line,
AUC / F0.5 in the title).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
UGL = SCRIPT_DIR.parent
sys.path.insert(0, str(UGL / "standard-eval"))
from pooling import flatten  # noqa: E402

OUTPUTS_DIR = UGL / "decoder-probability-thresh" / "outputs"
RESULTS_DIR = UGL / "standard-eval" / "results"
GOLD_PATH = UGL / "datasets" / "multiged-2023" / "english" / "en_fce_dev.gold.json"
GRAPHS_DIR = SCRIPT_DIR / "graphs"

# (slug, title, run tag, CDF legend location)
RUNS = [
    ("gemma4_base_noprompt", "Gemma 4 Base No Prompt",
     "en_fce_dev__gemma-4-12B__none__20260828-162553", "lower center"),
    ("gemma4_base_fewshot", "Gemma 4 Base Few-shot",
     "en_fce_dev__gemma-4-12B__fewshot__20260828-164421", "upper center"),
    ("gemma4_a2a_fewshot", "Gemma 4 Adapted Few-shot",
     "en_fce_dev__gemma-4-12B__en-de-it-sv__mlcheck__train20k__all__a2a-skip__20260903-115620__fewshot__20260907-015743",
     "upper center"),
]
COLOURS = {"c": "#356da8", "i": "#c95055"}
NAMES = {"c": "Correct", "i": "Erroneous"}
FLOOR = 1e-9  # probabilities are clipped here before taking logs (as in plot_report_cdf.py)
# scalar: probability on a linear axis, 0.02 bins.
# scalar-sqrt: as scalar, histogram y axis on a sqrt scale (no CDF).
# log: log10(probability), clipped to [-9, 0], 0.1-wide bins (90 bins).
MODES = {
    "scalar": dict(xlabel="Probability (minimum per dataset token)",
                   xlim=(0.0, 1.0), bin_width=0.02, transform=lambda p: p),
    # same as scalar, but the histogram's y axis is a square-root scale (histogram only)
    "scalar-sqrt": dict(xlabel="Probability (minimum per dataset token)",
                        xlim=(0.0, 1.0), bin_width=0.02, transform=lambda p: p,
                        hist_only=True, sqrt_y=True, font_boost=2),
    "log": dict(xlabel="Log probability (minimum per dataset token)",
                xlim=(-9.0, 0.0), bin_width=0.1, cdf_loc="upper left",
                transform=lambda p: np.log10(np.clip(p, FLOOR, 1.0))),
}


def word_probabilities(run_tag, gold_units):
    units = json.load(open(OUTPUTS_DIR / run_tag / "data.json"))
    assert len(units) == len(gold_units)
    out = {"c": [], "i": []}
    for unit, gold in zip(units, gold_units):
        probs = [1.0] * len(gold["labels"])
        for sub in flatten(unit):
            if sub["word_id"] is not None:
                probs[sub["word_id"]] = min(probs[sub["word_id"]], sub["probability"])
        for label, p in zip(gold["labels"], probs):
            out[label].append(p)
    return {k: np.array(v) for k, v in out.items()}


def auc_erroneous(correct, erroneous):
    """AUC for ranking erroneous tokens below correct ones by probability.

    Rank-sum (Mann-Whitney) form of roc_auc_score on -probability, ties
    counted 0.5; done in numpy because scipy/sklearn do not import here.
    """
    values = np.r_[erroneous, correct]
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2  # mean 1-based rank per tie group
    ranks = avg_rank[inverse]
    n_e, n_c = len(erroneous), len(correct)
    return 1 - (ranks[:n_e].sum() - n_e * (n_e + 1) / 2) / (n_e * n_c)


def finish(ax, fig, mode, title, threshold, ylabel, legend_loc, path, oracle_label):
    threshold = mode["transform"](threshold)
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.5,
               label=oracle_label)
    ax.set_xlim(*mode["xlim"])
    ax.set_xlabel(mode["xlabel"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="center")
    ax.grid(alpha=0.22)
    ax.legend(loc=legend_loc, fontsize=9 + mode.get("font_boost", 0), frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    gold_units = json.load(open(GOLD_PATH))["units"]
    for slug, name, tag, cdf_loc in RUNS:
        probs = word_probabilities(tag, gold_units)
        result = json.load(open(RESULTS_DIR / f"decoder-thresh__{tag}.json"))
        threshold, f05 = result["meta"]["threshold"], result["global"]["f0.5"]
        auc = auc_erroneous(probs["c"], probs["i"])
        title = rf"{name}: AUC={auc:.3f}, $F_{{0.5}}$={100 * f05:.2f}"
        oracle_label = rf"Oracle $\tau={threshold:.3g}$"

        for mode_name, mode in MODES.items():
            out_dir = GRAPHS_DIR / mode_name
            plt.rcdefaults()
            boost = mode.get("font_boost", 0)  # points added to every text element
            if boost:
                plt.rcParams.update({"axes.labelsize": 10 + boost, "axes.titlesize": 12 + boost,
                                     "xtick.labelsize": 10 + boost, "ytick.labelsize": 10 + boost})
            out_dir.mkdir(parents=True, exist_ok=True)
            values = {k: mode["transform"](v) for k, v in probs.items()}

            if not mode.get("hist_only"):
                fig, ax = plt.subplots(figsize=(6.4, 4.8))
                for label in ("c", "i"):
                    x = np.sort(values[label])
                    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOURS[label],
                            linewidth=2.2, label=f"{NAMES[label]} (n={len(x):,})")
                ax.set_ylim(0, 1.01)
                finish(ax, fig, mode, title, threshold, "Cumulative proportion",
                       mode.get("cdf_loc", cdf_loc),
                       out_dir / f"fce_cdf__{slug}.png", oracle_label)

            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            lo, hi = mode["xlim"]
            edges = np.linspace(lo, hi, round((hi - lo) / mode["bin_width"]) + 1)
            for label in ("c", "i"):
                counts, _ = np.histogram(values[label], bins=edges)
                share = counts / counts.sum()
                ax.stairs(share, edges, color=COLOURS[label], linewidth=1.8, fill=False,
                          label=f"{NAMES[label]} (n={len(values[label]):,})")
                ax.stairs(share, edges, color=COLOURS[label], alpha=0.5, fill=True)
            ylabel = "Proportion per bin"
            if mode.get("sqrt_y"):
                # bars are drawn at sqrt(proportion); ticks keep the real proportions
                ax.set_yscale("function", functions=(np.sqrt, np.square))
                top = max(np.histogram(values[l], bins=edges)[0].max()
                          / len(values[l]) for l in ("c", "i"))
                ax.set_ylim(0, top * 1.04)
                ax.set_yticks([t for t in (0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
                               if t <= top * 1.04])
                ylabel = "Proportion per bin (sqrt scale)"
            finish(ax, fig, mode, title, threshold, ylabel, "upper center",
                   out_dir / f"fce_hist__{slug}.png", oracle_label)


if __name__ == "__main__":
    main()
