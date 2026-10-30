"""Dataset error rate (%) vs oracle threshold, one circle per dev corpus, per run.

Error rate = share of gold tokens labelled incorrect. Oracle threshold = the
`meta.threshold` the standard-eval scorer selected on that dev set (auto, F0.5).
Each plot also carries a least-squares line (error rate on threshold) labelled with
its adjusted R^2. Graphs go to ./graphs/error-rates.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
UGL = SCRIPT_DIR.parent
sys.path.insert(0, str(UGL / "standard-eval"))
sys.path.insert(0, str(UGL))
import paths  # noqa: E402
from common import load_json  # noqa: E402
from plot_threshold_sweep import (  # noqa: E402
    CORPORA, CORPUS_COLOURS, CORPUS_NAMES, RESULTS_DIR, RUNS,
)

GRAPHS_DIR = SCRIPT_DIR / "graphs" / "error-rates"
PLOT_RUNS = ["gemma4_base_fewshot", "gemma4_a2a_fewshot"]


def error_rate(corpus):
    labels = [l for u in load_json(paths.gold(corpus))["units"] for l in u["labels"]]
    return 100 * sum(l == "i" for l in labels) / len(labels)


def main():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    rates = {c: error_rate(c) for c in CORPORA}
    for slug in PLOT_RUNS:
        title, run_dirs = RUNS[slug]
        thr = {c: json.load(open(RESULTS_DIR / f"decoder-thresh__{run_dirs[c]}.json"))
               ["meta"]["threshold"] for c in CORPORA}
        r = np.corrcoef([thr[c] for c in CORPORA], [rates[c] for c in CORPORA])[0, 1]
        print(slug, {c: (round(rates[c], 2), thr[c]) for c in CORPORA}, f"pearson r={r:.3f}")

        x = np.array([thr[c] for c in CORPORA])
        y = np.array([rates[c] for c in CORPORA])
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - 2)
        print(f"  fit: error_rate = {slope:.2f} * tau + {intercept:.2f}, "
              f"R2={r2:.3f}, adj R2={adj_r2:.3f}")

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for c, colour in zip(CORPORA, CORPUS_COLOURS):
            ax.plot(thr[c], rates[c], "o", color=colour, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.0, label=CORPUS_NAMES[c])
        xs = np.array([0.0, x.max() * 1.04])
        ax.plot(xs, slope * xs + intercept, color="#333333", linewidth=1.5, zorder=1,
                label=rf"Linear fit (Adj. $R^2$={adj_r2:.2f})")
        ax.set_xlabel("Oracle threshold")
        ax.set_ylabel("Error rate (% of dataset tokens)")
        ax.set_title(f"{title}: error rate vs oracle threshold")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", fontsize=9, frameon=True)
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / f"error_rate_vs_oracle_threshold__{slug}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
