"""F0.5 vs probability threshold (0..1 inclusive, step 0.001), base vs a2a-adapted Gemma 4.

1. Sweeps the threshold on every dev corpus for each run and writes one CSV per
   run to ./data (rows = thresholds, columns = corpora, values = incorrect-class
   F0.5 as a fraction).
2. Plots F0.5 vs threshold for one corpus (FCE) per run into ./graphs/thresholds,
   plus an all-corpora plot (dots mark each corpus's best threshold; dotted line + star = best shared
   threshold by macro-mean F0.5) for ALL_CORPORA_RUNS.

Scoring matches standard-eval/adapt_thresh.py: min sub-token probability per
word, a word is predicted incorrect when its probability is strictly < threshold.
"""

import csv
import json
import sys
from bisect import bisect_left
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
UGL = SCRIPT_DIR.parent
sys.path.insert(0, str(UGL / "standard-eval"))
import adapt_thresh  # noqa: E402
import paths  # noqa: E402
from common import load_json, prf  # noqa: E402

OUTPUTS_DIR = UGL / "decoder-probability-thresh" / "outputs"
RESULTS_DIR = UGL / "standard-eval" / "results"
DATA_DIR = SCRIPT_DIR / "data"
GRAPHS_DIR = SCRIPT_DIR / "graphs" / "thresholds"

THRESHOLDS = [round(i / 1000, 3) for i in range(1001)]  # 0.000 .. 1.000 inclusive
CORPORA = ["cs_geccc_dev", "en_fce_dev", "en_realec_dev",
           "de_falko-merlin_dev", "it_merlin_dev", "sv_swell_dev"]
ADAPTER = "gemma-4-12B__en-de-it-sv__mlcheck__train20k__all__a2a-skip__20260903-115620__fewshot__"
# slug -> (plot title, {corpus: run directory name})
RUNS = {
    "gemma4_base_noprompt": ("Gemma 4 Base No Prompt", {
        "cs_geccc_dev": "cs_geccc_dev__gemma-4-12B__none__20260828-155845",
        "en_fce_dev": "en_fce_dev__gemma-4-12B__none__20260828-162553",
        "en_realec_dev": "en_realec_dev__gemma-4-12B__none__20260828-165248",
        "de_falko-merlin_dev": "de_falko-merlin_dev__gemma-4-12B__none__20260828-172507",
        "it_merlin_dev": "it_merlin_dev__gemma-4-12B__none__20260828-174916",
        "sv_swell_dev": "sv_swell_dev__gemma-4-12B__none__20260828-180903"}),
    "gemma4_base_fewshot": ("Gemma 4 Base Few-shot", {
        "cs_geccc_dev": "cs_geccc_dev__gemma-4-12B__fewshot__20260828-161929",
        "en_fce_dev": "en_fce_dev__gemma-4-12B__fewshot__20260828-164421",
        "en_realec_dev": "en_realec_dev__gemma-4-12B__fewshot__20260828-171821",
        "de_falko-merlin_dev": "de_falko-merlin_dev__gemma-4-12B__fewshot__20260828-174439",
        "it_merlin_dev": "it_merlin_dev__gemma-4-12B__fewshot__20260828-180339",
        "sv_swell_dev": "sv_swell_dev__gemma-4-12B__fewshot__20260828-182358"}),
    "gemma4_a2a_c4200m_fewshot": ("Gemma 4 Adapted C4200M (Fewshot)", {
        c: f"{c}__gemma-4-12B__en-de-es-ro-ru__train20k__all__a2a-skip__20260829-205946__fewshot__{ts}"
        for c, ts in [
            ("cs_geccc_dev", "20260831-120219"), ("en_fce_dev", "20260831-121924"),
            ("en_realec_dev", "20260831-124258"), ("de_falko-merlin_dev", "20260831-130045"),
            ("it_merlin_dev", "20260831-131202"), ("sv_swell_dev", "20260831-132400")]}),
    "gemma4_a2a_fewshot": ("Gemma 4 Adapted Few-shot", {
        c: f"{c}__{ADAPTER}{ts}" for c, ts in [
            ("cs_geccc_dev", "20260907-014026"), ("en_fce_dev", "20260907-015743"),
            ("en_realec_dev", "20260907-021731"), ("de_falko-merlin_dev", "20260907-023921"),
            ("it_merlin_dev", "20260907-025409"), ("sv_swell_dev", "20260907-030617")]}),
}
COLOUR = "#356da8"
# all-corpora plots are made only for these runs
ALL_CORPORA_RUNS = ["gemma4_base_noprompt", "gemma4_base_fewshot", "gemma4_a2a_fewshot",
                     "gemma4_a2a_c4200m_fewshot"]
CORPUS_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CORPUS_NAMES = {"cs_geccc_dev": "cs GECCC", "en_fce_dev": "en FCE", "en_realec_dev": "en REALEC",
                "de_falko-merlin_dev": "de Falko-MERLIN", "it_merlin_dev": "it MERLIN",
                "sv_swell_dev": "sv SweLL"}


def sweep(corpus, run_dir):
    gold = load_json(paths.gold(corpus))
    aligned, skipped = adapt_thresh.collect_aligned(OUTPUTS_DIR / run_dir / "data.json", gold)
    assert not skipped, f"{run_dir}: {len(skipped)} skipped units"
    probs_i, probs_c = [], []
    for _, _, labels, word_probs in aligned:
        for label, prob in zip(labels, word_probs):
            (probs_i if label == "i" else probs_c).append(prob)
    probs_i.sort()
    probs_c.sort()
    out = []
    for thr in THRESHOLDS:
        tp, fp = bisect_left(probs_i, thr), bisect_left(probs_c, thr)
        out.append(prf(tp, fp, len(probs_i) - tp)[3])
    return out


def main():
    DATA_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    for slug, (title, run_dirs) in RUNS.items():
        curves = {c: sweep(c, run_dirs[c]) for c in CORPORA}
        with open(DATA_DIR / f"f05_vs_threshold__{slug}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold"] + CORPORA)
            for i, thr in enumerate(THRESHOLDS):
                w.writerow([f"{thr:.3f}"] + [f"{curves[c][i]:.6f}" for c in CORPORA])

        if slug in ALL_CORPORA_RUNS:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            for corpus, colour in zip(CORPORA, CORPUS_COLOURS):
                y = [100 * v for v in curves[corpus]]
                top = max(range(len(y)), key=y.__getitem__)
                ax.plot(THRESHOLDS, y, color=colour, linewidth=2.0, label=CORPUS_NAMES[corpus])
                ax.plot(THRESHOLDS[top], y[top], "o", color=colour, markersize=6,
                        markeredgecolor="white", markeredgewidth=1.0)
            # best single threshold shared by every corpus: maximises macro-mean F0.5
            macro = [100 * sum(curves[c][i] for c in CORPORA) / len(CORPORA)
                     for i in range(len(THRESHOLDS))]
            shared = max(range(len(macro)), key=macro.__getitem__)  # first max
            ax.axvline(THRESHOLDS[shared], color="#333333", linestyle=":", linewidth=1.8,
                       label=rf"Best shared $\tau$ ($F_{{0.5}}$={macro[shared]:.2f})")
            ax.plot(THRESHOLDS[shared], macro[shared], "*", color="#333333", markersize=14,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, None)
            ax.set_xlabel("Threshold probability")
            ax.set_ylabel(r"$F_{0.5}$")
            ax.set_title(f"{title}: $F_{{0.5}}$ vs threshold, all dev sets")
            ax.grid(alpha=0.22)
            ax.legend(loc="upper right" if slug == "gemma4_base_noprompt" else "lower center", fontsize=9, frameon=True)
            fig.tight_layout()
            fig.savefig(GRAPHS_DIR / f"all_corpora_f05_vs_threshold__{slug}.png", dpi=180)
            plt.close(fig)

        fce = [100 * v for v in curves["en_fce_dev"]]
        best = max(range(len(fce)), key=fce.__getitem__)  # first max, as adapt_thresh
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot(THRESHOLDS, fce, color=COLOUR, linewidth=2.2, label=r"$F_{0.5}$")
        ax.axvline(THRESHOLDS[best], color="#333333", linestyle="--", linewidth=1.5,
                   label=rf"Best $\tau={THRESHOLDS[best]:.3f}$ ($F_{{0.5}}$={fce[best]:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel(r"$F_{0.5}$")
        ax.set_title(f"{title}: $F_{{0.5}}$ vs threshold, FCE dev")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / f"fce_f05_vs_threshold__{slug}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
