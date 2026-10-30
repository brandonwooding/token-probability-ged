import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, fbeta_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LABEL_PATH = REPO_ROOT / "llm-probability-thresh/fce-dev-paragraph-labels.json"
RUNS_DIR = SCRIPT_DIR / "runs"


def safe_mean(values):
    return mean(values) if values else float("nan")


def get_metrics_from_probs(c_probs, i_probs, threshold):
    correct_labels = [0] * len(c_probs) + [1] * len(i_probs)
    predicted_labels = [int(p < threshold) for p in c_probs] + [int(p < threshold) for p in i_probs]
    report = classification_report(correct_labels, predicted_labels)
    incorrect_f05 = fbeta_score(correct_labels, predicted_labels, beta=0.5, pos_label=1)
    macro_f05 = fbeta_score(correct_labels, predicted_labels, beta=0.5, average="macro")
    weighted_f05 = fbeta_score(correct_labels, predicted_labels, beta=0.5, average="weighted")
    return (
        f"threshold: {threshold:.2f}\n"
        + report
        + f"incorrect_f0.5: {incorrect_f05}\n"
        + f"macro_f0.5: {macro_f05}\n"
        + f"weighted_f0.5: {weighted_f05}\n"
    )


def prob_bins():
    return np.concatenate([np.linspace(0, 0.1, 30), np.linspace(0.1, 1.0, 21)[1:]])


def plot_overlay_hist(ax, c_vals, i_vals, title, xlabel, log_x=False, bins=None):
    c_arr, i_arr = np.array(c_vals), np.array(i_vals)
    if bins is None:
        if log_x:
            bins = np.logspace(np.log10(max(min(c_arr.min(), i_arr.min()), 1)), np.log10(max(c_arr.max(), i_arr.max())), 60)
        else:
            bins = prob_bins()
    ax.hist(c_arr, bins=bins, density=True, alpha=0.5, color='steelblue', label='correct')
    ax.hist(i_arr, bins=bins, density=True, alpha=0.5, color='tomato', label='incorrect')
    if log_x:
        ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()


def plot_cdf(ax, c_vals, i_vals, title, xlabel, log_x=False):
    for vals, label, color in [(c_vals, 'correct', 'steelblue'), (i_vals, 'incorrect', 'tomato')]:
        s = np.sort(vals)
        ax.plot(s, np.linspace(0, 1, len(s)), label=label, color=color)
    if log_x:
        ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Density')
    ax.legend()


def latest_run() -> Path:
    # Sort by when data.json was written, not by folder name -- folder names lead
    # with the template, so alphabetical order would always return the same
    # template (e.g. "variant_b") rather than the most recently scored run.
    runs = [p for p in RUNS_DIR.iterdir() if (p / "data.json").exists()]
    if not runs:
        raise FileNotFoundError(f"No runs with data.json found under {RUNS_DIR}")
    return max(runs, key=lambda p: (p / "data.json").stat().st_mtime)


def evaluate(run_dir: Path):
    with open(run_dir / "meta.json", "r") as file:
        meta = json.load(file)
    # Prefix for output filenames / plot titles, e.g. "gemma-4-12B__fewshot".
    tag = f"{meta['model_id']}__{meta['template']}"

    with open(LABEL_PATH, "r") as file:
        label_data = json.load(file)

    with open(run_dir / "data.json", "r") as file:
        prob_data = json.load(file)

    prob_data_flat = [paragraph for essay in prob_data for paragraph in essay]
    label_data_flat = [paragraph for essay in label_data for paragraph in essay]

    assert len(prob_data_flat) == len(label_data_flat)

    skipped_paragraphs = 0
    skipped_label_tokens = 0

    c_probs = []
    c_percs = []
    c_ranks = []

    i_probs = []
    i_percs = []
    i_ranks = []

    for paragraph_idx, prob_paragraph in enumerate(prob_data_flat):
        labels = label_data_flat[paragraph_idx]["labels"]

        word_ids = [
            token["word_id"]
            for token in prob_paragraph
            if token["word_id"] is not None
        ]

        if not word_ids:
            skipped_paragraphs += 1
            skipped_label_tokens += len(labels)
            continue

        model_word_count = max(word_ids) + 1
        label_word_count = len(labels)

        if model_word_count != label_word_count:
            print(label_data_flat[paragraph_idx]["tokens"])
            skipped_paragraphs += 1
            skipped_label_tokens += len(labels)
            continue

        word_probs = [1.0] * label_word_count
        word_percs = [1.0] * label_word_count
        word_ranks = [1] * label_word_count

        for token in prob_paragraph:
            word_id = token["word_id"]
            if word_id is None:
                continue

            word_probs[word_id] = min(word_probs[word_id], token["probability"])
            word_percs[word_id] = min(word_percs[word_id], token["percentile"])
            word_ranks[word_id] = max(word_ranks[word_id], token["rank"])

        for label, prob, perc, rank in zip(labels, word_probs, word_percs, word_ranks):
            if label == "c":
                c_probs.append(prob)
                c_percs.append(perc)
                c_ranks.append(rank)
            elif label == "i":
                i_probs.append(prob)
                i_percs.append(perc)
                i_ranks.append(rank)
            else:
                raise ValueError(f"unexpected label: {label}")

    lines = [f"== {tag} =="]
    for threshold in np.arange(0.02, 1.0, 0.02):
        lines.append(get_metrics_from_probs(c_probs, i_probs, threshold=round(float(threshold), 2)))

    lines += [
        f"paragraphs: {len(prob_data_flat)}",
        f"skipped_paragraphs: {skipped_paragraphs}",
        f"used_paragraphs: {len(prob_data_flat) - skipped_paragraphs}",
        f"skipped_label_tokens: {skipped_label_tokens}",
        "",
        "correct labels",
        f"count: {len(c_probs)}",
        f"avg_probability: {safe_mean(c_probs)}",
        f"avg_percentile: {safe_mean(c_percs)}",
        f"avg_rank: {safe_mean(c_ranks)}",
        "",
        "incorrect labels",
        f"count: {len(i_probs)}",
        f"avg_probability: {safe_mean(i_probs)}",
        f"avg_percentile: {safe_mean(i_percs)}",
        f"avg_rank: {safe_mean(i_ranks)}",
    ]

    report = "\n".join(lines)
    print(report)
    with open(run_dir / "metrics.txt", "w") as file:
        file.write(report + "\n")

    ## PLOTS

    c_ppls = -np.log(np.array(c_probs) + 1e-9)
    i_ppls = -np.log(np.array(i_probs) + 1e-9)

    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
    plot_overlay_hist(axes1[0], c_probs, i_probs, 'Probability', 'Min Token Probability')
    plot_overlay_hist(axes1[1], c_ranks, i_ranks, 'Rank (log scale)', 'Max Token Rank', log_x=True)
    fig1.suptitle('Correct vs Incorrect — Density (overlaid)', fontsize=13)
    fig1.tight_layout()
    fig1.savefig(run_dir / f"{tag}-distributions_overlay.png", dpi=150)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    plot_cdf(axes2[0], c_probs, i_probs, 'Probability CDF', 'Min Token Probability')
    plot_cdf(axes2[1], c_ranks, i_ranks, 'Log Rank CDF', 'Log Max Token Rank', log_x=True)
    fig2.suptitle('Correct vs Incorrect — CDF', fontsize=13)
    fig2.tight_layout()
    fig2.savefig(run_dir / f"{tag}-distributions_cdf.png", dpi=150)

    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
    plot_overlay_hist(axes3[0], c_ppls, i_ppls, 'Surprisal (log density)', 'Surprisal / Token PPL (-log P)', bins=60)
    plot_overlay_hist(axes3[1], c_ranks, i_ranks, 'Rank (log-log)', 'Max Token Rank', log_x=True)
    axes3[0].set_yscale('log')
    axes3[1].set_yscale('log')
    fig3.suptitle('Correct vs Incorrect — Density (log y-axis)', fontsize=13)
    fig3.tight_layout()
    fig3.savefig(run_dir / f"{tag}-distributions_log.png", dpi=150)

    rng = np.random.default_rng(42)
    n_sample = min(2000, len(c_ppls), len(i_ppls))
    c_idx = rng.choice(len(c_ppls), n_sample, replace=False)
    i_idx = rng.choice(len(i_ppls), n_sample, replace=False)

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.scatter(np.array(c_probs)[c_idx], np.log(np.array(c_ranks)[c_idx]), alpha=0.2, s=8, color='steelblue', label='correct')
    ax4.scatter(np.array(i_probs)[i_idx], np.log(np.array(i_ranks)[i_idx]), alpha=0.2, s=8, color='tomato', label='incorrect')
    ax4.set_xlabel('Min Token Probability')
    ax4.set_ylabel('Log Rank')
    ax4.set_title('Probability vs Log Rank')
    ax4.legend(markerscale=3)
    fig4.tight_layout()
    fig4.savefig(run_dir / f"{tag}-scatter_ppl_rank.png", dpi=150)

    c_probs_arr = np.array(c_probs)
    c_ranks_arr = np.array(c_ranks)
    i_probs_arr = np.array(i_probs)
    i_ranks_arr = np.array(i_ranks)

    c_mask = c_probs_arr > 0.5
    i_mask = i_probs_arr > 0.5

    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.scatter(c_probs_arr[c_mask], c_ranks_arr[c_mask], alpha=0.2, s=8, color='steelblue', label=f'correct (n={c_mask.sum()})')
    ax5.scatter(i_probs_arr[i_mask], i_ranks_arr[i_mask], alpha=0.2, s=8, color='tomato', label=f'incorrect (n={i_mask.sum()})')
    ax5.set_xlabel('Min Token Probability')
    ax5.set_ylabel('Rank')
    ax5.set_title('Probability vs Rank (prob > 0.5)')
    ax5.legend(markerscale=3)
    fig5.tight_layout()
    fig5.savefig(run_dir / f"{tag}-scatter_highprob_rank.png", dpi=150)

    plt.close('all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=None,
        help="Path to a run folder under runs/ to evaluate. Repeat for multiple. "
             "Default: the most recent run.",
    )
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs] if args.runs else [latest_run()]
    for run_dir in run_dirs:
        evaluate(run_dir)


if __name__ == "__main__":
    main()
