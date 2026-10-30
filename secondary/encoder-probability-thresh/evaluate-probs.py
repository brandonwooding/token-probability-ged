import json
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


with open("llm-probability-thresh/fce-dev-paragraph-labels.json", "r") as file:
    label_data = json.load(file)

with open("llm-probability-thresh/data.json", "r") as file:
    prob_data = json.load(file)


prob_data_flat = [paragraph for essay in prob_data for paragraph in essay]
label_data_flat = [paragraph for essay in label_data for paragraph in essay]

assert len(prob_data_flat) == len(label_data_flat)

skipped_paragraphs = 0
skipped_label_tokens = 0

c_probs = []
c_percs = []
c_ranks = []
c_ents = []

i_probs = []
i_percs = []
i_ranks = []
i_ents = []


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

    label_word_count = len(labels)
    seen_word_ids = set(word_ids)
    expected_word_ids = set(range(label_word_count))

    if seen_word_ids != expected_word_ids:
        print(label_data_flat[paragraph_idx]["tokens"])
        skipped_paragraphs += 1
        skipped_label_tokens += len(labels)
        continue

    word_probs = [1.0] * label_word_count
    word_percs = [1.0] * label_word_count
    word_ranks = [1] * label_word_count
    word_ents = [0] * label_word_count

    for token in prob_paragraph:
        word_id = token["word_id"]
        if word_id is None:
            continue

        word_probs[word_id] = min(word_probs[word_id], token["probability"])
        word_percs[word_id] = min(word_percs[word_id], token["percentile"])
        word_ranks[word_id] = max(word_ranks[word_id], token["rank"])
        word_ents[word_id] = max(word_ents[word_id], token["entropy"])

    for label, prob, perc, rank, ent in zip(labels, word_probs, word_percs, word_ranks, word_ents):
        if label == "c":
            c_probs.append(prob)
            c_percs.append(perc)
            c_ranks.append(rank)
            c_ents.append(ent)
        elif label == "i":
            i_probs.append(prob)
            i_percs.append(perc)
            i_ranks.append(rank)
            i_ents.append(ent)
        else:
            raise ValueError(f"unexpected label: {label}")


def safe_mean(values):
    return mean(values) if values else float("nan")

def get_metrics_from_probs(c_probs, i_probs, threshold):
    correct_labels = [0]*len(c_probs) + [1]*len(i_probs)
    predicted_labels = [int(p < threshold) for p in c_probs] + [int(p < threshold) for p in i_probs]
    report = classification_report(correct_labels, predicted_labels)
    incorrect_f05 = fbeta_score(correct_labels, predicted_labels, beta=0.5, pos_label=1)
    macro_f05 = fbeta_score(correct_labels, predicted_labels, beta=0.5, average="macro")
    weighted_f05 = fbeta_score(correct_labels, predicted_labels, beta=0.5, average="weighted")
    return (
        report
        + f"incorrect_f0.5: {incorrect_f05}\n"
        + f"macro_f0.5: {macro_f05}\n"
        + f"weighted_f0.5: {weighted_f05}\n"
    )

for threshold in np.arange(0.3, 0.5, 0.05):
    print(get_metrics_from_probs(c_probs, i_probs, threshold=threshold))

print(f"paragraphs: {len(prob_data_flat)}")
print(f"skipped_paragraphs: {skipped_paragraphs}")
print(f"used_paragraphs: {len(prob_data_flat) - skipped_paragraphs}")
print(f"skipped_label_tokens: {skipped_label_tokens}")
print()

print("correct labels")
print(f"count: {len(c_probs)}")
print(f"avg_probability: {safe_mean(c_probs)}")
print(f"avg_entropy: {safe_mean(c_ents)}")
print(f"avg_percentile: {safe_mean(c_percs)}")
print(f"avg_rank: {safe_mean(c_ranks)}")
print()

print("incorrect labels")
print(f"count: {len(i_probs)}")
print(f"avg_probability: {safe_mean(i_probs)}")
print(f"avg_entropy: {safe_mean(i_ents)}")
print(f"avg_percentile: {safe_mean(i_percs)}")
print(f"avg_rank: {safe_mean(i_ranks)}")

## Logistic regression predictor

feature_names = ["probability", "log_rank", "entropy"]
X = np.column_stack([
    c_probs + i_probs,
    np.log1p(c_ranks + i_ranks),
    c_ents + i_ents,
])
y = np.array([0] * len(c_probs) + [1] * len(i_probs))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_scaled, y)

y_pred = clf.predict(X_scaled)
y_pred_prob = clf.predict_proba(X_scaled)[:, 1]

print()
print("logistic regression: probability + log_rank + entropy")
print(classification_report(y, y_pred))
print(f"roc_auc: {roc_auc_score(y, y_pred_prob)}")
print("standardized_coefficients:")
for name, coef in zip(feature_names, clf.coef_[0]):
    print(f"{name}: {coef}")
print(f"intercept: {clf.intercept_[0]}")

## PLOTS

c_ppls = -np.log(np.array(c_probs) + 1e-9)
i_ppls = -np.log(np.array(i_probs) + 1e-9)

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

fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
plot_overlay_hist(axes1[0], c_probs, i_probs, 'Probability', 'Min Token Probability')
plot_overlay_hist(axes1[1], c_ranks, i_ranks, 'Rank (log scale)', 'Max Token Rank', log_x=True)
fig1.suptitle('Correct vs Incorrect — Density (overlaid)', fontsize=13)
fig1.tight_layout()
fig1.savefig('llm-probability-thresh/distributions_overlay.png', dpi=150)

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
plot_cdf(axes2[0], c_probs, i_probs, 'Probability CDF', 'Min Token Probability')
plot_cdf(axes2[1], c_ranks, i_ranks, 'Log Rank CDF', 'Log Max Token Rank', log_x=True)
fig2.suptitle('Correct vs Incorrect — CDF', fontsize=13)
fig2.tight_layout()
fig2.savefig('llm-probability-thresh/distributions_cdf.png', dpi=150)

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
plot_overlay_hist(axes3[0], c_ppls, i_ppls, 'Surprisal (log density)', 'Surprisal / Token PPL (-log P)', bins=60)
plot_overlay_hist(axes3[1], c_ranks, i_ranks, 'Rank (log-log)', 'Max Token Rank', log_x=True)
axes3[0].set_yscale('log')
axes3[1].set_yscale('log')
fig3.suptitle('Correct vs Incorrect — Density (log y-axis)', fontsize=13)
fig3.tight_layout()
fig3.savefig('llm-probability-thresh/distributions_log.png', dpi=150)

fig_entropy, axes_entropy = plt.subplots(1, 2, figsize=(10, 4))
plot_overlay_hist(axes_entropy[0], c_ents, i_ents, 'Entropy', 'Max Token Entropy (nats)', bins=60)
plot_cdf(axes_entropy[1], c_ents, i_ents, 'Entropy CDF', 'Max Token Entropy (nats)')
fig_entropy.suptitle('Correct vs Incorrect — Entropy', fontsize=13)
fig_entropy.tight_layout()
fig_entropy.savefig('llm-probability-thresh/distributions_entropy.png', dpi=150)

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
fig4.savefig('llm-probability-thresh/scatter_ppl_rank.png', dpi=150)

c_probs_arr = np.array(c_probs)
c_ranks_arr = np.array(c_ranks)
c_ents_arr = np.array(c_ents)
i_probs_arr = np.array(i_probs)
i_ranks_arr = np.array(i_ranks)
i_ents_arr = np.array(i_ents)

c_mask = c_probs_arr > 0.5
i_mask = i_probs_arr > 0.5

fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.scatter(c_probs_arr[c_idx], c_ents_arr[c_idx], alpha=0.2, s=8, color='steelblue', label='correct')
ax5.scatter(i_probs_arr[i_idx], i_ents_arr[i_idx], alpha=0.2, s=8, color='tomato', label='incorrect')
ax5.set_xlabel('Min Token Probability')
ax5.set_ylabel('Max Token Entropy (nats)')
ax5.set_title('Probability vs Entropy')
ax5.legend(markerscale=3)
fig5.tight_layout()
fig5.savefig('llm-probability-thresh/scatter_prob_entropy.png', dpi=150)

fig6, ax6 = plt.subplots(figsize=(8, 6))
ax6.scatter(c_probs_arr[c_mask], c_ranks_arr[c_mask], alpha=0.2, s=8, color='steelblue', label=f'correct (n={c_mask.sum()})')
ax6.scatter(i_probs_arr[i_mask], i_ranks_arr[i_mask], alpha=0.2, s=8, color='tomato', label=f'incorrect (n={i_mask.sum()})')
ax6.set_xlabel('Min Token Probability')
ax6.set_ylabel('Rank')
ax6.set_title('Probability vs Rank (prob > 0.5)')
ax6.legend(markerscale=3)
fig6.tight_layout()
fig6.savefig('llm-probability-thresh/scatter_highprob_rank.png', dpi=150)

plt.close('all')
