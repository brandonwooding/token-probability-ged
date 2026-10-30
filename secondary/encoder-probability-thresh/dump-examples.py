"""Dump per-word encoder (ModernBERT) statistics for a few example paragraphs.

For each selected paragraph we show, per word, the aggregated probability, rank
and entropy (same aggregation as evaluate-probs.py: min probability, max rank,
max entropy across the word's subword pieces) alongside the gold c/i label.

By default the longest well-aligned paragraphs are chosen, since they make the
correct-vs-incorrect signal easier to eyeball. Output is printed and written to
a text file.
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LABEL_PATH = SCRIPT_DIR / "fce-dev-paragraph-labels.json"
PROB_PATH = SCRIPT_DIR / "data.json"
DEFAULT_OUT = SCRIPT_DIR / "encoder-examples.txt"


def aggregate_words(prob_paragraph, label_word_count):
    """Collapse subword stats to one row per word, or return None if the
    paragraph's subwords don't cover exactly words 0..label_word_count-1."""
    word_ids = [t["word_id"] for t in prob_paragraph if t["word_id"] is not None]
    if not word_ids or set(word_ids) != set(range(label_word_count)):
        return None

    word_probs = [1.0] * label_word_count
    word_ranks = [1] * label_word_count
    word_ents = [0.0] * label_word_count
    for token in prob_paragraph:
        wid = token["word_id"]
        if wid is None:
            continue
        word_probs[wid] = min(word_probs[wid], token["probability"])
        word_ranks[wid] = max(word_ranks[wid], token["rank"])
        word_ents[wid] = max(word_ents[wid], token["entropy"])
    return word_probs, word_ranks, word_ents


def format_paragraph(idx, tokens, labels, probs, ranks, ents):
    lines = []
    n_err = labels.count("i")
    lines.append(f"=== paragraph {idx} | {len(tokens)} words | {n_err} erroneous ===")
    lines.append(f"{'word':<22} {'prob':>10} {'rank':>8} {'entropy':>9}  label")
    lines.append("-" * 60)
    for tok, lab, prob, rank, ent in zip(tokens, labels, probs, ranks, ents):
        mark = "  <- i" if lab == "i" else ""
        lines.append(
            f"{tok[:22]:<22} {prob:>10.4f} {rank:>8d} {ent:>9.3f}  {lab}{mark}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num", type=int, default=5,
                        help="Number of example paragraphs to dump (default 5).")
    parser.add_argument("--min-words", type=int, default=40,
                        help="Only consider paragraphs with at least this many "
                             "words (default 40).")
    parser.add_argument("--require-errors", action="store_true",
                        help="Only show paragraphs that contain at least one "
                             "erroneous (i) word.")
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output text file (default {DEFAULT_OUT.name}).")
    args = parser.parse_args()

    label_flat = [p for essay in json.loads(LABEL_PATH.read_text()) for p in essay]
    prob_flat = [p for essay in json.loads(PROB_PATH.read_text()) for p in essay]
    assert len(label_flat) == len(prob_flat)

    candidates = []
    for idx, (lab_para, prob_para) in enumerate(zip(label_flat, prob_flat)):
        labels = lab_para["labels"]
        if len(labels) < args.min_words:
            continue
        if args.require_errors and "i" not in labels:
            continue
        agg = aggregate_words(prob_para, len(labels))
        if agg is None:
            continue
        candidates.append((idx, lab_para["tokens"], labels, *agg))

    # Longest first so the examples are the meaty ones.
    candidates.sort(key=lambda c: len(c[2]), reverse=True)
    selected = candidates[: args.num]

    blocks = [format_paragraph(*c) for c in selected]
    text = "\n".join(blocks)
    print(text)
    args.out.write_text(text)
    print(f"Wrote {len(selected)} example paragraphs to {args.out}")


if __name__ == "__main__":
    main()
