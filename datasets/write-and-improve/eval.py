#!/usr/bin/env python3
"""Strict token-level GED evaluation for packaged W&I TSV files.

Unlike the original MultiGED helper, this scorer rejects truncated files,
different sentence boundaries, token drift, extra columns, and invalid labels
before computing incorrect-class precision, recall, and F-beta.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_tsv(path: Path):
    units, current = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(
                csv.reader(handle, delimiter="\t", quotechar='"', escapechar="\\"), 1):
            if not row or not row[0].strip():
                if current:
                    units.append(current)
                    current = []
                continue
            if len(row) != 2:
                raise ValueError(f"{path}:{line_number}: expected 2 columns, got {len(row)}")
            token = '"' if row[0] == r'\"' else row[0]
            label = row[1]
            if label not in {"c", "i"}:
                raise ValueError(f"{path}:{line_number}: invalid GED label {label!r}")
            current.append((token, label))
    if current:
        units.append(current)
    if not units:
        raise ValueError(f"{path}: no sentences")
    return units


def score(hypothesis, reference, beta):
    if len(hypothesis) != len(reference):
        raise ValueError(
            f"sentence-count mismatch: hypothesis={len(hypothesis)}, reference={len(reference)}")
    tp = fp = fn = tokens = 0
    for unit_id, (hyp_unit, ref_unit) in enumerate(zip(hypothesis, reference)):
        if len(hyp_unit) != len(ref_unit):
            raise ValueError(
                f"sentence {unit_id}: token-count mismatch "
                f"hypothesis={len(hyp_unit)}, reference={len(ref_unit)}")
        for token_id, ((hyp_token, hyp_label), (ref_token, ref_label)) in enumerate(
                zip(hyp_unit, ref_unit)):
            if hyp_token != ref_token:
                raise ValueError(
                    f"sentence {unit_id}, token {token_id}: token mismatch "
                    f"hypothesis={hyp_token!r}, reference={ref_token!r}")
            tokens += 1
            tp += hyp_label == ref_label == "i"
            fp += hyp_label == "i" and ref_label != "i"
            fn += hyp_label != "i" and ref_label == "i"
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    beta2 = beta ** 2
    fscore = ((1 + beta2) * precision * recall / (beta2 * precision + recall)
              if precision + recall else 0.0)
    return {
        "sentences": len(reference), "tokens": tokens,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        f"f{beta:g}": fscore,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-hyp", "--hyp", type=Path, required=True,
                        help="system TSV with the packaged tokens and predicted c/i labels")
    parser.add_argument("-ref", "--ref", type=Path, required=True,
                        help="gold TSV produced by build_wi_ged.py")
    parser.add_argument("-b", "--beta", type=float, default=0.5)
    args = parser.parse_args()
    if args.beta <= 0:
        parser.error("--beta must be positive")
    result = score(read_tsv(args.hyp), read_tsv(args.ref), args.beta)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
