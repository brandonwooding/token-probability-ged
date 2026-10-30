"""Pure-Python multilingual GED operating-point metrics."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denom = .25 * precision + recall
    f05 = 1.25 * precision * recall / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f0.5": f05,
            "tp": tp, "fp": fp, "fn": fn}


def _threshold(values: list[tuple[float, int]], error_when_low: bool) -> tuple[float, dict]:
    """Exact best threshold; strict comparisons and low-threshold tie-break."""
    if not values:
        return .5, prf(0, 0, 0)
    unique = sorted({p for p, _ in values})
    candidates = [unique[0] - 1e-12]
    candidates += [(a + b) / 2 for a, b in zip(unique, unique[1:])]
    candidates.append(unique[-1] + 1e-12)
    probs_i = sorted(p for p, y in values if y == 1)
    probs_c = sorted(p for p, y in values if y == 0)
    best = (-1.0, candidates[0], None)
    for threshold in candidates:
        if error_when_low:
            tp, fp = bisect_left(probs_i, threshold), bisect_left(probs_c, threshold)
        else:
            tp = len(probs_i) - bisect_right(probs_i, threshold)
            fp = len(probs_c) - bisect_right(probs_c, threshold)
        score = prf(tp, fp, len(probs_i) - tp)
        if score["f0.5"] > best[0]:
            best = (score["f0.5"], threshold, score)
    return best[1], best[2]


def multilingual_metrics(rows: list[tuple[str, float, int]], error_when_low: bool) -> dict:
    """Tune one threshold/language, then pool resulting counts (micro F0.5)."""
    grouped = defaultdict(list)
    for lang, probability, label in rows:
        grouped[lang].append((float(probability), int(label)))
    per_language, totals = {}, [0, 0, 0]
    for lang in sorted(grouped):
        threshold, score = _threshold(grouped[lang], error_when_low)
        per_language[lang] = {"threshold": threshold, **score, "n": len(grouped[lang])}
        totals[0] += score["tp"]
        totals[1] += score["fp"]
        totals[2] += score["fn"]
    return {"pooled_micro": prf(*totals), "per_language": per_language,
            "selection": "per-language thresholds; pooled micro incorrect-class F0.5"}
