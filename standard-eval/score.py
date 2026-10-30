"""Score a canonical predictions file against a canonical gold file.

Produces:
  * global binary detection metrics (precision / recall / F1 / F0.5), positive = "i"
  * recall per gold error type (how sensitive the system is to each phenomenon)
  * coverage (how many units/tokens were actually scored vs skipped)

Per-type PRECISION is deliberately not reported: these systems predict only
"error", never "error of type T", so a false positive has no gold type to
attribute it to. Recall-per-type is well defined; aggregate precision is global.

The scorer is dataset- and approach-agnostic. It only understands the canonical
shape and joins on unit_id. Tokens are asserted to match per unit so any silent
misalignment fails loudly instead of corrupting a metric.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from common import POSITIVE, REPO_ROOT, load_json, save_json, prf

sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402


def _tally(gold_units, pred_units, unit_ids):
    """Confusion counts + per-type recall over `unit_ids`.

    Split out from score() so a stratum is measured by exactly the same code as
    the whole set -- a group's numbers are then guaranteed to be the global ones
    restricted to its units, not a parallel reimplementation that could drift.
    """
    tp = fp = fn = tn = 0
    type_stats = defaultdict(lambda: [0, 0])  # gold type -> [hits, support]
    scored_units = scored_tokens = 0

    for unit_id in unit_ids:
        pu = pred_units[unit_id]
        gu = gold_units[unit_id]
        if gu["tokens"] != pu["tokens"]:
            raise ValueError(
                f"unit {unit_id}: prediction tokens do not match gold tokens "
                f"(misaligned adapter)"
            )

        scored_units += 1
        scored_tokens += len(gu["tokens"])

        for gold_label, gold_type, pred_label in zip(
            gu["labels"], gu["types"], pu["predicted_labels"]
        ):
            g_pos = gold_label == POSITIVE
            p_pos = pred_label == POSITIVE

            if g_pos and p_pos:
                tp += 1
            elif p_pos:
                fp += 1
            elif g_pos:
                fn += 1
            else:
                tn += 1

            # recall bookkeeping per gold error type
            if g_pos:
                type_stats[gold_type][1] += 1
                if p_pos:
                    type_stats[gold_type][0] += 1

    precision, recall, f1, f05 = prf(tp, fp, fn)
    per_type = {
        etype: {
            "recall": hits / support if support else 0.0,
            "hits": hits,
            "support": support,
        }
        for etype, (hits, support) in type_stats.items()
    }
    return {
        "global": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f0.5": f05,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "scored_units": scored_units,
        "scored_tokens": scored_tokens,
        "per_type_recall": dict(
            sorted(per_type.items(), key=lambda kv: kv[1]["support"], reverse=True)
        ),
    }


def score(gold, pred, strata=None):
    """Metrics for one predictions file, optionally also broken down by stratum.

    `strata` maps unit_id -> group label (e.g. a CEFR band from the dataset's
    .cefr.json). Units whose group is None are left out of the breakdown but
    still counted globally, so the global numbers never depend on whether a
    breakdown was requested.

    Note when reading a breakdown: the positive base rate varies by group, and
    the operating point that maximises F0.5 moves with base rate on its own. A
    group whose threshold differs is not by itself evidence that the group
    differs in any way except error density.
    """
    gold_units = {u["unit_id"]: u for u in gold["units"]}
    pred_units = {u["unit_id"]: u for u in pred["units"]}

    overall = _tally(gold_units, pred_units, list(pred_units))

    total_units = len(gold_units)
    total_tokens = sum(len(u["tokens"]) for u in gold["units"])

    result = {
        "approach": pred.get("approach"),
        "tag": pred.get("tag"),
        "meta": pred.get("meta", {}),
        "global": overall["global"],
        "coverage": {
            "scored_units": overall["scored_units"],
            "total_units": total_units,
            "scored_tokens": overall["scored_tokens"],
            "total_tokens": total_tokens,
            "skipped_units": total_units - overall["scored_units"],
        },
        "per_type_recall": overall["per_type_recall"],
    }

    if strata:
        groups = defaultdict(list)
        for unit_id in pred_units:
            label = strata.get(unit_id)
            if label is not None:
                groups[label].append(unit_id)
        per_group = {}
        for label in sorted(groups):
            tally = _tally(gold_units, pred_units, groups[label])
            gold_pos = tally["global"]["tp"] + tally["global"]["fn"]
            per_group[label] = {
                **tally["global"],
                "units": tally["scored_units"],
                "tokens": tally["scored_tokens"],
                "gold_positives": gold_pos,
                "base_rate": gold_pos / tally["scored_tokens"] if tally["scored_tokens"] else 0.0,
                "per_type_recall": tally["per_type_recall"],
            }
        result["per_group"] = per_group
        result["grouped_units"] = sum(len(v) for v in groups.values())

    return result


def format_report(result):
    g = result["global"]
    c = result["coverage"]
    lines = [
        f"== {result['approach']} / {result['tag']} ==",
        f"P={g['precision']*100:6.2f}%  R={g['recall']*100:6.2f}%  "
        f"F1={g['f1']*100:6.2f}%  F0.5={g['f0.5']*100:6.2f}%",
        f"tp={g['tp']} fp={g['fp']} fn={g['fn']} tn={g['tn']}",
        f"coverage: units {c['scored_units']}/{c['total_units']}  "
        f"tokens {c['scored_tokens']}/{c['total_tokens']}  "
        f"skipped_units={c['skipped_units']}",
        "",
        "recall by gold error type:",
        f"  {'type':16s} {'recall':>8s} {'hits':>6s} {'support':>8s}",
    ]
    for etype, s in result["per_type_recall"].items():
        lines.append(
            f"  {etype:16s} {s['recall']*100:7.2f}% {s['hits']:6d} {s['support']:8d}"
        )
    if result.get("per_group"):
        lines += [
            "",
            f"by group ({result['grouped_units']} units grouped):",
            f"  {'group':10s} {'units':>6s} {'tokens':>8s} {'base':>7s} "
            f"{'P':>7s} {'R':>7s} {'F0.5':>7s}",
        ]
        for label, g in result["per_group"].items():
            lines.append(
                f"  {label:10s} {g['units']:6d} {g['tokens']:8d} "
                f"{g['base_rate']*100:6.2f}% {g['precision']*100:6.2f}% "
                f"{g['recall']*100:6.2f}% {g['f0.5']*100:6.2f}%"
            )
        lines.append(
            "  note: base rate differs by group, and the F0.5-optimal operating "
            "point moves with base rate alone."
        )
    return "\n".join(lines)


def load_strata(dataset, field):
    """unit_id -> group label, from the dataset's .cefr.json."""
    path = paths.cefr(dataset)
    if path is None:
        raise SystemExit(
            f"dataset '{dataset}' has no .cefr.json, so --group-by is unavailable"
        )
    payload = load_json(path)
    units = payload["units"]
    available = sorted(set(units[0]) - {"unit_id"}) if units else []
    if field not in available:
        raise SystemExit(f"unknown field '{field}'; available: {', '.join(available)}")
    return {u["unit_id"]: u[field] for u in units}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="a run's predictions.json.")
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. english/en_fce_dev.")
    parser.add_argument("--out", type=Path, default=None,
                        help="metrics json; default: results/<run-folder>.json.")
    parser.add_argument("--group-by", default=None, metavar="FIELD",
                        help="also break metrics down by a .cefr.json field, "
                             "e.g. human_cefr_coarse.")
    args = parser.parse_args()

    gold = load_json(paths.gold(args.dataset))
    pred = load_json(args.pred)
    strata = load_strata(args.dataset, args.group_by) if args.group_by else None
    result = score(gold, pred, strata)
    if args.group_by:
        result["grouped_by"] = args.group_by

    print(format_report(result))

    out = args.out or (paths.RESULTS_DIR / f"{args.pred.parent.name}.json")
    save_json(out, result)


if __name__ == "__main__":
    main()
