"""Run the standardised evaluation for one or more datasets and print a
cross-approach comparison per dataset.

For each --dataset:
  1. ensure the reconstructed sentences + gold exist (built from the TSV/M2).
  2. discover every run under each approach's outputs/ for that dataset's stem,
     adapt it to canonical predictions, and score it.
  3. print one comparison table (+ a per-type recall table when the gold carries
     error types).

A dataset id is '<language>/<stem>' or a bare '<stem>' (see paths.py). Generation
of each approach's data.json (the GPU step) is separate; this scores whatever
runs already exist under outputs/.

--tune-on selects threshold operating points on a different dataset:

    python run.py --dataset english/en_fce_dev --tune-on english/en_c4200_eval5k

Without it, the threshold approaches pick the threshold that maximises F0.5 on
the very set they are then scored on -- an optimistic operating point. With it,
the threshold is fitted on the named dataset and only applied here, which is
what makes the threshold methods genuinely unsupervised when the tuning set is
synthetic: no human-labelled data touches the operating point. It needs a run of
the same model and prompt template over the tuning dataset, which this pairs up
automatically, and it leaves the self-tuned results in place beside the new ones.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from common import REPO_ROOT, load_json, save_json
import build_gold
import adapt_instruction
import adapt_thresh
import score as scorer

sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402
import reconstruct_from_tsv  # noqa: E402


def ensure_dataset(dataset, rebuild=False):
    """Make sure <stem>.sentences.json and <stem>.gold.json exist; return gold."""
    sent = paths.sentences(dataset)
    if rebuild or not sent.exists():
        units = reconstruct_from_tsv.build(paths.tsv(dataset))
        sent.parent.mkdir(parents=True, exist_ok=True)
        sent.write_text(json.dumps(units, indent=2, ensure_ascii=False))
        print(f"[prep] wrote {sent}")
    gold_path = paths.gold(dataset)
    if rebuild or not gold_path.exists():
        build_gold.build(sent, paths.m2(dataset), gold_path, dataset=paths.stem(dataset))
    return load_json(gold_path)


def _runs(approach, stem):
    return sorted((paths.APPROACH_DIRS[approach] / "outputs").glob(f"{stem}__*"))


# Anchored: a signature strips the trailing timestamp only, so a model whose
# name somehow contained a timestamp-like run would not be mangled.
_TIMESTAMP = re.compile(r"__\d{8}-\d{6}$")


def _signature(run_name, stem):
    """Run folder name -> what identifies the *system*, ignoring dataset and time.

    `en_fce_dev__gemma-4-12B__fewshot__20260722-161841` -> `gemma-4-12B__fewshot`.
    Used to pair a run with the run of the same model and prompt template over
    a different dataset, which is what --tune-on needs: a threshold is only
    meaningful against the probability distribution that produced it, so
    tuning a fewshot-decoder operating point on a no-prompt-decoder run would
    be silently wrong.
    """
    rest = run_name[len(stem) + 2:] if run_name.startswith(stem + "__") else run_name
    return _TIMESTAMP.sub("", rest)


def _tuning_aligned(approach, tune_dataset, tune_gold, signature, cache):
    """Aligned probabilities from the --tune-on run matching `signature`.

    Returns None (with a printed reason) when there is no counterpart run, so a
    partially-covered tuning dataset degrades to "this one system is skipped"
    rather than silently falling back to self-tuning, which would put an
    optimistic number in the same table as honest ones.
    """
    key = (approach, signature)
    if key in cache:
        return cache[key]

    tune_stem = paths.stem(tune_dataset)
    candidates = [d for d in _runs(approach, tune_stem)
                  if _signature(d.name, tune_stem) == signature
                  and (d / "data.json").exists()]
    if not candidates:
        cache[key] = None
        return None

    # Newest wins if the same system was run over the tuning set more than once.
    tune_dir = sorted(candidates)[-1]
    aligned, _ = adapt_thresh.collect_aligned(tune_dir / "data.json", tune_gold)
    cache[key] = (aligned, tune_stem, tune_dir.name)
    return cache[key]


def build_and_score(dataset, gold, tune_dataset=None, tune_gold=None, strata=None):
    """Adapt + score every discovered run for this dataset; return result dicts."""
    stem = paths.stem(dataset)
    results = []
    tune_cache = {}
    tune_suffix = f"__tuned-on-{paths.stem(tune_dataset)}" if tune_dataset else ""

    def _finish(approach_key, run_dir, pred_path, suffix=""):
        result = scorer.score(gold, load_json(pred_path), strata)
        save_json(paths.RESULTS_DIR / f"{approach_key}__{run_dir.name}{suffix}.json", result)
        results.append(result)

    # threshold approaches (encoder + decoder): raw data.json -> predictions.json
    #
    # With --tune-on, output goes to predictions.tuned-on-<stem>.json and is
    # scored under a suffixed name, so the self-tuned and cross-tuned operating
    # points coexist instead of one overwriting the other -- and a later plain
    # `run.py --dataset X` cannot clobber a cross-tuned result.
    # ft-encoder rides the same path: it emits P(correct), and adapt_thresh's
    # "pool by min, flag below threshold" is exactly right for that.
    for approach in ("encoder-thresh", "decoder-thresh", "ft-encoder",
                     "decoder-ged-head"):
        for run_dir in _runs(approach, stem):
            data = run_dir / "data.json"
            if not data.exists():
                continue
            tune = None
            if tune_dataset:
                found = _tuning_aligned(approach, tune_dataset, tune_gold,
                                        _signature(run_dir.name, stem), tune_cache)
                if found is None:
                    print(f"[skip] {run_dir.name}: no matching run over "
                          f"{paths.stem(tune_dataset)} to tune on")
                    continue
                tune = (found[0], found[1])
            try:
                pred = run_dir / (f"predictions{tune_suffix.replace('__', '.', 1)}.json"
                                  if tune_dataset else "predictions.json")
                adapt_thresh.adapt(data, run_dir.name + tune_suffix, gold, "auto",
                                   pred, tune=tune)
                _finish(approach, run_dir, pred, tune_suffix)
            except ValueError as exc:
                print(f"[skip] {run_dir.name}: {exc}")

    if tune_dataset:
        # --tune-on only changes how a threshold is chosen, so the approaches
        # that have no threshold have nothing to re-run under it.
        return results

    # GED-by-instruction
    for run_dir in _runs("instruction", stem):
        data = run_dir / "data.json"
        if not data.exists():
            continue
        try:
            pred = run_dir / "predictions.json"
            adapt_instruction.adapt(data, pred, tag=run_dir.name, dataset=dataset)
            _finish("instruction", run_dir, pred)
        except ValueError as exc:
            print(f"[skip] {run_dir.name}: {exc}")

    # confidence classifiers: the trainer already writes canonical predictions
    # (fit on train, predicted on this dev split), so there is nothing to adapt --
    # just score whatever predictions.json each run produced.
    for run_dir in _runs("confidence-clf", stem):
        pred = run_dir / "predictions.json"
        if not pred.exists():
            continue
        try:
            _finish("confidence-clf", run_dir, pred)
        except ValueError as exc:
            print(f"[skip] {run_dir.name}: {exc}")

    # linear-probe: like confidence-clf, a run that already holds canonical
    # predictions just gets scored. The GPU job writes them itself, because the
    # adapter needs torch and reads tens of GB of features that only exist on
    # the compute node -- so probes trained on any dataset, in any language,
    # land here without this scorer needing to know about them.
    probe_runs = _runs("linear-probe", stem)
    for run_dir in probe_runs:
        pred = run_dir / "predictions.json"
        if not pred.exists():
            continue
        try:
            _finish("linear-probe", run_dir, pred)
        except ValueError as exc:
            print(f"[skip] {run_dir.name}: {exc}")

    # Fallback for the original FCE probe, which saved only aggregate metrics:
    # adapt it in place if no run directory exists for it yet. Needs torch, so
    # skip cleanly in the bare scoring environment.
    if stem == "en_fce_dev" and not probe_runs:
        try:
            import adapt_linear_probes
            if adapt_linear_probes.DEFAULT_FEATURE_DIR.exists():
                run_dir = paths.run_dir("linear-probe", dataset, "fce_binary")
                pred = run_dir / "predictions.json"
                adapt_linear_probes.adapt(
                    adapt_linear_probes.DEFAULT_PROBE_DIR,
                    adapt_linear_probes.DEFAULT_FEATURE_DIR,
                    gold, None, pred, "fce_binary",
                )
                _finish("linear-probe", run_dir, pred)
            else:
                print("[skip] linear-probe: feature dir not found")
        except ImportError as exc:
            print(f"[skip] linear-probe: {exc} (needs numpy/torch; run in GPU env)")

    return results


RUN_COL = 62


def _elide(text, width=RUN_COL):
    """Shorten `text` to `width`, cutting from the middle.

    Both ends of a run label carry identifying information -- the model at the
    front, the template and any `tuned-on` suffix at the back -- so a plain
    [:width] can drop exactly the part that distinguishes two rows. A LoRA run
    is named after its adapter directory, which is long enough on its own
    (`gemma-4-12B__en-de-es-ro-ru__ft200k__4000`) to push `variant_a` and
    `variant_b` off the end and render two different systems identically.
    """
    if len(text) <= width:
        return text
    head = (width - 1 + 1) // 2
    return text[:head] + "…" + text[-(width - 1 - head):]


def _print_group_table(results, field):
    """F0.5 per approach x group, with each group's gold base rate alongside.

    The base-rate row is printed deliberately: the operating point that maximises
    F0.5 moves with base rate on its own, so a column that improves down the
    groups is not evidence of anything until it is read against that row.
    """
    scored = [r for r in results if r.get("per_group")]
    if not scored:
        return
    groups = sorted({g for r in scored for g in r["per_group"]})
    width = max((len(g) for g in groups), default=5)
    print(f"\nF0.5 by {field}:")
    header = f"  {'approach / tag':38s}" + "".join(f"{g:>{max(width, 8)}s}" for g in groups)
    print(header)
    base = next(iter(scored))["per_group"]
    print(f"  {'(gold base rate)':38s}"
          + "".join(f"{base[g]['base_rate']*100:>{max(width, 8)}.2f}" for g in groups))
    for result in sorted(scored, key=lambda r: -r["global"]["f0.5"]):
        name = f"{result['approach']} / {result['tag']}"[:38]
        cells = []
        for group in groups:
            g = result["per_group"].get(group)
            cells.append(f"{g['f0.5']*100:>{max(width, 8)}.2f}" if g else f"{'-':>{max(width, 8)}s}")
        print(f"  {name:38s}" + "".join(cells))


def _print_tables(dataset, gold, results, tune_dataset=None):
    stem = paths.stem(dataset)
    # With --tune-on, build_and_score returns only threshold runs, so *every*
    # row carries the same '__tuned-on-<stem>'. That belongs in the heading once
    # rather than in every label, where it would crowd out the model and
    # template -- the two things the reader is actually comparing.
    tune_suffix = f"__tuned-on-{paths.stem(tune_dataset)}" if tune_dataset else ""
    print("\n" + "=" * 82)
    print(f"CROSS-APPROACH COMPARISON  ({dataset}, positive class = 'i', sorted by F0.5)")
    if tune_dataset:
        print(f"thresholds fitted on {tune_dataset}, not on the set being scored")
    print("=" * 82)
    if not results:
        print("(no runs found -- generate each approach's data.json under outputs/ first)")
        return

    results.sort(key=lambda r: r["global"]["f0.5"], reverse=True)
    header = (f"{'approach':18s} {'run':{RUN_COL}s} {'P':>7s} {'R':>7s} "
              f"{'F1':>7s} {'F0.5':>7s} {'cov':>6s}")
    print(header)
    print("-" * len(header))
    for r in results:
        g, c = r["global"], r["coverage"]
        cov = f"{100*c['scored_units']/c['total_units']:.0f}%"
        # Drop what every row genuinely shares -- the dataset prefix and the
        # tuned-on suffix -- but KEEP the timestamp. Re-running the same model
        # and template over one dataset is normal (a prompt or decoding change,
        # a rerun after a bug fix), and those runs differ in nothing but the
        # timestamp: stripping it renders them as identical rows, so the table
        # silently presents two different systems as one.
        label = str(r["tag"]).replace(f"{stem}__", "")
        if tune_suffix:
            label = label.replace(tune_suffix, "")
        label = _elide(label)
        print(f"{r['approach']:18s} {label:{RUN_COL}s} "
              f"{g['precision']*100:6.2f}% {g['recall']*100:6.2f}% "
              f"{g['f1']*100:6.2f}% {g['f0.5']*100:6.2f}% {cov:>6s}")

    # per-type recall only when the gold carries error types (has an m2)
    has_types = any(t for u in gold["units"] for t in u["types"])
    if not has_types:
        print("\n(detection-only dataset: no error types, per-type recall table omitted)")
        return

    print("\n" + "=" * 82)
    print(f"RECALL BY GOLD ERROR TYPE  ({dataset}, best system per approach, by F0.5)")
    print("=" * 82)
    seen, chosen = set(), []
    for r in results:
        if r["approach"] not in seen:
            seen.add(r["approach"])
            chosen.append(r)
    types = list(chosen[0]["per_type_recall"].keys()) if chosen else []
    col_head = f"{'type':16s} {'support':>8s}  " + "  ".join(
        f"{r['approach'][:10]:>10s}" for r in chosen
    )
    print(col_head)
    print("-" * len(col_head))
    for etype in types:
        support = chosen[0]["per_type_recall"][etype]["support"]
        row = f"{str(etype):16s} {support:8d}  "
        cells = []
        for r in chosen:
            s = r["per_type_recall"].get(etype)
            cells.append(f"{s['recall']*100:9.1f}%" if s else f"{'-':>10s}")
        print(row + "  ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True,
                        help="one or more dataset ids, e.g. english/en_fce_dev czech/cs_geccc_dev.")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild sentences + gold even if present.")
    parser.add_argument("--tune-on", default=None,
                        help="dataset id to fit threshold operating points on instead "
                             "of the set being scored, e.g. english/en_c4200_eval5k. "
                             "Affects the encoder/decoder threshold approaches only; "
                             "results are written under a separate '__tuned-on-<stem>' "
                             "name and never overwrite the self-tuned ones.")
    parser.add_argument("--group-by", default=None, metavar="FIELD",
                        help="also break every result down by a .cefr.json field "
                             "(e.g. human_cefr_coarse). Only datasets carrying a "
                             "sidecar support it; the breakdown lands in each "
                             "result json under 'per_group'.")
    args = parser.parse_args()

    tune_gold = ensure_dataset(args.tune_on) if args.tune_on else None

    for dataset in args.dataset:
        if args.tune_on and paths.stem(args.tune_on) == paths.stem(dataset):
            raise SystemExit(
                f"--tune-on {args.tune_on} is the dataset being scored; that is the "
                "self-tuned default, run without --tune-on for it")
        gold = ensure_dataset(dataset, rebuild=args.rebuild)
        strata = scorer.load_strata(dataset, args.group_by) if args.group_by else None
        results = build_and_score(dataset, gold, args.tune_on, tune_gold, strata)
        _print_tables(dataset, gold, results, args.tune_on)
        if strata:
            _print_group_table(results, args.group_by)
    print(f"\nresults in {paths.RESULTS_DIR}")


if __name__ == "__main__":
    main()
