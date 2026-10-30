#!/usr/bin/env python3
"""Build and validate MultiGED test prediction TSVs.

A threshold can be supplied directly, loaded from the frozen system manifest,
or read from an existing standard-eval development result.  This module never
fits a threshold and never reads private test gold.  Raw model output is the
usual sentence-level data.json containing P(correct) per subtoken.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from itertools import zip_longest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "standard-eval"))
from adapt_thresh import pool_word_probs  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_lines(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.readlines()


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


def _content(line: str) -> str:
    ending = _line_ending(line)
    return line[:-len(ending)] if ending else line


def _parse_row(line: str, path: Path, line_no: int):
    content = _content(line)
    if not content:
        return []
    try:
        return next(csv.reader([content], delimiter="\t", quotechar='"', escapechar="\\"))
    except csv.Error as exc:
        raise ValueError(f"{path}:{line_no}: invalid TSV: {exc}") from exc


def source_units(path: Path):
    """Read labelled or unlabelled TSV units, retaining each raw token field."""
    units = []
    current = []
    lines = _read_lines(path)
    for line_no, line in enumerate(lines, start=1):
        row = _parse_row(line, path, line_no)
        if not row:
            if current:
                units.append(current)
                current = []
            continue
        if len(row) not in (1, 2):
            raise ValueError(f"{path}:{line_no}: expected TOKEN or TOKEN<TAB>LABEL")
        raw_token = _content(line).split("\t", 1)[0]
        token = '"' if row[0] == r'\"' else row[0]
        current.append({"token": token, "raw_token": raw_token, "line_no": line_no})
    if current:
        units.append(current)
    return units, lines


def align_source(source_path: Path, sentence_units):
    raw_units, lines = source_units(source_path)
    if len(raw_units) != len(sentence_units):
        raise ValueError(
            f"{source_path}: {len(raw_units)} TSV sentences but sentences JSON has "
            f"{len(sentence_units)}"
        )
    for unit_id, (raw, sentence) in enumerate(zip(raw_units, sentence_units)):
        raw_tokens = [record["token"] for record in raw]
        if raw_tokens != sentence["tokens"]:
            raise ValueError(f"{source_path}: sentence {unit_id} token mismatch")
    return raw_units, lines


def probabilities_to_predictions(data_path: Path, sentence_units, threshold: float):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    raw = load_json(data_path)
    if len(raw) != len(sentence_units):
        raise ValueError(
            f"{data_path}: {len(raw)} model sentences but input has {len(sentence_units)}"
        )
    units = []
    for unit_id, (subtokens, sentence) in enumerate(zip(raw, sentence_units)):
        probabilities = pool_word_probs(subtokens, len(sentence["tokens"]))
        if probabilities is None:
            raise ValueError(f"{data_path}: sentence {unit_id} has an out-of-range word_id")
        if len(probabilities) != len(sentence["tokens"]):
            raise AssertionError(f"sentence {unit_id}: pooled probability length mismatch")
        units.append({
            "unit_id": unit_id,
            "tokens": sentence["tokens"],
            "predicted_labels": ["i" if p < threshold else "c" for p in probabilities],
        })
    return units


def write_submission(source_path: Path, sentence_units, prediction_units, out_path: Path):
    """Copy raw token fields and blank lines, replacing/adding only the label."""
    _, lines = align_source(source_path, sentence_units)
    labels = iter(
        label for unit in prediction_units for label in unit["predicted_labels"]
    )
    output = []
    consumed = 0
    for line_no, line in enumerate(lines, start=1):
        row = _parse_row(line, source_path, line_no)
        if not row:
            output.append(line)
            continue
        try:
            label = next(labels)
        except StopIteration as exc:
            raise ValueError("fewer predictions than source tokens") from exc
        raw_token = _content(line).split("\t", 1)[0]
        output.append(f"{raw_token}\t{label}{_line_ending(line)}")
        consumed += 1
    try:
        next(labels)
    except StopIteration:
        pass
    else:
        raise ValueError("more predictions than source tokens")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("".join(output))
    return consumed


def validate_submission(source_path: Path, hypothesis_path: Path):
    source_lines = _read_lines(source_path)
    hyp_lines = _read_lines(hypothesis_path)
    tokens = sentences = 0
    in_sentence = False
    for line_no, pair in enumerate(zip_longest(source_lines, hyp_lines), start=1):
        source_line, hyp_line = pair
        if source_line is None or hyp_line is None:
            raise ValueError(
                f"line-count mismatch: source={len(source_lines)} hypothesis={len(hyp_lines)}"
            )
        source_row = _parse_row(source_line, source_path, line_no)
        hyp_row = _parse_row(hyp_line, hypothesis_path, line_no)
        if not source_row:
            if hyp_row:
                raise ValueError(f"{hypothesis_path}:{line_no}: expected a blank line")
            if in_sentence:
                sentences += 1
                in_sentence = False
            continue
        if not hyp_row:
            raise ValueError(f"{hypothesis_path}:{line_no}: unexpected blank line")
        if len(hyp_row) != 2:
            raise ValueError(f"{hypothesis_path}:{line_no}: expected exactly two columns")
        if source_row[0] != hyp_row[0]:
            raise ValueError(f"{hypothesis_path}:{line_no}: token differs from source")
        if hyp_row[1] not in ("c", "i"):
            raise ValueError(f"{hypothesis_path}:{line_no}: label must be c or i")
        tokens += 1
        in_sentence = True
    if in_sentence:
        sentences += 1
    return {"lines": len(source_lines), "sentences": sentences, "tokens": tokens}


def build_one(dataset: str, data_path: Path, threshold: float, system: str,
              output_root: Path, source_result=None, threshold_selection="fixed",
              tuned_on=None, dev_metrics=None, threshold_data=None):
    source_path = paths.tsv(dataset)
    sentences_path = paths.sentences(dataset)
    if not sentences_path.exists():
        raise FileNotFoundError(
            f"missing {sentences_path}; reconstruct the unlabelled dataset first"
        )
    sentence_units = load_json(sentences_path)
    align_source(source_path, sentence_units)
    prediction_units = probabilities_to_predictions(data_path, sentence_units, threshold)

    stem = paths.stem(dataset).replace("_unlabelled", "")
    system_dir = output_root / system
    tsv_path = system_dir / f"{stem}.tsv"
    json_path = system_dir / f"{stem}.predictions.json"
    canonical = {
        "approach": "fixed-prob-threshold",
        "tag": system,
        "meta": {
            "dataset": dataset,
            "source": str(data_path),
            "threshold": threshold,
            "threshold_selection": threshold_selection,
            "tuned_on": tuned_on,
            "threshold_data": str(threshold_data) if threshold_data else None,
            "dev_metrics": dev_metrics,
            "source_result": source_result,
            "pooling": "min sub-token probability per word",
            "skipped_units": 0,
        },
        "units": prediction_units,
    }
    save_json(json_path, canonical)
    write_submission(source_path, sentence_units, prediction_units, tsv_path)
    summary = validate_submission(source_path, tsv_path)
    print(
        f"wrote {tsv_path}  threshold={threshold:.6g}  "
        f"sentences={summary['sentences']} tokens={summary['tokens']}"
    )
    return tsv_path, json_path


def discover_data(dataset: str, config: dict, entry: dict):
    approach = config["approach"]
    checkpoint = config["checkpoint"]
    tag = entry.get("tag", config.get("tag"))
    stem = paths.stem(dataset)
    prefix = f"{stem}__{checkpoint}"
    if tag:
        prefix += f"__{tag}"
    output_root = paths.APPROACH_DIRS[approach] / "outputs"
    candidates = sorted(
        run / "data.json" for run in output_root.glob(f"{prefix}__*")
        if (run / "data.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no {approach} data.json for {dataset}, checkpoint={checkpoint}, tag={tag}"
        )
    return candidates[-1]


def instruction_predictions(data_path: Path, sentence_units):
    """GED-by-instruction data.json -> prediction units.

    A sentence whose predicted_labels is null (invalid JSON / wrong length) is
    emitted as all-"c", matching standard-eval/adapt_instruction.py. The count is
    returned so a run with failures is visible rather than silently scored.
    """
    raw = load_json(data_path)
    if len(raw) != len(sentence_units):
        raise ValueError(
            f"{data_path}: {len(raw)} model sentences but input has {len(sentence_units)}"
        )
    units, failures = [], 0
    for unit_id, (sent, source) in enumerate(zip(raw, sentence_units)):
        if sent["tokens"] != source["tokens"]:
            raise ValueError(f"{data_path}: sentence {unit_id} token mismatch")
        labels = sent.get("predicted_labels")
        if labels is None:
            failures += 1
            labels = ["c"] * len(source["tokens"])
        units.append({"unit_id": unit_id, "tokens": source["tokens"],
                      "predicted_labels": labels})
    return units, failures


def build_instruction_all(model_id: str, datasets, output_root: Path, system: str):
    """Build and validate TSVs from GED-by-instruction runs (no threshold)."""
    root = paths.APPROACH_DIRS["instruction"] / "outputs"
    built = []
    for dataset in datasets:
        runs = sorted(
            run / "data.json" for run in root.glob(f"{paths.stem(dataset)}__{model_id}__*")
            if (run / "data.json").exists() and "__rep" not in run.name
        )
        if not runs:
            raise FileNotFoundError(f"no instruction run for {dataset}, model={model_id}")
        data_path = runs[-1]
        source_path = paths.tsv(dataset)
        sentence_units = load_json(paths.sentences(dataset))
        align_source(source_path, sentence_units)
        prediction_units, failures = instruction_predictions(data_path, sentence_units)
        stem = paths.stem(dataset).replace("_unlabelled", "")
        tsv_path = output_root / system / f"{stem}.tsv"
        save_json(output_root / system / f"{stem}.predictions.json", {
            "approach": "ged-by-instruction", "tag": system,
            "meta": {"dataset": dataset, "source": str(data_path), "model": model_id,
                     "parse_failures_as_all_c": failures, "threshold": None},
            "units": prediction_units,
        })
        write_submission(source_path, sentence_units, prediction_units, tsv_path)
        summary = validate_submission(source_path, tsv_path)
        flagged = sum(l == "i" for u in prediction_units for l in u["predicted_labels"])
        print(f"wrote {tsv_path}  sentences={summary['sentences']} tokens={summary['tokens']} "
              f"flagged={flagged / summary['tokens'] * 100:.1f}% parse_failures={failures}")
        built.append(tsv_path)
    return built


def threshold_from_result(result_path: Path):
    """Read a threshold already selected and recorded by standard-eval."""
    result = load_json(result_path)
    meta = result.get("meta", {})
    threshold = meta.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(f"{result_path}: missing numeric meta.threshold")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"{result_path}: threshold is outside [0, 1]")
    return float(threshold), result


def build_all(manifest_path: Path, system: str, output_root: Path):
    manifest = load_json(manifest_path)
    try:
        config = manifest["systems"][system]
    except KeyError as exc:
        raise ValueError(f"unknown system {system!r} in {manifest_path}") from exc
    built = []
    for dataset, entry in config["datasets"].items():
        threshold = entry.get("threshold")
        if threshold is None:
            raise ValueError(
                f"{system}/{dataset}: threshold is not frozen; run freeze_thresholds.py"
            )
        data_path = discover_data(dataset, config, entry)
        built.append(build_one(
            dataset, data_path, float(threshold), system, output_root,
            source_result=entry.get("source_result"),
        )[0])
    destination = output_root / system / "thresholds.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, destination)
    print(f"validated {len(built)} submission files; copied manifest to {destination}")
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build one prediction TSV")
    build.add_argument("--dataset", required=True)
    build.add_argument("--data", type=Path,
                       help="test data.json; default: discover the matching run")
    threshold_source = build.add_mutually_exclusive_group()
    threshold_source.add_argument("--threshold", type=float,
                                  help="fixed threshold (default: manifest value)")
    threshold_source.add_argument(
        "--threshold-result", type=Path, metavar="RESULT_JSON",
        help="standard-eval dev result containing the selected threshold")
    build.add_argument("--system", required=True)
    build.add_argument("--manifest", type=Path, default=HERE / "thresholds.json")
    build.add_argument("--output-root", type=Path, default=HERE / "submissions")

    batch = subparsers.add_parser("build-all", help="build every dataset for one system")
    batch.add_argument("--manifest", type=Path, default=HERE / "thresholds.json")
    batch.add_argument("--system", required=True)
    batch.add_argument("--output-root", type=Path, default=HERE / "submissions")

    instr = subparsers.add_parser(
        "build-instruction", help="build every MultiGED test TSV from GED-by-instruction runs")
    instr.add_argument("--model-id", default="gemma-4-12B-it")
    instr.add_argument("--system", default=None, help="output folder; default: <model-id>-instruction")
    instr.add_argument("--output-root", type=Path, default=HERE / "submissions")

    validate = subparsers.add_parser("validate", help="strictly validate one TSV")
    validate.add_argument("--source", type=Path, required=True)
    validate.add_argument("--hyp", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build":
        manifest = load_json(args.manifest)
        try:
            config = manifest["systems"][args.system]
            entry = config["datasets"][args.dataset]
        except KeyError as exc:
            parser.error(
                f"no configuration for {args.system}/{args.dataset} in {args.manifest}"
            )
        data_path = args.data or discover_data(args.dataset, config, entry)
        if args.threshold_result:
            threshold, result = threshold_from_result(args.threshold_result)
            meta = result["meta"]
            build_one(
                args.dataset, data_path, threshold, args.system, args.output_root,
                source_result=str(args.threshold_result),
                threshold_selection="read from existing standard-eval development result",
                tuned_on=meta.get("tuned_on") or meta.get("dataset"),
                dev_metrics=result.get("global"), threshold_data=meta.get("source"),
            )
        else:
            threshold = args.threshold if args.threshold is not None else entry.get("threshold")
            if threshold is None:
                parser.error("no frozen threshold; use --threshold or --threshold-result")
            build_one(
                args.dataset, data_path, float(threshold), args.system, args.output_root,
                source_result=entry.get("source_result"),
                threshold_selection=(
                    "explicit fixed threshold" if args.threshold is not None
                    else "frozen in threshold manifest from labelled development data"
                ),
                tuned_on=entry.get("dev_dataset") if args.threshold is None else None,
                dev_metrics=entry.get("dev_metrics") if args.threshold is None else None,
                threshold_data=entry.get("source_data") if args.threshold is None else None,
            )
    elif args.command == "build-instruction":
        build_instruction_all(
            args.model_id,
            ["czech/cs_geccc_test_unlabelled", "english/en_fce_test_unlabelled",
             "english/en_realec_test_unlabelled", "german/de_falko-merlin_test_unlabelled",
             "italian/it_merlin_test_unlabelled", "swedish/sv_swell_test_unlabelled"],
            args.output_root, args.system or f"{args.model_id}-instruction")
    elif args.command == "build-all":
        build_all(args.manifest, args.system, args.output_root)
    else:
        print(json.dumps(validate_submission(args.source, args.hyp), indent=2))


if __name__ == "__main__":
    main()
