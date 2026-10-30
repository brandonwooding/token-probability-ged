#!/usr/bin/env python3
"""Convert aligned English corrupted/clean sentences to W&I GED M2 and TSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path


POLICY = "exclude noop, UNK, nonzero annotators, and every *:PUNCT; retain U:DET"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(text: str) -> str:
    """Match the train/dev build: strip and collapse every whitespace run."""
    return " ".join(text.split())


def read_sentences(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as stream:
        return [line.rstrip("\n") for line in stream]


def escape_token(token: str) -> str:
    if token == '"':
        return '\\"'
    return token.replace("\\", "\\\\")


def punctuation_type(error_type: str) -> bool:
    return error_type == "PUNCT" or error_type.endswith(":PUNCT")


def parse_m2(path: Path):
    records, current = [], None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if line.startswith("S "):
                if current is not None:
                    records.append(current)
                current = {"tokens": line[2:].split(" "), "edits": []}
            elif line.startswith("A "):
                fields = line[2:].split("|||")
                span = fields[0].split()
                current["edits"].append({
                    "start": int(span[0]), "end": int(span[1]),
                    "type": fields[1], "correction": fields[2],
                    "annotator": fields[-1],
                })
    if current is not None:
        records.append(current)
    return records


def labels(record: dict) -> list[str]:
    output = ["c"] * len(record["tokens"])
    for edit in record["edits"]:
        error_type = edit["type"]
        if (error_type in {"noop", "UNK"} or edit["annotator"] != "0"
                or punctuation_type(error_type)):
            continue
        start, end = edit["start"], edit["end"]
        indices = [start] if start == end else range(start, end)
        for index in indices:
            # Missing material attaches to the token on the right. An insertion
            # beyond the final source token has no token to mark.
            if 0 <= index < len(output):
                output[index] = "i"
    return output


def write_m2(corrupted: list[str], clean: list[str], output: Path,
             batch_size: int):
    import errant

    annotator = errant.load("en")
    temporary = output.with_suffix(output.suffix + ".tmp")
    edit_types = Counter()
    edits_total = 0
    tokens_total = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for begin in range(0, len(corrupted), batch_size):
                source_docs = list(annotator.nlp.pipe(corrupted[begin:begin + batch_size]))
                target_docs = list(annotator.nlp.pipe(clean[begin:begin + batch_size]))
                for row, (source_doc, target_doc) in enumerate(
                        zip(source_docs, target_docs), begin + 1):
                    tokens = [token.text for token in source_doc]
                    if not tokens or any(not token.strip() for token in tokens):
                        raise ValueError(f"line {row}: empty or whitespace source token")
                    edits = annotator.annotate(source_doc, target_doc)
                    tokens_total += len(tokens)
                    stream.write("S " + " ".join(tokens) + "\n")
                    if edits:
                        for edit in edits:
                            stream.write(edit.to_m2() + "\n")
                            edit_types[edit.type] += 1
                            edits_total += 1
                    else:
                        stream.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
                    stream.write("\n")
                print(f"aligned {min(begin + batch_size, len(corrupted))}/{len(corrupted)}",
                      flush=True)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return tokens_total, edits_total, edit_types


def write_tsv(records, output: Path):
    temporary = output.with_suffix(output.suffix + ".tmp")
    positives = tokens = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                record_labels = labels(record)
                for token, label in zip(record["tokens"], record_labels):
                    stream.write(f"{escape_token(token)}\t{label}\n")
                    tokens += 1
                    positives += label == "i"
                stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return tokens, positives


def read_tsv(path: Path):
    units, current = [], []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream, delimiter="\t", quotechar='"', escapechar="\\"):
            if not row or not row[0].strip():
                if current:
                    units.append(current)
                    current = []
                continue
            if len(row) != 2 or row[1] not in {"c", "i"}:
                raise ValueError(f"invalid generated TSV row: {row!r}")
            current.append(('"' if row[0] == r'\"' else row[0], row[1]))
    if current:
        units.append(current)
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrupted", type=Path, required=True,
                        help="one erroneous learner/source sentence per line")
    parser.add_argument("--clean", type=Path, required=True,
                        help="one corrected gold target per corresponding line")
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    outputs = {
        "m2": Path(str(args.out_prefix) + ".m2"),
        "tsv": Path(str(args.out_prefix) + ".tsv"),
        "manifest": Path(str(args.out_prefix) + ".manifest.json"),
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.force:
        raise SystemExit("refusing to overwrite " + ", ".join(existing) + "; pass --force")

    raw_corrupted = read_sentences(args.corrupted)
    raw_clean = read_sentences(args.clean)
    if len(raw_corrupted) != len(raw_clean):
        raise SystemExit(
            f"line-count mismatch: corrupted={len(raw_corrupted)}, clean={len(raw_clean)}")
    if not raw_corrupted:
        raise SystemExit("input files contain no sentences")
    corrupted = [normalise(text) for text in raw_corrupted]
    clean = [normalise(text) for text in raw_clean]
    blank = [index for index, pair in enumerate(zip(corrupted, clean), 1)
             if not pair[0] or not pair[1]]
    if blank:
        raise SystemExit(f"blank sentence at one-based lines {blank[:10]}")

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    m2_tokens, edits, edit_types = write_m2(
        corrupted, clean, outputs["m2"], args.batch_size)
    records = parse_m2(outputs["m2"])
    tsv_tokens, positives = write_tsv(records, outputs["tsv"])
    units = read_tsv(outputs["tsv"])
    if len(records) != len(corrupted) or len(units) != len(corrupted):
        raise AssertionError("generated sentence count changed")
    m2_stream = [token for record in records for token in record["tokens"]]
    tsv_stream = [token for unit in units for token, _ in unit]
    if m2_tokens != tsv_tokens or m2_stream != tsv_stream:
        raise AssertionError("generated M2 and TSV token streams differ")

    manifest = {
        "format_version": 1,
        "language": "English",
        "direction": "corrupted learner/source -> clean corrected target",
        "policy": POLICY,
        "insertion_anchor": "source token on the right; none beyond sentence end",
        "whitespace": "strip ends and collapse runs before tokenisation/alignment",
        "inputs": {
            "corrupted": {"path": str(args.corrupted), "sha256": sha256(args.corrupted)},
            "clean": {"path": str(args.clean), "sha256": sha256(args.clean)},
        },
        "outputs": {
            "m2": {"path": str(outputs["m2"]), "sha256": sha256(outputs["m2"])},
            "tsv": {"path": str(outputs["tsv"]), "sha256": sha256(outputs["tsv"])},
        },
        "counts": {
            "sentences": len(records), "tokens": tsv_tokens,
            "errant_edits": edits, "ged_positive_tokens": positives,
        },
        "normalised_lines": {
            "corrupted": sum(a != b for a, b in zip(raw_corrupted, corrupted)),
            "clean": sum(a != b for a, b in zip(raw_clean, clean)),
        },
        "errant_edit_types": dict(edit_types.most_common()),
        "runtime": {name: version(name) for name in
                    ("errant", "spacy", "en-core-web-sm", "rapidfuzz")},
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {outputs['m2']}")
    print(f"wrote {outputs['tsv']}")
    print(f"wrote {outputs['manifest']}")
    print(f"sentences={len(records)} tokens={tsv_tokens} i={positives}")


if __name__ == "__main__":
    main()
