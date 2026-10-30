#!/usr/bin/env python3
"""Validate every canonical view of one mERRANT-derived C4 split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_tsv(path):
    sentences, sentence = [], []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream, delimiter="\t", quotechar='"', escapechar="\\"):
            if row and row[0].strip():
                if len(row) != 2 or row[1] not in {"c", "i"}:
                    raise ValueError(f"bad TSV row: {row!r}")
                sentence.append((row[0], row[1]))
            elif sentence:
                sentences.append(sentence)
                sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences


def read_m2(path):
    records, current = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("S "):
            if current is not None:
                records.append(current)
            current = {"tokens": line[2:].split(" "), "edits": []}
        elif line.startswith("A "):
            # Correction text may itself contain `|` or even `|||`. Split the
            # two leading fields from the left and the three fixed M2 metadata
            # fields from the right rather than treating every delimiter as a
            # new column.
            span_field, error_type, remainder = line[2:].split("|||", 2)
            correction, _required, _annotator_meta, _annotator = remainder.rsplit("|||", 3)
            span = span_field.split()
            current["edits"].append(
                (int(span[0]), int(span[1]), error_type, correction))
    if current is not None:
        records.append(current)
    return records


def validate(prefix: Path):
    manifest_path = prefix.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = []
    for line in prefix.with_suffix(".pairs.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"malformed pair: {line[:120]!r}")
        pairs.append(tuple(fields))
    tsv = read_tsv(prefix.with_suffix(".tsv"))
    m2 = read_m2(prefix.with_suffix(".m2"))
    with prefix.with_suffix(".jsonl").open(encoding="utf-8") as stream:
        rich = [json.loads(line) for line in stream if line.strip()]
    if not (len(pairs) == len(tsv) == len(m2) == len(rich)):
        raise ValueError(
            f"row mismatch pairs={len(pairs)} tsv={len(tsv)} m2={len(m2)} json={len(rich)}")
    if manifest.get("counts", {}).get("sentences") != len(rich):
        raise ValueError("manifest sentence count does not match data")
    if manifest.get("counts", {}).get("pairs_written") != len(rich):
        raise ValueError("manifest pair count does not match data")
    previous_line = -1
    for row_index, (pair, ged, m2_record, record) in enumerate(zip(pairs, tsv, m2, rich)):
        if pair != (record["source"], record["target"]):
            raise ValueError(f"row {row_index}: pair/JSON mismatch")
        if record.get("schema") != "c4-merrant-rich-v1" or record.get("alignment") != {
                "authority": "mERRANT", "policy": "exclude-noop-unk-punct"}:
            raise ValueError(f"row {row_index}: unexpected rich-record contract")
        source_tokens = [word["text"] for word in record["source_words"]]
        target_tokens = [word["text"] for word in record["target_words"]]
        if any(any(character.isspace() for character in token)
               for token in source_tokens + target_tokens):
            raise ValueError(f"row {row_index}: token text crosses whitespace")
        if source_tokens != [token for token, _ in ged] or source_tokens != m2_record["tokens"]:
            raise ValueError(f"row {row_index}: source token views differ")
        if record["labels"] != [label for _, label in ged]:
            raise ValueError(f"row {row_index}: JSON/TSV labels differ")
        if record["source_line_index"] <= previous_line:
            raise ValueError(f"row {row_index}: source line indices are not increasing")
        previous_line = record["source_line_index"]

        projected = ["c"] * len(source_tokens)
        reconstructed, cursor = [], 0
        json_m2 = []
        for edit in record["edits"]:
            ss, se = edit["source_start"], edit["source_end"]
            ts, te = edit["target_start"], edit["target_end"]
            if not (0 <= ss <= se <= len(source_tokens) and
                    0 <= ts <= te <= len(target_tokens)):
                raise ValueError(f"row {row_index}: out-of-range edit {edit}")
            if ss < cursor:
                raise ValueError(f"row {row_index}: overlapping/out-of-order edits")
            reconstructed.extend(source_tokens[cursor:ss])
            reconstructed.extend(target_tokens[ts:te])
            cursor = se
            if not edit["included_in_ged"] and edit["label_indices"]:
                raise ValueError(f"row {row_index}: excluded edit carries GED labels")
            base_type = edit["error_type"].split(":")[-1]
            should_include = edit["error_type"] not in {"noop", "UNK"} and base_type not in {
                "UNK", "PUNCT"}
            if edit["included_in_ged"] != should_include:
                raise ValueError(f"row {row_index}: edit filtering policy drift")
            for index in edit["label_indices"]:
                projected[index] = "i"
            correction = " ".join(target_tokens[ts:te])
            json_m2.append((ss, se, edit["error_type"], correction))
        reconstructed.extend(source_tokens[cursor:])
        if reconstructed != target_tokens:
            raise ValueError(f"row {row_index}: edits do not reconstruct target tokens")
        if projected != record["labels"]:
            raise ValueError(f"row {row_index}: edit projection does not reproduce labels")
        actual_m2 = [edit for edit in m2_record["edits"] if edit[2] != "noop"]
        if actual_m2 != json_m2:
            raise ValueError(f"row {row_index}: M2/JSON edits differ")
    print(json.dumps({"prefix": str(prefix), "sentences": len(rich),
                      "tokens": sum(len(row) for row in tsv),
                      "incorrect": sum(label == "i" for row in tsv for _, label in row)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path, help="path without .tsv/.m2/.jsonl suffix")
    args = parser.parse_args()
    validate(args.prefix)


if __name__ == "__main__":
    main()
