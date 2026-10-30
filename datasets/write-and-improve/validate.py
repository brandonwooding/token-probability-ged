#!/usr/bin/env python3
"""Re-read a finished W&I dataset and check the invariants that matter.

Independent of build_wi_ged.py on purpose: it reads only the finished artefacts,
so a bug in the builder's bookkeeping cannot hide here.

    python validate.py                # both splits
    python validate.py --split dev
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUT_DIR = HERE / "english"
# The licensed W&I 2024 v2 release, unpacked by hand (see README.md).
CORPUS = (HERE / "source" / "write-and-improve-corpus-2024-v2")


def read_tsv_units(path):
    """Exactly as reconstruct_from_tsv.read_sentences parses it."""
    units, current = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t", quotechar='"', escapechar="\\"):
            if not row or not row[0].strip():
                if current:
                    units.append(current)
                    current = []
                continue
            current.append(('"' if row[0] == r"\"" else row[0], row[1]))
    if current:
        units.append(current)
    return units


def m2_records(path):
    records, current = [], None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("S "):
                if current is not None:
                    records.append(current)
                current = {"tokens": line[2:].rstrip("\n").split(" "), "edits": []}
            elif line.startswith("A "):
                fields = line[2:].rstrip("\n").split("|||")
                span = fields[0].split()
                current["edits"].append(
                    (int(span[0]), int(span[1]), fields[1], fields[-1]))
    if current is not None:
        records.append(current)
    return records


def expected_labels(record):
    """Independent implementation of the W&I `exclude-punct` contract."""
    labels = ["c"] * len(record["tokens"])
    for start, end, error_type, annotator in record["edits"]:
        if (error_type in {"noop", "UNK"} or annotator != "0"
                or error_type == "PUNCT" or error_type.endswith(":PUNCT")):
            continue
        indices = [start] if start == end else range(start, end)
        for index in indices:
            if 0 <= index < len(labels):
                labels[index] = "i"
    return labels


def check(label, ok, detail=""):
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
    return ok


def validate_split(split, tokenizer):
    stem = f"en_wi2024_{split}"
    print(f"\n== {stem} ==")
    tsv = OUT_DIR / f"{stem}.tsv"
    m2 = OUT_DIR / f"{stem}.m2"
    cefr = OUT_DIR / f"{stem}.cefr.json"
    sentences = OUT_DIR / f"{stem}.sentences.json"
    gold = OUT_DIR / f"{stem}.gold.json"

    passed = True
    units = read_tsv_units(tsv)
    tokens = [t for u in units for t, _ in u]
    labels = [l for u in units for _, l in u]
    n_err = labels.count("i")
    print(f"  units={len(units)} tokens={len(tokens)} i={n_err} "
          f"({100 * n_err / len(tokens):.2f}%)")

    # 1. TSV survives its own reader: no token lost to the blank-field rule and
    #    no backslash eaten by the csv escapechar.
    raw_rows = sum(1 for line in open(tsv, encoding="utf-8") if line.strip())
    passed &= check("TSV round-trips its reader (no token lost)",
                    raw_rows == len(tokens), f"{raw_rows} rows vs {len(tokens)} tokens")
    passed &= check("labels are only c/i", set(labels) <= {"c", "i"})
    passed &= check("no whitespace/empty token", all(t.strip() for t in tokens))

    # 2. TSV and M2 are the same token stream (the TSV is derived from the M2).
    records = m2_records(m2)
    m2_toks = [t for record in records for t in record["tokens"]]
    passed &= check("TSV tokens == M2 tokens", tokens == m2_toks,
                    f"{len(tokens)} vs {len(m2_toks)}")
    expected = [label for record in records for label in expected_labels(record)]
    passed &= check("TSV labels == exclude-punct policy", labels == expected,
                    f"expected {expected.count('i')} i, found {n_err}")

    # 3. Generated artefacts agree, if they have been built.
    if sentences.exists():
        sent = json.loads(sentences.read_text())
        s_toks = [t for u in sent for t in u["tokens"]]
        passed &= check("sentences.json tokens match", s_toks == tokens,
                        f"{len(s_toks)} vs {len(tokens)}")
        exact = sum(1 for u in sent
                    if [t.text for t in tokenizer(u["text"]) if not t.text.isspace()]
                    == u["tokens"])
        rate = 100 * exact / len(sent)
        passed &= check("detokenise -> retokenise recovers tokens",
                        rate >= 99.0, f"{exact}/{len(sent)} = {rate:.2f}%")
    if gold.exists():
        g = json.loads(gold.read_text())
        g_toks = [t for u in g["units"] for t in u["tokens"]]
        g_lab = [l for u in g["units"] for l in u["labels"]]
        passed &= check("gold.json tokens+labels match", g_toks == tokens and g_lab == labels)
        untyped = sum(1 for u in g["units"]
                      for l, ty in zip(u["labels"], u["types"])
                      if l == "i" and ty in (None, "UNSPEC"))
        passed &= check("every gold 'i' carries an error type", untyped == 0,
                        f"{untyped} untyped")

    # 4. CEFR sidecar lines up with the units and is complete.
    meta = json.loads(cefr.read_text())
    passed &= check("cefr n_units == TSV units", meta["n_units"] == len(units),
                    f"{meta['n_units']} vs {len(units)}")
    ids = [u["unit_id"] for u in meta["units"]]
    passed &= check("cefr unit_ids are 0..n-1 in order", ids == list(range(len(units))))
    no_level = sum(1 for u in meta["units"] if u["human_cefr"] is None)
    passed &= check("every unit has a human CEFR grade", no_level == 0,
                    f"{no_level} missing")

    # 5. The essay ids the sidecar carries must be the corpus's own, in order.
    src_ids = [l.strip() for l in open(
        CORPUS / "user-prompt-final-versions"
        / f"en-writeandimprove2024-final-versions-{split}-sentences.ids",
        encoding="utf-8")]
    passed &= check("cefr essay_ids == corpus .ids",
                    [u["essay_id"] for u in meta["units"]] == src_ids)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "train"], action="append")
    args = parser.parse_args()
    import spacy
    tokenizer = spacy.blank("en").tokenizer
    ok = True
    for split in args.split or ["dev", "train"]:
        ok &= validate_split(split, tokenizer)
    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
