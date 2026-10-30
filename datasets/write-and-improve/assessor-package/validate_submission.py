#!/usr/bin/env python3
"""Strictly validate a GED hypothesis against a reference TSV."""
from __future__ import annotations

import argparse
import csv
from itertools import zip_longest
from pathlib import Path


def read(path: Path):
    units, current = [], []
    with path.open(newline="", encoding="utf-8") as stream:
        for line, row in enumerate(
                csv.reader(stream, delimiter="\t", quotechar='"', escapechar="\\"), 1):
            if not row or not row[0].strip():
                if current:
                    units.append(current)
                    current = []
                continue
            if len(row) != 2 or row[1] not in {"c", "i"}:
                raise ValueError(f"{path}:{line}: expected TOKEN<TAB>c|i")
            current.append('"' if row[0] == r'\"' else row[0])
    if current:
        units.append(current)
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyp", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    args = parser.parse_args()
    hyp, ref = read(args.hyp), read(args.ref)
    tokens = 0
    for unit_id, pair in enumerate(zip_longest(hyp, ref)):
        h_unit, r_unit = pair
        if h_unit is None or r_unit is None:
            raise ValueError(f"sentence-count mismatch: hyp={len(hyp)}, ref={len(ref)}")
        if len(h_unit) != len(r_unit):
            raise ValueError(f"sentence {unit_id}: token-count mismatch")
        for token_id, (h_token, r_token) in enumerate(zip(h_unit, r_unit)):
            if h_token != r_token:
                raise ValueError(f"sentence {unit_id}, token {token_id}: token mismatch")
            tokens += 1
    print(f"valid: sentences={len(ref)} tokens={tokens}")


if __name__ == "__main__":
    main()
