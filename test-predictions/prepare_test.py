#!/usr/bin/env python3
"""Reconstruct all configured one-column test sets for model inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402
import reconstruct_from_tsv  # noqa: E402


TEST_DATASETS = (
    "czech/cs_geccc_test_unlabelled",
    "english/en_fce_test_unlabelled",
    "english/en_realec_test_unlabelled",
    "english/en_wi2024_test_unlabelled",
    "german/de_falko-merlin_test_unlabelled",
    "italian/it_merlin_test_unlabelled",
    "swedish/sv_swell_test_unlabelled",
)


def prepare(dataset: str, rebuild: bool = False):
    output = paths.sentences(dataset)
    units = reconstruct_from_tsv.build(paths.tsv(dataset), unlabelled=True)
    if any("labels" in unit for unit in units):
        raise AssertionError("unlabelled reconstruction unexpectedly contains labels")
    payload = json.dumps(units, indent=2, ensure_ascii=False) + "\n"
    if output.exists() and not rebuild:
        if output.read_text(encoding="utf-8") != payload:
            raise ValueError(f"{output} exists but differs; inspect it or pass --rebuild")
        action = "checked"
    else:
        output.write_text(payload, encoding="utf-8")
        action = "wrote"
    print(
        f"{action} {output}  sentences={len(units)} "
        f"tokens={sum(len(unit['tokens']) for unit in units)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    for dataset in TEST_DATASETS:
        prepare(dataset, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
