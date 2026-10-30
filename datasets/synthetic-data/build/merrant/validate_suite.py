#!/usr/bin/env python3
"""Validate every canonical split and prove nested fine-tuning identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate import validate

LANGUAGES = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "ro": "romanian",
    "ru": "russian",
    "it": "italian",
    "sv": "swedish",
    "zh": "chinese",
}

RANGES = {
    "eval5k": (0, 5_000),
    "val5k": (55_000, 60_000),
    "train10k": (60_000, 70_000),
    "train20k": (60_000, 80_000),
    "train50k": (60_000, 110_000),
    "train100k": (60_000, 160_000),
}
MERRANT_COMMIT = "fc8409b89317efac81dba0ba8788c744d19ffb2c"


def read_ids(prefix: Path):
    ids = []
    with prefix.with_suffix(".jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                ids.append((record["pair_id"], record["source_line_index"],
                            record["labels"], record["edits"]))
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--langs", default="en,de,es,ro,ru")
    parser.add_argument("--corpus", default="c4200",
                        help="stem tag written by build.py --corpus")
    args = parser.parse_args()
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    for lang in langs:
        folder, stem = LANGUAGES[lang], f"{lang}_{args.corpus}"
        prefixes = {name: args.data_root / folder / f"{stem}_{name}" for name in RANGES}
        for name, prefix in prefixes.items():
            manifest = json.loads(prefix.with_suffix(".manifest.json").read_text())
            found = manifest["source_line_range"]
            expected = {"start": RANGES[name][0], "end": RANGES[name][1]}
            if found != expected:
                raise ValueError(f"{prefix}: range {found} != {expected}")
            annotator = manifest.get("annotator", {})
            if (annotator.get("implementation_commit") != MERRANT_COMMIT
                    or not isinstance(
                        annotator.get("whitespace_crossing_tokenizer_rules_removed"), int)):
                raise ValueError(f"{prefix}: incomplete annotator provenance")
            validate(prefix)

        ten = read_ids(prefixes["train10k"])
        twenty = read_ids(prefixes["train20k"])
        fifty = read_ids(prefixes["train50k"])
        hundred = read_ids(prefixes["train100k"])
        if (ten != twenty[:len(ten)] or twenty != fifty[:len(twenty)]
                or fifty != hundred[:len(fifty)]):
            raise ValueError(f"{lang}: nested train JSON records are not identical prefixes")
        print(json.dumps({"language": lang, "suite": "valid",
                          "nested": [len(ten), len(twenty), len(fifty), len(hundred)]}))


if __name__ == "__main__":
    main()
