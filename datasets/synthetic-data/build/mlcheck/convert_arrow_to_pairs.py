#!/usr/bin/env python3
"""Convert the Ubisoft MultiLingualCheck Arrow datasets to sentence-pair TSV.

The published corpus is one HuggingFace dataset per language holding two string
columns, `correct` and `incorrect`.  `../merrant/build.py` consumes the
`<lang>.sentence_pairs.tsv` dialect established by C4_200M, where column one is
the corrupted source and column two is its clean target -- the opposite order
to the Arrow schema, so the columns are swapped here.

Rows keep their published order, which is already well shuffled, so the
builder's contiguous split ranges stay representative.  Text is written
verbatim: whitespace normalisation and the invalid-Unicode filter belong to
build.py, and running them there keeps both corpora on one cleaning path.

Unlike C4_200M, whose edits guarantee at least one difference per pair, this
corpus contains pairs whose two sides are identical (3.6% of Czech up to 44.2%
of English).  They are kept by default: they carry the sentence-level error
base rate, which C4_200M lacks.  `--drop-identical` removes them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[1] / "sources" / "mlcheck" / "synthetic_all"
DEFAULT_OUT = HERE.parents[1] / "sources" / "mlcheck" / "pairs"

# Mirrors build.py's normalise(). Used only to decide whether the two sides of a
# pair are identical as the builder will see them; the text written is verbatim.
def normalise(text: str) -> str:
    return " ".join(text.split())


def read_table(lang_dir: Path) -> pa.Table:
    """Reads one HuggingFace Arrow dataset directory into a single table."""
    shards = sorted(lang_dir.glob("data-*.arrow"))
    if not shards:
        raise FileNotFoundError(f"no data-*.arrow under {lang_dir}")
    tables = []
    for shard in shards:
        with pa.memory_map(str(shard)) as handle:
            try:
                reader = ipc.open_stream(handle)
            except pa.ArrowInvalid:
                handle.seek(0)
                reader = ipc.open_file(handle)
            tables.append(reader.read_all())
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    missing = {"correct", "incorrect"} - set(table.schema.names)
    if missing:
        raise ValueError(f"{lang_dir}: missing column(s) {sorted(missing)}")
    return table


def convert(lang: str, source_root: Path, out_root: Path, *,
            drop_identical: bool, force: bool) -> dict:
    lang_dir = source_root / lang
    final = out_root / f"{lang}.sentence_pairs.tsv"
    manifest_path = out_root / f"{lang}.convert.json"
    if final.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {final}; pass --force")

    table = read_table(lang_dir)
    correct = table.column("correct").to_pylist()
    incorrect = table.column("incorrect").to_pylist()

    counts = {"rows_read": len(correct), "written": 0, "identical": 0,
              "identical_dropped": 0, "skipped_empty": 0, "skipped_separator": 0}

    out_root.mkdir(parents=True, exist_ok=True)
    temporary = final.with_suffix(final.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for target, source in zip(correct, incorrect):
            # A tab or newline in either side would silently reshape the TSV.
            # The published corpus has none; count rather than mangle any.
            if any(character in field
                   for field in (source, target) for character in ("\t", "\n", "\r")):
                counts["skipped_separator"] += 1
                continue
            if not source.strip() or not target.strip():
                counts["skipped_empty"] += 1
                continue
            if normalise(source) == normalise(target):
                counts["identical"] += 1
                if drop_identical:
                    counts["identical_dropped"] += 1
                    continue
            stream.write(source + "\t" + target + "\n")
            counts["written"] += 1
    temporary.replace(final)

    manifest = {
        "schema": "mlcheck-pairs-v1",
        "language": lang,
        "source": str(lang_dir.resolve()),
        "output": final.name,
        "column_order": "source (incorrect) TAB target (correct)",
        "row_order": "as published; not reshuffled",
        "identical_pairs": "dropped" if drop_identical else "kept",
        "counts": counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", default="de,en,es,it,ru,sv,zh")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--drop-identical", action="store_true",
                        help="omit pairs whose two sides match after whitespace collapse")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    langs = [item.strip() for item in args.langs.split(",") if item.strip()]
    for lang in langs:
        manifest = convert(lang, args.source_root, args.out_root,
                           drop_identical=args.drop_identical, force=args.force)
        print(json.dumps({"language": lang, **manifest["counts"]}), flush=True)


if __name__ == "__main__":
    main()
