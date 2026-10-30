"""Check the frequency tables, and measure the only coverage figure that matters.

Two passes:

STRUCTURE -- every table's own invariants. len(counts) == vocab_size,
sum(counts) == total_tokens (the reference implementation gets this one wrong by
summing a Counter's keys), no special token counted, revision pinned.

USE -- the real adequacy test. A table's `vocab_coverage` says how much of the
tokenizer's vocabulary the sample touched, and for these models that number is
mostly meaningless: Apertus, gemma, tiny-aya, mmBERT and XLM-R all carry
multilingual vocabularies, so counting Czech will never reach their Chinese or
Arabic pieces and 60% "coverage" is the expected shape rather than a shortfall.
What decides whether the sample was big enough is the OOV rate over the
sub-tokens the scored runs actually emit -- the tokens a normaliser will have to
look up. That is measured here by joining each archived data.json against the
table for its model and language.

An OOV sub-token is not a bug; it is a token that never occurred in 150M tokens
of web text, which for a learner corpus means a misspelling or a rare form. But
the *rate* says whether the tables are usable, and the OOV tokens' identity says
what a scoring-time floor will be standing in for.

    python verify_tables.py                 # structure only
    python verify_tables.py --runs          # structure + OOV over archived runs
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "standard-eval"))
from sources import MODELS, LANGUAGES, model_id  # noqa: E402
import paths  # noqa: E402
from pooling import flatten  # noqa: E402

TABLES_DIR = SCRIPT_DIR / "tables"


def load_table(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def check_structure(tables_dir):
    print("=== STRUCTURE ===")
    rows, problems = [], []
    for model in MODELS:
        for language in LANGUAGES:
            mid = model_id(model)
            path = tables_dir / f"{mid}__{language}.freqs.json.gz"
            if not path.exists():
                problems.append(f"{path.name}: missing")
                continue
            t = load_table(path)
            counts = t["counts"]
            if len(counts) != t["vocab_size"]:
                problems.append(f"{path.name}: len(counts) != vocab_size")
            if sum(counts) != t["total_tokens"]:
                problems.append(f"{path.name}: sum(counts) != total_tokens")
            if not t["corpus"]["revision"]:
                problems.append(f"{path.name}: unpinned revision")
            rows.append((mid, language, t["total_tokens"], t["unique_tokens"],
                         t["vocab_size"], t["vocab_hash"][:12],
                         path.stat().st_size))

    for mid, lang, total, uniq, vsize, vhash, size in rows:
        print(f"  {mid:28s} {lang:8s} {total/1e6:6.0f}M tok  "
              f"{uniq:>7,}/{vsize:<7,} ({uniq/vsize:5.1%})  {vhash}  "
              f"{size/1e6:4.1f} MB")

    # Tables sharing a vocab hash must agree exactly; if they do not, two models
    # that tokenize identically were counted over different text and the tables
    # are not interchangeable in the way the duplication assumes.
    by_hash = {}
    for model in MODELS:
        for language in LANGUAGES:
            path = tables_dir / f"{model_id(model)}__{language}.freqs.json.gz"
            if path.exists():
                t = load_table(path)
                key = (t["vocab_hash"], language)
                if key in by_hash and by_hash[key][1] != t["total_tokens"]:
                    problems.append(
                        f"{path.name}: same vocab as {by_hash[key][0]} but different totals")
                by_hash.setdefault(key, (path.name, t["total_tokens"]))

    print(f"\n  {len(rows)} tables, {len({h for h, _ in by_hash})} distinct vocabularies")
    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    {p}")
    else:
        print("  all structural checks passed")
    return not problems


def check_runs(tables_dir):
    """OOV rate over the sub-tokens archived runs actually emit."""
    print("\n=== USE (OOV over archived runs) ===")
    approaches = {"decoder-thresh": paths.APPROACH_DIRS["decoder-thresh"],
                  "encoder-thresh": paths.APPROACH_DIRS["encoder-thresh"]}
    cache = {}
    seen = set()
    for approach, root in approaches.items():
        for run in sorted((root / "outputs").glob("*/data.json")):
            meta_path = run.parent / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            mid, dataset = meta.get("model_id"), meta.get("dataset")
            if not mid or not dataset:
                continue
            try:
                language = paths.language(dataset)
            except ValueError:
                continue
            # One run per (model, language) is enough to measure the rate.
            if (mid, language) in seen:
                continue
            table_path = tables_dir / f"{mid}__{language}.freqs.json.gz"
            if not table_path.exists():
                continue
            seen.add((mid, language))

            if table_path not in cache:
                cache[table_path] = load_table(table_path)["counts"]
            counts = cache[table_path]

            total = oov = 0
            for sentence in json.loads(run.read_text()):
                for sub in flatten(sentence):
                    tid = sub.get("token_id")
                    if tid is None:
                        continue
                    total += 1
                    if tid >= len(counts) or counts[tid] == 0:
                        oov += 1
            if total:
                print(f"  {approach:15s} {mid:28s} {language:8s} "
                      f"{oov:>6,}/{total:<8,} sub-tokens OOV ({oov/total:6.3%})")
    if not seen:
        print("  no archived runs matched a table")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, default=TABLES_DIR)
    parser.add_argument("--runs", action="store_true",
                        help="also join archived runs against the tables")
    args = parser.parse_args()

    ok = check_structure(args.tables)
    if args.runs:
        check_runs(args.tables)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
