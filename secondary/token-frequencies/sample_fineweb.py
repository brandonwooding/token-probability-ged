"""Freeze a fixed sample of web documents per language. The only network step.

Why a frozen sample rather than streaming per model: there are 10 models and 5
languages, so counting straight off the Hub means 50 streaming passes over the
same text. Sampling once per language and tokenizing the saved file 10 times
costs 5 passes, runs offline afterwards, and -- the reason that actually
matters -- guarantees every model's table is counted from the *identical*
corpus, so a difference between two tables is a difference between tokenizers
and nothing else.

Sampling is a deterministic two-level stride, not `.shuffle(seed=...)`. A
streaming shuffle permutes shard order and buffers within it, so the sample it
draws depends on the repo's file listing at the time and cannot be reproduced
later. It is also skewed: FineWeb shards are grouped by CommonCrawl dump
(`000_*` .. `004_*`), so reading off the front of the corpus buys one crawl's
worth of the web. Instead we take an even spread of shards (covering every dump
group) and, within each, an even spread of row groups (covering the whole shard
rather than its first few thousand rows). Both strides land in the manifest, so
the draw repeats exactly.

Read through pyarrow over HfFileSystem rather than `datasets`. Same bytes, but
`load_dataset(..., streaming=True)` spends ~39s resolving each shard against the
Hub before yielding a row -- 50 minutes of pure overhead across a full draw --
where `pq.ParquetFile` opens the same shard in 1.6s and, unlike the streaming
reader, can address individual row groups, which is what makes the second-level
stride free.

Documents are weighted uniformly -- FineWeb-2 rows carry `minhash_cluster_size`
from near-duplicate collapsing, and the "rehydration" recipe upweights by it,
but a unigram frequency estimate wants the deduplicated distribution and that is
what counting each retained document once gives.

    python sample_fineweb.py --language german --docs 150000
    python sample_fineweb.py --all
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from sources import SOURCES, LANGUAGES  # noqa: E402

SAMPLES_DIR = SCRIPT_DIR / "samples"

# How many shards to spread the draw across. Swedish has the fewest (20), so 16
# keeps every language sampling from a comparable number of distinct dumps
# rather than silently taking one shard for the small languages and 100 for
# German. English (sample-10BT) has only 15 and uses all of them.
N_SHARDS = 16
DEFAULT_DOCS = 150_000


def stride(items, n):
    """An even spread of `n` items across the list, always including the first.

    Evenly spaced rather than random: the spacing is the whole point (cover the
    corpus), and a deterministic rule needs no seed recorded to reproduce it.
    """
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def list_shards(repo, revision, prefix):
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not shards:
        raise ValueError(f"no parquet shards under {repo}/{prefix}")
    return shards


def read_shard(fs, repo, revision, shard, quota, want):
    """Yield up to `quota` docs spread evenly over the shard's row groups.

    `want` is the number of row groups to touch: enough to fill the quota, but
    strided so they come from across the shard rather than its head.
    """
    import pyarrow.parquet as pq

    path = f"datasets/{repo}@{revision}/{shard}"
    with fs.open(path, "rb") as handle:
        pf = pq.ParquetFile(handle)
        groups = stride(list(range(pf.num_row_groups)), want)
        taken = 0
        for g in groups:
            if taken >= quota:
                break
            table = pf.read_row_group(g, columns=["id", "text"])
            for doc_id, text in zip(table.column("id").to_pylist(),
                                    table.column("text").to_pylist()):
                if taken >= quota:
                    break
                if not text or not text.strip():
                    continue
                yield doc_id, text
                taken += 1


def sample_language(language, n_docs, out_dir, force=False):
    from huggingface_hub import HfFileSystem

    repo, revision, prefix = SOURCES[language]
    out_path = out_dir / f"{language}.jsonl.gz"
    manifest_path = out_dir / f"{language}.manifest.json"
    if out_path.exists() and manifest_path.exists() and not force:
        print(f"[{language}] exists, skipping (use --force to redraw)", flush=True)
        return out_path

    all_shards = list_shards(repo, revision, prefix)
    chosen = stride(all_shards, N_SHARDS)
    per_shard = -(-n_docs // len(chosen))    # ceil, so the quota is always met
    print(f"[{language}] {repo} @ {revision[:8]}  {len(all_shards)} shards -> "
          f"{len(chosen)} sampled, {per_shard} docs each", flush=True)

    fs = HfFileSystem()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = chars = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for shard_no, shard in enumerate(chosen):
            if written >= n_docs:
                break
            before = written
            # Row groups hold ~1000 rows; ask for 50% more than the arithmetic
            # needs so short groups or blank documents cannot leave a shard
            # under quota.
            want_groups = max(1, int(per_shard / 1000 * 1.5))
            for doc_id, text in read_shard(fs, repo, revision, shard,
                                           min(per_shard, n_docs - written),
                                           want_groups):
                # id is kept so a table traces back to exact documents; nothing
                # else from the row is needed to count tokens.
                f.write(json.dumps({"id": doc_id, "text": text},
                                   ensure_ascii=False) + "\n")
                written += 1
                chars += len(text)
            print(f"[{language}] shard {shard_no + 1}/{len(chosen)} "
                  f"({Path(shard).name}): +{written - before}  total {written}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    manifest = {
        "language": language,
        "repo": repo,
        "revision": revision,
        "prefix": prefix,
        "n_shards_total": len(all_shards),
        "shards_sampled": chosen,
        "docs_per_shard": per_shard,
        "n_docs": written,
        "n_chars": chars,
        "doc_weighting": "uniform (minhash_cluster_size ignored)",
        "selection": f"deterministic stride: {len(chosen)} shards x strided row groups",
        "seconds": round(time.time() - t0, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[{language}] wrote {written} docs, {chars / 1e6:.0f}M chars -> "
          f"{out_path} ({out_path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", action="append", dest="languages",
                        choices=LANGUAGES, help="repeat for several; default --all")
    parser.add_argument("--all", action="store_true", help="every MultiGED language")
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCS,
                        help=f"documents per language (default {DEFAULT_DOCS})")
    parser.add_argument("--out", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--force", action="store_true", help="redraw an existing sample")
    args = parser.parse_args()

    languages = list(LANGUAGES) if args.all or not args.languages else args.languages
    for language in languages:
        sample_language(language, args.docs, args.out, force=args.force)

    # fsspec's HTTP session threads can trip a GIL assertion during interpreter
    # finalisation, which prints "Fatal Python error" long after every byte is
    # on disk and would read as a failed job. Everything is written and closed
    # by here, so skip finalisation rather than let it fabricate an error.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
