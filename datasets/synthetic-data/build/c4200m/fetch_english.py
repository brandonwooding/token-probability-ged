"""Fetch the pre-built English C4_200M (martinsr/c4_200m) as raw Parquet.

The other four languages are reconstructed from mC4 by scan.py; English is not,
because the English corruption edits live behind Kaggle auth rather than in the
upstream repo. martinsr/c4_200m is a Parquet conversion of liweili/c4_200m and
reports exactly 183,894,319 pairs -- the count upstream's own script documents
as recoverable from AllenAI C4 v3.0.1 (see c4200m_get_target_sentences_json.py).

The revision is pinned so this stays reproducible if the repo changes. Treat
the download as immutable raw input; derive samples from it rather than
re-downloading.

Note: the `hf` CLI is unusable in this environment (huggingface_hub 1.20.1
needs click>=8.2, and ~/.local has an older click shared with datasets and
transformers), so this uses the Python API directly.
"""
import argparse
import os

from huggingface_hub import snapshot_download

REPO = "martinsr/c4_200m"
REVISION = "c46acd6e099d2bf712cd8d0496cb2ba135e786c8"
EXPECTED_PAIRS = 183894319
EXPECTED_SHARDS = 62


def main():
    default_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "sources", "c4_200m_en", "raw"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=default_dir)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
        local_dir=args.out,
        max_workers=args.workers,
    )

    shards = sorted(f for f in os.listdir(path) if f.endswith(".parquet"))
    total = sum(os.path.getsize(os.path.join(path, f)) for f in shards)
    print("\n%s @ %s" % (REPO, REVISION[:12]))
    print("  %d parquet shards, %.1f GB -> %s" % (len(shards), total / 1e9, path))
    if len(shards) != EXPECTED_SHARDS:
        print("  WARNING: expected %d shards" % EXPECTED_SHARDS)
    print("  expected %d pairs; verify with sample_english.py" % EXPECTED_PAIRS)


if __name__ == "__main__":
    main()
