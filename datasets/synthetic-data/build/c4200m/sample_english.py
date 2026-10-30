"""Normalise the English C4_200M into the same shape as the other languages.

The raw download is Parquet with `input` (corrupted) / `output` (clean) at
183,894,319 rows. The reconstructed languages are TSV `source<TAB>target` at
1.5-1.7M pairs each, so English needs both a column rename and a subsample --
at native scale it would be ~110x larger than any other arm.

Sampling is a single streaming pass with a fixed seed, so the selection is
reproducible and spread across the whole corpus rather than taken from one
contiguous block.

Rows whose text contains a tab or newline are dropped: they cannot be
represented in the TSV layout the other languages use. In the reconstructed
languages this could not arise (targets are C4 lines, split on newline, and
written with a single tab separator); English came via CSV so it must be
checked rather than assumed.
"""
import argparse
import glob
import os

import numpy as np
import pyarrow.parquet as pq

DEFAULT_N = 1_500_000
BATCH = 65536


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.normpath(
        os.path.join(here, "..", "..", "sources", "c4_200m_en", "raw")))
    ap.add_argument("--out", default=os.path.normpath(os.path.join(
        here, "..", "..", "sources", "c4200m_multilingual", "pairs",
        "en.sentence_pairs.tsv")))
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="pairs to sample (default matches the other languages)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.raw, "*.parquet")))
    if not shards:
        raise SystemExit("no parquet shards in %s" % args.raw)

    total = sum(pq.ParquetFile(p).metadata.num_rows for p in shards)
    # Oversample slightly so dropped rows and sampling variance cannot leave us
    # short; the result is shuffled and truncated to exactly --n below.
    p_keep = min(1.0, (args.n / total) * 1.05)
    print("%d shards, %d rows total; sampling p=%.5f for %d pairs"
          % (len(shards), total, p_keep, args.n))

    rng = np.random.default_rng(args.seed)
    kept, n_seen, n_unsafe = [], 0, 0
    for path in shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH,
                                                       columns=["input", "output"]):
            n = batch.num_rows
            n_seen += n
            take = np.nonzero(rng.random(n) < p_keep)[0]
            if take.size == 0:
                continue
            src = batch.column("input").take(take).to_pylist()
            tgt = batch.column("output").take(take).to_pylist()
            for s, t in zip(src, tgt):
                if s is None or t is None:
                    n_unsafe += 1
                elif "\t" in s or "\n" in s or "\t" in t or "\n" in t:
                    n_unsafe += 1
                else:
                    kept.append((s, t))
        print("  %-22s seen %d  kept %d" % (os.path.basename(path), n_seen, len(kept)),
              flush=True)

    if len(kept) < args.n:
        raise SystemExit("only %d usable pairs sampled, wanted %d" % (len(kept), args.n))
    order = rng.permutation(len(kept))[:args.n]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i in order:
            s, t = kept[i]
            f.write("%s\t%s\n" % (s, t))
    os.replace(tmp, args.out)

    print("\nwrote %d pairs -> %s" % (args.n, args.out))
    print("dropped %d rows containing tab/newline or nulls" % n_unsafe)
    print("seed=%d, source=martinsr/c4_200m @ c46acd6e" % args.seed)


if __name__ == "__main__":
    main()
