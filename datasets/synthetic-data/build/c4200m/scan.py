"""Phase 1: recover C4_200M target sentences by scanning mC4 shards.

Each shard is streamed straight from HuggingFace through curl into gzip into
the hasher -- nothing is staged on disk, so download and scan overlap inside a
single worker and the job needs no scratch space. A shard is the unit of work:
its output file is written atomically, so a job killed at walltime loses at
most one shard per worker and resubmitting the identical script resumes.

Shards are processed head-first (low index first) because the target sentences
are heavily concentrated in the first few dozen shards of each language -- see
datasets/synthetic-data/README.md.
"""
import argparse
import gzip
import hashlib
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

try:
    import orjson

    def load_text(raw):
        return orjson.loads(raw)["text"]
except ImportError:
    import json

    def load_text(raw):
        return json.loads(raw)["text"]

NSHARDS = {"de": 2048, "es": 2048, "ro": 1024, "ru": 4096}

# Shards below this index are the dense region for each language (measured).
# Everything from here on yields ~100-550 target sentences per shard.
HEAD = {"de": 64, "es": 64, "ro": 128, "ru": 64}

URL = ("https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/"
       "c4-{lang}.tfrecord-{idx:05d}-of-{n:05d}.json.gz")

MAX_RETRY = 4
_HASHES = None


def _init(npy_path):
    global _HASHES
    _HASHES = set(np.load(npy_path).tolist())


def _shard_url(lang, idx):
    return URL.format(lang=lang, idx=idx, n=NSHARDS[lang])


def scan_shard(task):
    """Stream one shard and write its recovered (md5, sentence) pairs."""
    lang, idx, out_dir = task
    out_path = os.path.join(out_dir, lang, "%05d.tsv" % idx)
    if os.path.exists(out_path):
        return (lang, idx, -1, 0.0)

    url = _shard_url(lang, idx)
    md5 = hashlib.md5
    hashes = _HASHES

    last_err = None
    for attempt in range(MAX_RETRY):
        proc = None
        t0 = time.time()
        try:
            proc = subprocess.Popen(
                ["curl", "-sfL", "--retry", "3", "--retry-delay", "2",
                 "--connect-timeout", "30", url],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            hits = []
            with gzip.GzipFile(fileobj=proc.stdout) as gz:
                for raw in gz:
                    for line in load_text(raw).split("\n"):
                        digest = md5(line.encode("utf-8")).digest()
                        if int.from_bytes(digest[:8], "big") in hashes:
                            hits.append((digest.hex(), line))
            if proc.wait() != 0:
                raise IOError("curl exited %d" % proc.returncode)

            # Atomic publish: a truncated file must never look complete.
            tmp = out_path + ".tmp%d" % os.getpid()
            with open(tmp, "w", encoding="utf-8") as f:
                for h, line in hits:
                    f.write("%s\t%s\n" % (h, line))
            os.replace(tmp, out_path)
            return (lang, idx, len(hits), time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last_err = exc
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            time.sleep(2 * (attempt + 1))

    sys.stderr.write("FAILED %s/%05d after %d attempts: %s\n"
                     % (lang, idx, MAX_RETRY, last_err))
    return (lang, idx, None, 0.0)


def segments(langs, mode):
    """Work segments, heads of every language before any tail."""
    out = []
    if mode in ("head", "all"):
        for lang in langs:
            out.append((lang, 0, HEAD[lang]))
    if mode in ("tail", "all"):
        for lang in langs:
            out.append((lang, HEAD[lang], NSHARDS[lang]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True, help="dir holding hashes/ from prepare_hashes.py")
    ap.add_argument("--out", required=True, help="output dir for per-shard tsv")
    ap.add_argument("--langs", default="de,es,ro,ru")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--mode", choices=["head", "tail", "all"], default="all",
                    help="head = dense region only (~67 GB, most of the yield)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap shards per segment (0 = no cap)")
    args = ap.parse_args()

    langs = args.langs.split(",")
    for lang in langs:
        os.makedirs(os.path.join(args.out, lang), exist_ok=True)

    for lang, lo, hi in segments(langs, args.mode):
        if args.limit:
            hi = min(hi, lo + args.limit)
        npy = os.path.join(args.work, "hashes", "%s.npy" % lang)
        tasks = [(lang, i, args.out) for i in range(lo, hi)]
        todo = [t for t in tasks
                if not os.path.exists(os.path.join(args.out, lang, "%05d.tsv" % t[1]))]
        print("[%s %d-%d] %d shards, %d already done"
              % (lang, lo, hi - 1, len(tasks), len(tasks) - len(todo)), flush=True)
        if not todo:
            continue

        t0 = time.time()
        done = hits = failed = 0
        with mp.Pool(args.workers, initializer=_init, initargs=(npy,)) as pool:
            for lg, idx, n, _dt in pool.imap_unordered(scan_shard, todo, chunksize=1):
                done += 1
                if n is None:
                    failed += 1
                elif n >= 0:
                    hits += n
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    print("  %s %d/%d shards  %d hits  %.1f shard/min  eta %.1f min"
                          % (lg, done, len(todo), hits, done / el * 60,
                             (len(todo) - done) / (done / el) / 60), flush=True)
        print("[%s %d-%d] done: %d hits, %d failed, %.1f min"
              % (lang, lo, hi - 1, hits, failed, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
