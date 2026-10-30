"""One-off prep: turn the bz2 edit files into artefacts the scan job can use.

For each language this writes:

  edits/<lang>.tsv      decompressed edits, needed by c4200m_make_sentence_pairs.py
  hashes/<lang>.npy     sorted uint64 array of the first 8 bytes of each target
                        MD5, used as the in-memory lookup set by the scanners

The uint64 prefix (rather than the full 16-byte digest) keeps the per-worker
set around 200 MB instead of ~350 MB. Across the whole corpus (~29e9 lines
scanned, ~2.7e6 targets per language) the expected number of 64-bit prefix
collisions is ~4e-3, i.e. less than one spurious line in the entire run, and
merge.py filters the output against the exact hex hashes anyway.
"""
import argparse
import bz2
import os

import numpy as np

LANGS = ["de", "es", "ro", "ru"]
REPO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "sources", "C4_200M", "multilingual")


def prepare(lang, out_dir):
    src = os.path.normpath(os.path.join(REPO, "%s.tsv.bz2" % lang))
    edits_dir = os.path.join(out_dir, "edits")
    hash_dir = os.path.join(out_dir, "hashes")
    os.makedirs(edits_dir, exist_ok=True)
    os.makedirs(hash_dir, exist_ok=True)

    edits_path = os.path.join(edits_dir, "%s.tsv" % lang)
    npy_path = os.path.join(hash_dir, "%s.npy" % lang)

    prefixes = set()
    n_edits = 0
    prev = ""
    sorted_ok = True
    tmp = edits_path + ".tmp"
    with bz2.open(src, "rt", encoding="utf-8") as fin, \
            open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
            n_edits += 1
            h = line.split("\t", 1)[0]
            if h < prev:
                sorted_ok = False
            prev = h
            prefixes.add(int(h[:16], 16))
    os.replace(tmp, edits_path)

    # c4200m_make_sentence_pairs.py walks edits and targets in lockstep with a
    # one-way `while edit_md5 < md5` advance, so a non-sorted edits file would
    # silently drop pairs rather than error.
    if not sorted_ok:
        raise SystemExit("%s edits are not hash-sorted; phase 2 would lose data"
                         % lang)

    arr = np.fromiter(prefixes, dtype=np.uint64, count=len(prefixes))
    arr.sort()
    np.save(npy_path, arr)
    print("%s: %d edits, %d unique targets -> %s (%.1f MB)"
          % (lang, n_edits, len(arr), npy_path, arr.nbytes / 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="work directory")
    ap.add_argument("--langs", default=",".join(LANGS))
    args = ap.parse_args()
    for lang in args.langs.split(","):
        prepare(lang, args.out)


if __name__ == "__main__":
    main()
