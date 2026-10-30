"""Phase 2: merge shard outputs, then apply the corruption edits.

Merging is not just concatenation. c4200m_make_sentence_pairs.py walks the
edits and targets in lockstep with a one-way `while edit_md5 < md5` advance, so
it requires hash-sorted targets; feeding it concatenated shard output would
silently drop most pairs instead of erroring. We sort with LC_ALL=C so the byte
ordering matches Python's string comparison.

The actual edit splicing is delegated to the upstream apply_edits() so the
byte-offset semantics stay exactly as published. We only wrap it, because those
offsets index into UTF-8 bytes and a boundary landing mid-codepoint would raise
on the non-English data.
"""
import argparse
import glob
import os
import subprocess
import sys

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "sources", "C4_200M"))
sys.path.insert(0, REPO)

import c4200m_make_sentence_pairs as upstream  # noqa: E402


def load_target_hashes(edits_path):
    hashes = set()
    with open(edits_path, encoding="utf-8") as f:
        for line in f:
            hashes.add(line.split("\t", 1)[0])
    return hashes


def merge_targets(lang, shard_dir, edits_path, out_path):
    """Concatenate shard outputs, drop prefix-collisions, sort by hash."""
    exact = load_target_hashes(edits_path)
    shards = sorted(glob.glob(os.path.join(shard_dir, lang, "*.tsv")))
    tmp = out_path + ".unsorted"
    n_in = n_kept = 0
    with open(tmp, "w", encoding="utf-8") as fout:
        for path in shards:
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    n_in += 1
                    if line.split("\t", 1)[0] in exact:
                        fout.write(line)
                        n_kept += 1

    env = dict(os.environ, LC_ALL="C")
    with open(out_path, "w", encoding="utf-8") as fout:
        subprocess.run(["sort", "-t", "\t", "-k1,1", "-u", tmp],
                       stdout=fout, env=env, check=True)
    os.remove(tmp)

    n_uniq = sum(1 for _ in open(out_path, encoding="utf-8"))
    print("  targets : %d shards, %d lines in, %d after collision filter, "
          "%d unique" % (len(shards), n_in, n_kept, n_uniq))
    print("  recall  : %d / %d  (%.1f%%)"
          % (n_uniq, len(exact), 100.0 * n_uniq / len(exact)))
    return n_uniq, len(exact)


def make_pairs(targets_path, edits_path, out_path):
    """Upstream pairing loop, guarded against edit-file exhaustion and
    mid-codepoint byte offsets."""
    n_pairs = n_decode_fail = 0
    with open(edits_path, encoding="utf-8") as edits_f, \
            open(targets_path, encoding="utf-8") as targets_f, \
            open(out_path, "w", encoding="utf-8") as out_f:
        edits_iter = upstream.get_edits(edits_f)
        edit_md5, edits = "0", None
        for line in targets_f:
            md5, target_text = line.rstrip("\n").split("\t", 1)
            try:
                while edit_md5 < md5:
                    edit_md5, edits = next(edits_iter)
            except StopIteration:
                break
            if edit_md5 == md5:
                try:
                    source_text = upstream.apply_edits(edits, target_text)
                except UnicodeDecodeError:
                    n_decode_fail += 1
                    continue
                out_f.write("%s\t%s\n" % (source_text, target_text))
                n_pairs += 1
    print("  pairs   : %d written, %d skipped on byte-offset decode failure"
          % (n_pairs, n_decode_fail))
    return n_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True, help="dir holding edits/ from prepare_hashes.py")
    ap.add_argument("--shards", required=True, help="per-shard tsv dir from scan.py")
    ap.add_argument("--out", required=True, help="output dir for final datasets")
    ap.add_argument("--langs", default="de,es,ro,ru")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "targets"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "pairs"), exist_ok=True)

    summary = []
    for lang in args.langs.split(","):
        print("[%s]" % lang, flush=True)
        edits = os.path.join(args.work, "edits", "%s.tsv" % lang)
        targets = os.path.join(args.out, "targets", "%s.target_sentences.tsv" % lang)
        pairs = os.path.join(args.out, "pairs", "%s.sentence_pairs.tsv" % lang)
        n_uniq, n_total = merge_targets(lang, args.shards, edits, targets)
        n_pairs = make_pairs(targets, edits, pairs)
        summary.append((lang, n_uniq, n_total, n_pairs))

    print("\n%-6s %12s %12s %9s %12s" % ("lang", "targets", "published", "recall", "pairs"))
    for lang, n_uniq, n_total, n_pairs in summary:
        print("%-6s %12d %12d %8.1f%% %12d"
              % (lang, n_uniq, n_total, 100.0 * n_uniq / n_total, n_pairs))


if __name__ == "__main__":
    main()
