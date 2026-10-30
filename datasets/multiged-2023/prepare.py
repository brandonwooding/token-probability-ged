"""Install MultiGED-2023 into this folder exactly as the project used it.

The corpora are licensed and not redistributed here. Fetch the two public
releases, then run this once (stdlib only):

    git clone https://github.com/spraakbanken/multiged-2023 /tmp/multiged-2023
    # FCE v2.1 from the BEA-2019 shared task (licence agreement required):
    #   https://www.cl.cam.ac.uk/research/nl/bea2019st/#data -> fce_v2.1.bea19.tar.gz
    python3 datasets/multiged-2023/prepare.py \
        --multiged /tmp/multiged-2023 --bea-fce /path/to/fce/m2

What it does on top of copying the upstream TSVs:

1. Repairs nine blanked tokens. Upstream blanks every token that begins with
   uppercase ``NA`` (six in FCE train, three in Falko-MERLIN train), which
   leaves a stray blank line that splits the sentence in two. Blank lines carry
   no position information, so the affected line numbers and their labels are
   pinned below; each token's text is recovered from the corpus M2 by matching
   the surrounding tokens, so no corpus text is stored in this repository.
2. Installs the M2 files used only for error *types* (labels always come from
   the TSV): BEA-2019 FCE M2s, and upstream's ``fm-*.m2`` under the
   ``de_falko-merlin_*`` stems that ``paths.py`` resolves.

Every written file is checked against the SHA-256 of the copy behind the
reported results, so a silent upstream change fails loudly.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (language folder, stem) -> splits copied verbatim from the upstream clone.
TSVS = {
    "czech": ("cs_geccc", ("train", "dev", "test_unlabelled")),
    "english": ("en_fce", ("train", "dev", "test_unlabelled")),
    "german": ("de_falko-merlin", ("train", "dev", "test_unlabelled")),
    "italian": ("it_merlin", ("train", "dev", "test_unlabelled")),
    "swedish": ("sv_swell", ("train", "dev", "test_unlabelled")),
}
EXTRA_TSVS = (("english", "en_realec_dev"), ("english", "en_realec_test_unlabelled"))

# 1-based line numbers of the stray blank lines in the upstream train TSVs,
# with the gold label MultiGED assigns the token that should be there.
REPAIRS = {
    "english/en_fce_train.tsv": {
        1057: "c", 4165: "c", 437773: "c", 437817: "c", 437870: "c", 437946: "c"},
    "german/de_falko-merlin_train.tsv": {241719: "i", 286651: "i", 301779: "i"},
}
CONTEXT = 3  # tokens either side used to locate a repaired token in the M2

EXPECTED = {
    "czech/cs_geccc_dev.tsv": "b3f6aa141677d7bb698652ad14b74a7a8316cdd548bd0cb729e5c35cad1c26a3",
    "czech/cs_geccc_test_unlabelled.tsv": "ee88607340c6acf84b036d16a6b60a5855652ce523f8bdb02dde2bf51a03d5f5",
    "czech/cs_geccc_train.tsv": "0e2cde8a16bd23a118815914c8ee0249bb0dc90e329670116ed45ff05191cab9",
    "english/en_fce_dev.tsv": "1ca5de77214a3fb9b1ebdc8dc16715f31971a8195dd095bbb524d2c08d233e4f",
    "english/en_fce_test_unlabelled.tsv": "c5983cb13c2aa82ffa5dd95d69ccdb603b98549121ba17be9f386fe7e70443a9",
    "english/en_fce_train.tsv": "f5fe27caedb9531c982eba679c917388bca4d35efc92acd6300dee2e6032ba70",
    "english/en_realec_dev.tsv": "bfbba6a99efcf3135704ad5f7e397456168863c211941027b1bc8fed1507c586",
    "english/en_realec_test_unlabelled.tsv": "38d8ae901fd70884b3484a907c127a9696f014eca00f466d24b4f5c0fcd8e336",
    "german/de_falko-merlin_dev.tsv": "4f759001cf2f3e6cf75b6038722de3c8e9a0eab43536a55c072596d04db90573",
    "german/de_falko-merlin_test_unlabelled.tsv": "44fe6508144620adc2e7d63108a8a5d423a2f6c28b35faec14988b25e2e3a43e",
    "german/de_falko-merlin_train.tsv": "ba480ea30ea5c907338a3ab36d0f1f732a7626f99c5edf6af90b62670c9b2534",
    "italian/it_merlin_dev.tsv": "5b50b0e443c6b37413a0b1f1ffe92c874017b02118a843f4fa063b42bf93948a",
    "italian/it_merlin_test_unlabelled.tsv": "ac952c034b957336559e13060cf4078c9ec5a2b2d5713eb1e21ffecd34e17f6e",
    "italian/it_merlin_train.tsv": "fd2f0322157723937731f5733e16d088d347c55a1cb533bbadeeea8cafbdc0d0",
    "swedish/sv_swell_dev.tsv": "1888881e1e33d5ddd6d659a72f94f727601854a40cc55e64536f5a61d16a67e5",
    "swedish/sv_swell_test_unlabelled.tsv": "20a46f2f2fe422233f23f9a047f5a493d8e81cde4bbc8c6d7ad61f9f2d64ede8",
    "swedish/sv_swell_train.tsv": "18a56e653c726c4aa4d1ae369d0e1c1019af99f891bb93f032e76fd8be427e0c",
    "english/en_fce_dev.m2": "3c651088c029005b0072a99f747ff38db7bce0de7f975a1c3645c48e9c64064e",
    "english/en_fce_test.m2": "085ac316aaa2751738fd59208d4404cadd190bc369ac5915fe988c4b6641dd9b",
    "english/en_fce_train.m2": "fa2d2c563976b02521c1f58d2f3e78a1ac905fae312d7c0a32bbc3e69b604d5d",
    "german/de_falko-merlin_dev.m2": "24541dd8b6172083456f7e8cf4543a6225d8a285ebc9245030d73bbb4bd921d4",
    "german/de_falko-merlin_train.m2": "c555ddf98af689e7613b596c18f2461f1b03fab286599866ba5a0907ea555ed7",
}

# Destination M2 -> (source flag, file name inside that source).
M2S = {
    "english/en_fce_train.m2": ("bea", "fce.train.gold.bea19.m2"),
    "english/en_fce_dev.m2": ("bea", "fce.dev.gold.bea19.m2"),
    "english/en_fce_test.m2": ("bea", "fce.test.gold.bea19.m2"),
    "german/de_falko-merlin_train.m2": ("multiged", "german/fm-train.m2"),
    "german/de_falko-merlin_dev.m2": ("multiged", "german/fm-dev.m2"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def m2_tokens(path):
    """Flat token stream of every ``S`` line, in file order."""
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("S "):
                tokens.extend(line[2:].split())
    return tokens


def tsv_token(line):
    # MultiGED escapes double quotes; the M2 does not.
    return line.split("\t", 1)[0].replace('\\"', '"')


def repair(lines, fixes, m2):
    """Fill each pinned blank line with its token recovered from ``m2``."""
    for lineno, label in sorted(fixes.items()):
        i = lineno - 1
        if lines[i].strip():
            raise SystemExit(f"line {lineno} is not blank -- upstream file changed?")
        # Context stops at a real sentence break on either side.
        before = []
        for l in reversed(lines[max(0, i - CONTEXT):i]):
            if not l.strip():
                break
            before.insert(0, tsv_token(l))
        after = []
        for l in lines[i + 1:i + 1 + CONTEXT]:
            if not l.strip():
                break
            after.append(tsv_token(l))
        hits = [j for j in range(len(before), len(m2) - len(after))
                if m2[j - len(before):j] == before and m2[j + 1:j + 1 + len(after)] == after]
        if len(hits) != 1 or not m2[hits[0]].startswith("NA"):
            raise SystemExit(f"line {lineno}: could not locate token in M2 ({len(hits)} hits)")
        lines[i] = f"{m2[hits[0]]}\t{label}\n"
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multiged", type=Path, required=True,
                    help="clone of github.com/spraakbanken/multiged-2023")
    ap.add_argument("--bea-fce", type=Path,
                    help="directory holding fce.{train,dev,test}.gold.bea19.m2 "
                         "(omit to skip the FCE M2s; FCE scores as detection-only)")
    args = ap.parse_args()
    sources = {"multiged": args.multiged, "bea": args.bea_fce}

    written = []
    for folder, (stem, splits) in TSVS.items():
        for split in splits:
            written.append(f"{folder}/{stem}_{split}.tsv")
    written += [f"{folder}/{stem}.tsv" for folder, stem in EXTRA_TSVS]

    for rel in written:
        src = args.multiged / rel
        dst = HERE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel in REPAIRS:
            m2_rel = {"english/en_fce_train.tsv": "english/en_fce_train.m2",
                      "german/de_falko-merlin_train.tsv": "german/de_falko-merlin_train.m2"}[rel]
            flag, name = M2S[m2_rel]
            if sources[flag] is None:
                raise SystemExit(f"repairing {rel} needs --bea-fce (its M2 holds the tokens)")
            lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
            dst.write_text("".join(repair(lines, REPAIRS[rel], m2_tokens(sources[flag] / name))),
                           encoding="utf-8")
        else:
            shutil.copyfile(src, dst)

    for rel, (flag, name) in M2S.items():
        if sources[flag] is None:
            print(f"skip {rel} (no --bea-fce)")
            continue
        shutil.copyfile(sources[flag] / name, HERE / rel)
        written.append(rel)

    bad = [rel for rel in written if sha256(HERE / rel) != EXPECTED[rel]]
    for rel in written:
        print(("MISMATCH " if rel in bad else "ok       ") + rel)
    if bad:
        sys.exit(f"{len(bad)} file(s) differ from the copies behind the reported results")
    print(f"\n{len(written)} files installed and verified.")


if __name__ == "__main__":
    main()
