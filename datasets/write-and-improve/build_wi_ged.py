#!/usr/bin/env python3
"""Write & Improve 2024 -> sentence-level MultiGED-format GED datasets.

Produces, per split, alongside this script in `english/`:

    en_wi2024_<split>.tsv        token <TAB> c|i, blank line between sentences
    en_wi2024_<split>.m2         ERRANT edits (error types for standard-eval)
    en_wi2024_<split>.cefr.json  per-unit CEFR / L1 / essay id, keyed by unit_id

Why the corpus M2 is not reused
-------------------------------
The corpus ships its own `.m2`, but its edits were aligned against *whitespace*
tokens (`life.` and `However,` are single tokens). Re-projecting those spans onto
spaCy tokens preserves the coarse localisation: a span that says "this whitespace
token is wrong" cannot say whether the word or its punctuation was wrong, so both
halves inherit the `i`. Measured on dev, that route yields 10,558 `i` against
8,349 for a recomputed alignment at the same token count -- a 26% inflation that
is pure alignment granularity. The information needed to narrow it was never
recorded, so no projection can recover it. Hence ERRANT is re-run over the
corpus's own aligned `.orig`/`.corr` sentence pairs.

This is the opposite of the synthetic data, which records corruption sites
as it builds them and so needs no aligner of its own. That guarantee does not
exist for real data with human corrections; here an aligner is unavoidable, so
we use the one MultiGED itself was built with.

Tokenisation
------------
ERRANT's `en_core_web_sm` pipeline uses the same rule-based English tokenizer as
`spacy.blank("en")`, which the project standardised on for matching the
MultiGED gold (`Do` + `n't`). `--selftest`
asserts the two agree token-for-token.

Labelling policies
------------------
MultiGED's English gold is *sparser* than a naive "every edit marks its tokens"
conversion. Empirically (see `--selftest`), English FCE marks a token `i` iff the
edit's correction is non-empty AND not punctuation-only, and the category is not
UNK. So deletions (`U:*`), inserted punctuation (`M:PUNCT`) and punct->punct
`R:PUNCT` are all ignored. Applying that rule to the BEA-2019 FCE M2 reproduces
the official `en_fce_dev.tsv` to 5 tokens in 34,748 (0.014%), none missing.

The project W&I convention is now `--policy exclude-punct` (the default): ignore
every ERRANT `*:PUNCT` category, plus the official converter's technical
exclusions (`noop`, `UNK`, and non-zero annotators). All other deletions are
retained, including `U:DET`; `U:PUNCT` is excluded. `--policy fce` retains the
reconstructed FCE convention for analysis; `--policy all` marks every
non-noop/non-UNK annotator-0 edit.

Insertions attach to the token to the RIGHT of the gap, in all policies -- the
MultiGED convention, independently verified in `ged_common.project_spans`.

Stages (each skipped if its output exists; `--force` to rebuild):

    python build_wi_ged.py --selftest          # prove the policy against FCE
    python build_wi_ged.py --split dev         # m2 + tsv + cefr
    python build_wi_ged.py --split train

Portable test packaging from any line-aligned original/corrected files:

    python build_wi_ged.py --originals test.orig --corrections test.corr \
        --out-prefix en_wi2024_test

Here "original" means the learner/source sentence and "correction" means its
gold corrected target. This mode emits both `<prefix>.m2` and `<prefix>.tsv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import string
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
# The licensed W&I 2024 v2 release, unpacked by hand (see README.md).
CORPUS = (HERE / "source" / "write-and-improve-corpus-2024-v2" / "user-prompt-final-versions")
CORPUS_STEM = "en-writeandimprove2024-final-versions"
FCE_TSV = REPO_ROOT / "datasets" / "multiged-2023" / "english" / "en_fce_dev.tsv"
FCE_M2 = REPO_ROOT / "datasets" / "multiged-2023" / "english" / "en_fce_dev.m2"
OUT_DIR = HERE / "english"

PUNCT = set(string.punctuation) | set("’‘“”–—…«»")


# ---------------------------------------------------------------------------
# M2 -> MultiGED labels
# ---------------------------------------------------------------------------

def punct_only(text):
    """True if `text` is non-empty and contains nothing but punctuation/space."""
    return bool(text) and all(ch in PUNCT or ch.isspace() for ch in text)


def parse_m2(path):
    """M2 file -> [{tokens, edits:[(start, end, type, correction, annotator)]}]."""
    records, current = [], None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("S "):
                if current is not None:
                    records.append(current)
                current = {"tokens": line[2:].split(" "), "edits": []}
            elif line.startswith("A "):
                fields = line[2:].split("|||")
                span = fields[0].split()
                current["edits"].append(
                    (int(span[0]), int(span[1]), fields[1], fields[2], fields[-1])
                )
    if current is not None:
        records.append(current)
    return records


def is_punctuation_type(error_type):
    """Whether an ERRANT category is in the PUNCT family."""
    return error_type == "PUNCT" or error_type.endswith(":PUNCT")


def multiged_labels(record, policy="exclude-punct"):
    """Per-token c/i for one M2 record under the MultiGED convention.

    Insertions (start == end) mark the token to the right of the gap; span edits
    mark [start, end). Out-of-range indices (an insertion past the last token)
    mark nothing, which is what the official conversion script does.
    """
    n = len(record["tokens"])
    labels = ["c"] * n
    for start, end, etype, correction, annotator in record["edits"]:
        # The released MultiGED converter explicitly excludes noop and UNK and
        # processes annotator 0 only. These are format/annotation exclusions,
        # shared by every policy rather than linguistic error selection.
        if etype in ("noop", "UNK") or annotator != "0":
            continue
        if policy == "fce" and (not correction.strip() or punct_only(correction)):
            continue
        if policy == "exclude-punct" and is_punctuation_type(etype):
            continue
        indices = [start] if start == end else range(start, end)
        for i in indices:
            if 0 <= i < n:
                labels[i] = "i"
    return labels


DQUOTE = '"'
ESCAPED_DQUOTE = "\\" + DQUOTE


def escape_token(token):
    """Escape one token for the MultiGED TSV.

    The format spec only mentions a bare double quote, written `\\"`. But the
    reader (`reconstruct_from_tsv.read_sentences`) parses with
    `csv.reader(escapechar="\\")`, so a *literal* backslash in the data is
    consumed as an escape character and silently lost -- `I\\u00B4ve` reads back
    as `Iu00B4ve`. W&I has 279 such tokens (mojibake where unicode escapes leaked
    into the source text as literal characters), so backslashes are doubled to
    survive the round trip.

    Verified: this is the only scheme under which `I\\u00B4ve`, `"`, `a"b`, `\\`
    and `C:\\path\\x` all read back identically.
    """
    if token == DQUOTE:
        return ESCAPED_DQUOTE
    return token.replace("\\", "\\\\")


def write_tsv(path, records, policy):
    """MultiGED TSV: token <TAB> label, blank line between sentences.

    Double quotes are escaped as `\\"` per the MultiGED format spec, which is
    what `reconstruct_from_tsv.read_sentences` expects to decode.
    """
    n_tokens = n_err = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as out:
            for record in records:
                labels = multiged_labels(record, policy)
                for token, label in zip(record["tokens"], labels):
                    out.write(escape_token(token) + "\t" + label + "\n")
                    n_tokens += 1
                    n_err += label == "i"
                out.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return n_tokens, n_err


# ---------------------------------------------------------------------------
# selftest: the policy must reproduce the official MultiGED FCE gold
# ---------------------------------------------------------------------------

def read_tsv(path):
    """MultiGED TSV -> flat [(token, label)], unescaping `\\"`."""
    flat = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t", quotechar='"', escapechar="\\"):
            if not row or not row[0].strip():
                continue
            flat.append(('"' if row[0] == r"\"" else row[0], row[1]))
    return flat


def selftest():
    """Prove the conversion against gold we did not build, and fail loudly."""
    import spacy

    print("1. exclude-punct policy keeps non-punctuation deletions and excludes UNK")
    policy_probe = {
        "tokens": ["the", ",", ".", "odd", "mispelt"],
        "edits": [
            (0, 1, "U:DET", "", "0"),
            (1, 2, "U:PUNCT", "", "0"),
            (2, 3, "R:PUNCT", "!", "0"),
            (3, 3, "M:PUNCT", ";", "0"),
            (3, 4, "UNK", "strange", "0"),
            (4, 5, "R:SPELL", "misspelt", "0"),
        ],
    }
    assert multiged_labels(policy_probe, "exclude-punct") == ["i", "c", "c", "c", "i"]
    print("   OK (U:DET retained; UNK and M/R/U:PUNCT excluded)")

    print("2. tokenizer parity: errant's pipeline vs spacy.blank('en')")
    import errant
    annotator = errant.load("en")
    blank = spacy.blank("en").tokenizer
    probes = [
        "There are lots of things in our life. However, they can't do what I want.",
        "Terms & Conditions apply and/or 24/7 well-known AT&T support.",
        "I haven't seen it -- it's a 5/10 at best...",
    ]
    for text in probes:
        a = [t.text for t in annotator.nlp.tokenizer(text)]
        b = [t.text for t in blank(text)]
        assert a == b, f"tokenizer drift on {text!r}:\n  errant={a}\n  blank={b}"
    print(f"   OK ({len(probes)} probes agree token-for-token)")

    print("3. labelling policy: BEA-2019 FCE M2 -> official MultiGED en_fce_dev.tsv")
    gold = read_tsv(FCE_TSV)
    records = parse_m2(FCE_M2)
    m2_tokens = [t for r in records for t in r["tokens"]]
    gold_tokens = [t for t, _ in gold]
    assert m2_tokens == gold_tokens, (
        f"token streams differ ({len(m2_tokens)} vs {len(gold_tokens)})"
    )
    gold_labels = [l for _, l in gold]
    for policy, budget in (("all", None), ("fce", 10)):
        derived = [l for r in records for l in multiged_labels(r, policy)]
        wrong = [i for i, (x, y) in enumerate(zip(derived, gold_labels)) if x != y]
        missing = sum(1 for i in wrong if derived[i] == "c")
        rate = 100 * len(wrong) / len(gold_labels)
        print(f"   policy={policy:4s}: i={derived.count('i'):5d} vs gold "
              f"{gold_labels.count('i')}, {len(wrong)} mismatches ({rate:.3f}%), "
              f"{missing} missing")
        if budget is not None:
            assert missing == 0, f"policy '{policy}' misses {missing} gold errors"
            assert len(wrong) <= budget, (
                f"policy '{policy}' has {len(wrong)} mismatches, budget {budget}"
            )
    print("   OK (fce policy reproduces official gold within budget)")
    return 0


# ---------------------------------------------------------------------------
# stage 1: ERRANT over the corpus's aligned sentence pairs
# ---------------------------------------------------------------------------

def corpus_path(split, suffix):
    return CORPUS / f"{CORPUS_STEM}-{split}-sentences.{suffix}"


def read_lines(path):
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def normalise(text):
    """Collapse whitespace runs and strip the ends.

    spaCy emits a token whose text is whitespace for a leading space, and for any
    internal run of two or more (`ged_common.token_offsets` drops those, noting
    MultiGED gold has none). Dropping them after the fact is not an option here:
    the M2's edit offsets index this token sequence, so removing a token silently
    shifts every edit after it. Instead the text is normalised *before* parsing,
    so the sequence never contains one and the M2, the TSV and the reconstructed
    sentences all stay token-for-token identical.

    Measured on the corpus: 2 such tokens in dev, 12 in train, all from source
    lines with a leading or trailing space. Applied to originals and corrections
    alike, so the alignment sees a consistent pair.
    """
    return " ".join(text.split())


def build_m2_from_files(originals_path, corrections_path, out_path,
                        batch_size=2000, name="dataset"):
    """ERRANT-annotate line-aligned original/corrected files into an M2.

    Parsing goes through `nlp.pipe` rather than `annotator.parse` so spaCy
    batches; ERRANT's own parse() is one document per call and roughly an order
    of magnitude slower over 42k pairs.
    """
    import errant

    originals = [normalise(line) for line in read_lines(originals_path)]
    corrections = [normalise(line) for line in read_lines(corrections_path)]
    if len(originals) != len(corrections):
        raise SystemExit(
            f"{name}: original/correction line counts differ "
            f"({len(originals)} vs {len(corrections)})"
        )
    empty = [index for index, pair in enumerate(zip(originals, corrections), 1)
             if not pair[0] or not pair[1]]
    if empty:
        raise SystemExit(
            f"{name}: blank original/correction sentence at one-based lines "
            f"{empty[:10]}" + (" ..." if len(empty) > 10 else ""))

    annotator = errant.load("en")
    tmp = out_path.with_suffix(".m2.tmp")
    n_edits = 0
    started = time.time()
    with open(tmp, "w", encoding="utf-8") as out:
        for begin in range(0, len(originals), batch_size):
            chunk_o = originals[begin:begin + batch_size]
            chunk_c = corrections[begin:begin + batch_size]
            docs_o = list(annotator.nlp.pipe(chunk_o))
            docs_c = list(annotator.nlp.pipe(chunk_c))
            for doc_o, doc_c in zip(docs_o, docs_c):
                edits = annotator.annotate(doc_o, doc_c)
                tokens = [t.text for t in doc_o]
                # A whitespace or empty token would round-trip through the TSV as
                # a sentence break (a blank first field is the separator), losing
                # the token and splitting the sentence. normalise() should make
                # this impossible; assert rather than trust it.
                if any(not t.strip() for t in tokens):
                    raise SystemExit(
                        f"{name}: whitespace token in {tokens!r} -- normalise() "
                        f"failed to remove it"
                    )
                out.write("S " + " ".join(tokens) + "\n")
                if edits:
                    for edit in edits:
                        out.write(edit.to_m2() + "\n")
                        n_edits += 1
                else:
                    out.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
                out.write("\n")
            done = min(begin + batch_size, len(originals))
            rate = done / max(time.time() - started, 1e-9)
            print(f"   {done}/{len(originals)} pairs  ({rate:.0f}/s)", flush=True)
    os.replace(tmp, out_path)
    return len(originals), n_edits


def build_m2(split, out_path, batch_size=2000):
    """ERRANT-annotate one bundled W&I split."""
    return build_m2_from_files(
        corpus_path(split, "orig"), corpus_path(split, "corr"), out_path,
        batch_size=batch_size, name=split)


# ---------------------------------------------------------------------------
# stage 2: CEFR sidecar
# ---------------------------------------------------------------------------

COARSE = {"pre-A1": "A", "A1": "A", "A1+": "A", "A2": "A", "A2+": "A",
          "B1": "B", "B1+": "B", "B2": "B", "B2+": "B",
          "C1": "C", "C1+": "C", "C2": "C", "C2+": "C"}


def build_cefr(split, n_units, out_path):
    """Per-unit learner metadata, keyed by unit_id (= index in the TSV).

    `unit_id` is positional everywhere in standard-eval (`enumerate(sentences)`),
    so the sidecar is a list in the same order, with the id written explicitly so
    a consumer can join on it rather than trust the order.

    Essay-level fields are broadcast to every sentence of that essay via the
    corpus's `.ids` file. `human_cefr` is the annotator grade and the one to
    stratify on; `auto_cefr` is the automarker's and is a model reading the same
    text, so it partly reflects error density -- the quantity a threshold study
    is trying to hold apart from level.
    """
    ids = read_lines(corpus_path(split, "ids"))
    if len(ids) != n_units:
        raise SystemExit(
            f"{split}: .ids has {len(ids)} lines but the M2 has {n_units} units"
        )

    info_path = CORPUS / f"{CORPUS_STEM}-m2-essay-info.tsv"
    with open(info_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t", quotechar='"'))
    by_essay = {r["public_essay_id"]: r for r in rows if r["split"] == split}

    missing = {i for i in ids if i not in by_essay}
    if missing:
        raise SystemExit(
            f"{split}: {len(missing)} essay ids in .ids have no metadata row, "
            f"e.g. {sorted(missing)[:3]}"
        )

    units = []
    for unit_id, essay_id in enumerate(ids):
        row = by_essay[essay_id]
        human = row["human_cefr"]
        auto = row["auto_cefr"]
        units.append({
            "unit_id": unit_id,
            "essay_id": essay_id,
            "human_cefr": None if human == "NA" else human,
            "human_cefr_coarse": COARSE.get(human),
            "auto_cefr": None if auto == "NA" else auto,
            "auto_cefr_coarse": COARSE.get(auto),
            "l1": None if row["L1"] == "NA" else row["L1"],
            "prompt_id": row["public_prompt_id"],
        })

    payload = {
        "dataset": f"english/en_wi2024_{split}",
        "n_units": len(units),
        "key": "unit_id (positional index of the sentence in the .tsv)",
        "source": str(info_path.relative_to(REPO_ROOT.parent)),
        "units": units,
    }
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, ensure_ascii=False)
        out.write("\n")
    return units


def summarise(units, records, policy):
    """Token/error counts per coarse CEFR band, so skew is visible at build time."""
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for unit, record in zip(units, records):
        band = unit["human_cefr_coarse"] or "?"
        labels = multiged_labels(record, policy)
        agg[band][0] += 1
        agg[band][1] += len(labels)
        agg[band][2] += labels.count("i")
    lines = [f"   {'band':5s} {'units':>7s} {'tokens':>9s} {'i':>8s} {'i-rate':>8s}"]
    for band in sorted(agg):
        units_n, tokens, errors = agg[band]
        rate = 100 * errors / tokens if tokens else 0.0
        lines.append(f"   {band:5s} {units_n:7d} {tokens:9d} {errors:8d} {rate:7.2f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def build_split(split, policy, force):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"en_wi2024_{split}"
    m2_path = OUT_DIR / f"{stem}.m2"
    tsv_path = OUT_DIR / f"{stem}.tsv"
    cefr_path = OUT_DIR / f"{stem}.cefr.json"

    print(f"[{split}] m2 -> {m2_path.name}")
    if m2_path.exists() and not force:
        print("   exists, skipping (use --force to rebuild)")
    else:
        n_pairs, n_edits = build_m2(split, m2_path)
        print(f"   {n_pairs} pairs, {n_edits} edits")

    records = parse_m2(m2_path)
    print(f"[{split}] tsv -> {tsv_path.name}  (policy={policy})")
    n_tokens, n_err = write_tsv(tsv_path, records, policy)
    print(f"   {len(records)} sentences, {n_tokens} tokens, {n_err} i "
          f"({100 * n_err / n_tokens:.2f}%)")

    print(f"[{split}] cefr -> {cefr_path.name}")
    units = build_cefr(split, len(records), cefr_path)
    print(summarise(units, records, policy))


def package_pairs(originals, corrections, out_prefix, policy, force, batch_size):
    """Package arbitrary paired files for blind-test release by a data holder."""
    m2_path = Path(str(out_prefix) + ".m2")
    tsv_path = Path(str(out_prefix) + ".tsv")
    existing = [path for path in (m2_path, tsv_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite {names}; pass --force")
    m2_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[paired input] m2 -> {m2_path}")
    n_pairs, n_edits = build_m2_from_files(
        originals, corrections, m2_path, batch_size=batch_size, name=str(out_prefix))
    print(f"   {n_pairs} pairs, {n_edits} edits")
    records = parse_m2(m2_path)
    print(f"[paired input] tsv -> {tsv_path}  (policy={policy})")
    n_tokens, n_err = write_tsv(tsv_path, records, policy)
    print(f"   {len(records)} sentences, {n_tokens} tokens, {n_err} i "
          f"({100 * n_err / n_tokens:.2f}%)")
    return m2_path, tsv_path


def package_unlabelled(originals_path, out_prefix, force, batch_size):
    """Tokenize public originals into the one-column TSV used for inference."""
    import errant

    tsv_path = Path(str(out_prefix) + ".tsv")
    if tsv_path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {tsv_path}; pass --force")
    originals = [normalise(line) for line in read_lines(originals_path)]
    empty = [index for index, text in enumerate(originals, 1) if not text]
    if empty:
        raise SystemExit(
            f"blank original sentence at one-based lines {empty[:10]}" +
            (" ..." if len(empty) > 10 else ""))
    annotator = errant.load("en")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tsv_path.with_suffix(tsv_path.suffix + ".tmp")
    n_tokens = 0
    try:
        with temporary.open("w", encoding="utf-8") as out:
            for begin in range(0, len(originals), batch_size):
                for doc in annotator.nlp.pipe(originals[begin:begin + batch_size]):
                    tokens = [token.text for token in doc]
                    if not tokens or any(not token.strip() for token in tokens):
                        raise SystemExit(
                            f"invalid empty/whitespace token near input line {begin + 1}")
                    for token in tokens:
                        out.write(escape_token(token) + "\n")
                        n_tokens += 1
                    out.write("\n")
        os.replace(temporary, tsv_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[unlabelled input] wrote {tsv_path}  "
          f"sentences={len(originals)} tokens={n_tokens}")
    return tsv_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "train"], action="append",
                        help="repeatable; default is both.")
    parser.add_argument("--policy", choices=["exclude-punct", "fce", "all"],
                        default="exclude-punct",
                        help="GED policy (default: exclude-punct; retains U:DET).")
    parser.add_argument("--originals", type=Path,
                        help="line-aligned learner/source sentences for portable mode.")
    parser.add_argument("--corrections", type=Path,
                        help="line-aligned gold corrected sentences for portable mode.")
    parser.add_argument("--out-prefix", type=Path,
                        help="portable-mode output prefix; writes .m2 and .tsv.")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--force", action="store_true",
                        help="rebuild the m2 even if it exists.")
    parser.add_argument("--selftest", action="store_true",
                        help="prove the policy against official FCE gold and exit.")
    args = parser.parse_args()

    if args.selftest:
        if any((args.originals, args.corrections, args.out_prefix)):
            parser.error("--selftest cannot be combined with portable paired-input mode")
        return selftest()
    portable = (args.originals, args.corrections, args.out_prefix)
    if any(portable):
        if not args.originals or not args.out_prefix:
            parser.error("portable mode requires --originals and --out-prefix")
        if args.split:
            parser.error("--split cannot be combined with portable paired-input mode")
        if args.corrections:
            package_pairs(args.originals, args.corrections, args.out_prefix,
                          args.policy, args.force, args.batch_size)
        else:
            package_unlabelled(args.originals, args.out_prefix,
                               args.force, args.batch_size)
        return 0
    for split in args.split or ["dev", "train"]:
        build_split(split, args.policy, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
