"""Build the canonical FCE-dev gold file (sentence-level).

Design decision (see README): the binary c/i labels stay exactly as the
MultiGED-2023 convention that every approach has already been scored against.
The BEA-2019 M2 file is used ONLY to attach an error type to each token
MultiGED already flags as "i".

This works because MultiGED's flagged tokens are a strict subset of the M2
flagged tokens, and every one of them sits on an M2 replacement edit -- so each
gold "i" token gets a real type, and no binary number changes.

Unit = sentence, taken from the reconstructed sentences file (tokens + labels,
from reconstruct_from_tsv.py). The M2 is aligned to it on the flat token stream,
not on record index: the two sources agree on tokens but not always on sentence
segmentation (an M2 record can hold two MultiGED sentences, and MultiGED drops
records that are nothing but a transcription placeholder). We still assert the
token lists match per sentence, so any real source drift fails loudly.

Other datasets: write a sibling builder that emits the same canonical shape.
Nothing downstream is FCE-specific.
"""

import argparse
import json
import sys
from pathlib import Path

from common import REPO_ROOT, save_json

sys.path.insert(0, str(REPO_ROOT))
import paths  # noqa: E402


# Merlin transcription placeholders. MultiGED strips these from its TSV (see the
# german README) but they are still present in the M2, so they must be dropped
# before the two token streams can be compared.
PLACEHOLDER_TOKENS = {"-unreadable-", "-image-", "-------"}


def parse_m2(path):
    """Return a list of sentences: {tokens, edits} where edits are (start, end, type)."""
    sentences = []
    current = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("S "):
            if current is not None:
                sentences.append(current)
            current = {"tokens": line[2:].split(" "), "edits": []}
        elif line.startswith("A "):
            parts = line[2:].split("|||")
            span = parts[0].split()
            start, end = int(span[0]), int(span[1])
            current["edits"].append((start, end, parts[1]))
    if current is not None:
        sentences.append(current)
    return sentences


def sentence_token_types(sentence):
    """Per-token M2 type for one sentence (None where no edit covers the token).

    When several edits touch one token, the edit that best explains why the
    token is flagged wins, in priority order: replacement > deletion > insertion.
    Replacements physically change the token; a deletion says it shouldn't be
    there; an insertion is only adjacent to it. Within a priority level the
    first edit wins.

    Span-covering edits (start < end) cover tokens [start, end); insertions
    (start == end) attach to the token after the gap, clamped to the last token
    for end-of-sentence insertions.
    """
    n = len(sentence["tokens"])
    types = [None] * n
    edits = [(a, b, t) for a, b, t in sentence["edits"] if t != "noop"]

    def cover_span(selected):
        for start, end, etype in selected:
            for i in range(start, min(end, n)):
                if types[i] is None:
                    types[i] = etype

    cover_span([e for e in edits if e[0] < e[1] and e[2].startswith("R:")])  # replacements
    cover_span([e for e in edits if e[0] < e[1] and not e[2].startswith("R:")])  # deletions/other

    for start, end, etype in edits:  # insertions fill the gaps
        if start == end:
            idx = start if start < n else n - 1
            if 0 <= idx < n and types[idx] is None:
                types[idx] = etype
    return types


def flatten_m2(m2):
    """Flatten the M2 to one token stream plus a per-token type.

    Types are resolved per M2 record first, where the edit offsets are valid,
    and only then flattened -- so dropping a placeholder token can never shift
    an offset. Records that are nothing but placeholders vanish entirely, which
    is exactly what MultiGED did to them.
    """
    tokens, types = [], []
    for sentence in m2:
        sentence_types = sentence_token_types(sentence)
        for token, ttype in zip(sentence["tokens"], sentence_types):
            if token in PLACEHOLDER_TOKENS:
                continue
            tokens.append(token)
            types.append(ttype)
    return tokens, types


def build(sentences_path, m2_path, out_path, dataset="dataset"):
    """Build sentence-level gold from reconstructed sentences (+ optional m2 types).

    m2_path=None (a language with no error-type annotations) yields a valid
    detection-only gold: labels are kept, every ``types`` entry is None.
    """
    sentences = json.loads(Path(sentences_path).read_text())
    m2 = parse_m2(m2_path) if m2_path else None

    if m2 is not None:
        m2_tokens, m2_token_types = flatten_m2(m2)
        n_tokens = sum(len(s["tokens"]) for s in sentences)
        if n_tokens != len(m2_tokens):
            raise ValueError(
                f"{n_tokens} reconstructed tokens but {len(m2_tokens)} M2 tokens"
            )

    units = []
    unspecified = 0  # gold "i" tokens with no M2 type -- expected to be zero
    cursor = 0
    for unit_id, sent in enumerate(sentences):
        tokens = sent["tokens"]
        labels = sent["labels"]

        if m2 is not None:
            span = slice(cursor, cursor + len(tokens))
            if tokens != m2_tokens[span]:
                raise ValueError(
                    f"unit {unit_id}: sentence tokens do not match M2 tokens (source drift)"
                )
            m2_types = m2_token_types[span]
            cursor += len(tokens)
            # Types only for MultiGED-flagged tokens; correct tokens stay None.
            types = []
            for label, mtype in zip(labels, m2_types):
                if label == "i":
                    if mtype is None:
                        unspecified += 1
                        types.append("UNSPEC")
                    else:
                        types.append(mtype)
                else:
                    types.append(None)
        else:
            types = [None] * len(tokens)  # detection-only: no error types available

        units.append(
            {
                "unit_id": unit_id,
                "tokens": tokens,
                "labels": labels,
                "types": types,
            }
        )

    gold = {
        "dataset": dataset,
        "convention": "multiged-binary" + ("+m2-types" if m2 is not None else " (detection-only)"),
        "unit": "sentence",
        "units": units,
    }
    save_json(out_path, gold)

    total_tokens = sum(len(u["tokens"]) for u in units)
    i_tokens = sum(lbl == "i" for u in units for lbl in u["labels"])
    note = f"unspecified_type={unspecified}" if m2 is not None else "detection-only (no m2)"
    print(f"wrote {out_path}")
    print(f"units={len(units)} tokens={total_tokens} i_tokens={i_tokens} {note}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="dataset id, e.g. english/en_fce_dev (or bare stem).")
    parser.add_argument("--sentences", type=Path, default=None, help="override sentences json.")
    parser.add_argument("--m2", type=Path, default=None,
                        help="override m2 (types); omit to auto-detect, absent -> detection-only.")
    parser.add_argument("--out", type=Path, default=None, help="override gold output path.")
    args = parser.parse_args()

    if not args.dataset and not (args.sentences and args.out):
        parser.error("give --dataset (e.g. english/en_fce_dev), or both --sentences and --out")

    sentences_path = args.sentences or paths.sentences(args.dataset)
    m2_path = args.m2 or (paths.m2(args.dataset) if args.dataset else None)
    out_path = args.out or paths.gold(args.dataset)
    name = paths.stem(args.dataset) if args.dataset else Path(sentences_path).stem
    build(sentences_path, m2_path, out_path, dataset=name)


if __name__ == "__main__":
    main()
