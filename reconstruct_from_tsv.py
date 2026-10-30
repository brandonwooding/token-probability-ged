#!/usr/bin/env python3
"""Reconstruct sentence text + exact char spans from a MultiGED TSV -- one source.

The TSV alone gives tokens and c/i labels; we detokenise them back into
a natural sentence string and, crucially, emit each token's char span **by
construction** (we place the token, we record where it landed). So the spans are
always consistent with the text -- there is no text-to-token matching step that
can drift or raise AlignmentError.

Unit = sentence: the TSV's blank lines delimit sentences (2191 for FCE-dev), so
this is a sentence-level pipeline. Paragraph grouping lived only in the FCE JSONL
(\\n\\n), which we deliberately drop to remove the second source.

Detokenisation (three buckets + a both-sides bucket), building left to right:
  ATTACH_PREV  no space before   . , ! ? ; : ) ] } %  ''  n't  and 's/'m/'ve/...
  ATTACH_NEXT  no space after     ( [ { $ # £  ``  and an opening "
  ATTACH_BOTH  no space either    - – —                (compound joiners)
  "            toggles open/close: opening acts ATTACH_NEXT, closing ATTACH_PREV
  everything else                 space before (a normal word)

Gluing punctuation to the previous word keeps contractions ("do"+"n't" -> "don't")
and adjacent punctuation ("!"+"!" -> "!!") natural. The latter still tokenises as
one sub-token in the model, which the standard-eval pooling recovers (Lever A) --
this reconstruction is about a single self-consistent source, not about avoiding
merges.

Output: a flat JSON list of {text, tokens, labels, spans}, one per sentence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import paths

ATTACH_PREV = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%",
               "...", "....", "''", "”", "’", "»"}
ATTACH_NEXT = {"(", "[", "{", "$", "£", "#", "¿", "¡", "``", "“", "‘", "«"}
# Only characters that can appear *inside* a token are compound joiners. spaCy
# splits `well-known` into `well - known`, so a bare "-" must rejoin. It does NOT
# split `AT&T`, `R&D`, `and/or` or `5/10`, so a bare "&" or "/" can only have come
# from a spaced one and rejoining it corrupts the text ("came & bought" ->
# "came&bought", which then re-tokenises as a single token and never comes back).
ATTACH_BOTH = {"-", "–", "—"}
CONTRACTIONS = {"n't", "n’t"}
# A run of dots is one token in the gold ("...", "....", but also ".." and longer).
DOT_RUN_MIN = 2


def read_sentences(tsv_path: Path, unlabelled: bool = False):
    """TSV -> sentences of token/label pairs, or bare tokens when unlabelled."""
    sentences = []
    current = []
    with tsv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quotechar='"', escapechar="\\")
        for line_no, row in enumerate(reader, start=1):
            if not row or not row[0].strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            if not unlabelled and len(row) < 2:
                raise ValueError(f"{tsv_path}:{line_no}: expected TOKEN<TAB>LABEL")
            if unlabelled and len(row) != 1:
                raise ValueError(f"{tsv_path}:{line_no}: expected one TOKEN column")
            token = '"' if row[0] == r"\"" else row[0]
            current.append(token if unlabelled else (token, row[1]))
    if current:
        sentences.append(current)
    return sentences


def classify(token: str, quote_open: bool):
    """Return (attaches_to_prev, sets_glue_next, new_quote_open) for one token."""
    if token == '"':
        return (True, False, False) if quote_open else (False, True, True)
    if token in ATTACH_BOTH:
        return True, True, quote_open
    if (token in ATTACH_PREV or token in CONTRACTIONS
            or (len(token) >= DOT_RUN_MIN and set(token) == {"."})
            or token.startswith("'") or token.startswith("’")):
        return True, False, quote_open
    if token in ATTACH_NEXT:
        return False, True, quote_open
    return False, False, quote_open


def detokenise(tokens: list[str]) -> tuple[str, list[list[int]]]:
    """Join tokens into natural text; return (text, spans) with text[s:e] == token."""
    parts: list[str] = []
    spans: list[list[int]] = []
    pos = 0
    glue = False        # previous token forces the next one to attach
    quote_open = False
    for i, token in enumerate(tokens):
        attach_prev, sets_glue, quote_open = classify(token, quote_open)
        space_before = not (i == 0 or glue or attach_prev)
        if space_before:
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(token)
        pos += len(token)
        spans.append([start, pos])
        glue = sets_glue
    return "".join(parts), spans


def positions_for_words(offsets, word_spans, shift=0):
    """Return every model-token position overlapping each dataset word."""
    return [
        [shift + i for i, (start, end) in enumerate(offsets)
         if end > word_start and start < word_end]
        for word_start, word_end in word_spans
    ]


def build(tsv_path: Path, unlabelled: bool = False) -> list[dict]:
    units = []
    for sentence in read_sentences(tsv_path, unlabelled=unlabelled):
        tokens = sentence if unlabelled else [t for t, _ in sentence]
        text, spans = detokenise(tokens)
        for (start, end), token in zip(spans, tokens):
            if text[start:end] != token:  # by-construction invariant, checked loudly
                raise AssertionError(f"span mismatch: {text[start:end]!r} != {token!r}")
        unit = {"text": text, "tokens": tokens, "spans": spans}
        if not unlabelled:
            unit["labels"] = [label for _, label in sentence]
        units.append(unit)
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="dataset id, e.g. english/en_fce_dev (or bare stem).")
    parser.add_argument("--tsv", type=Path, default=None, help="override input TSV.")
    parser.add_argument("--output", type=Path, default=None,
                        help="override output sentences json.")
    parser.add_argument("--check", action="store_true",
                        help="Validate and print a summary without writing output.")
    parser.add_argument("--unlabelled", action="store_true",
                        help="accept a one-column token TSV and omit labels from JSON.")
    args = parser.parse_args()

    if not args.dataset and not args.tsv:
        parser.error("give --dataset (e.g. english/en_fce_dev) or --tsv")
    tsv_path = args.tsv or paths.tsv(args.dataset)
    out_path = args.output or (
        paths.sentences(args.dataset) if args.dataset
        else tsv_path.with_suffix(".sentences.json")
    )

    units = build(tsv_path, unlabelled=args.unlabelled)
    tokens = sum(len(u["tokens"]) for u in units)
    print(f"sentences={len(units)} tokens={tokens}  <- {tsv_path}")

    if args.check:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(units, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
