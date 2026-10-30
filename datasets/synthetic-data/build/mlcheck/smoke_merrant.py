#!/usr/bin/env python3
"""Vet mERRANT on a sentence-pair corpus before committing to a full build.

Runs build.py's own code path -- same normalisation, same Unicode filter, same
annotation and GED projection -- over the first N pairs of each language and
reports what the annotator is actually able to do with them.

The checks that matter per language:

  * pipeline_components: mERRANT's alignment cost discounts a substitution when
    two tokens share a lemma or a POS tag (utils._token_sub_cost). A model with
    no lemmatizer or morphologizer flattens that cost, so the aligner leans
    harder on 1:1 substitution and detects fewer insertions and deletions.
  * whitespace_crossing: build.py refuses tokens containing whitespace, because
    token-indexed M2 separates tokens with spaces and cannot round-trip them.
  * untyped_edits: edits whose classifier tag came back empty -- serialised as
    a bare "R:" -- which is what a missing POS tagger produces.
  * the label rates, to compare against the C4_200M sets and the MultiGED dev
    sets before the build rather than after.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "merrant"))

import spacy  # noqa: E402
import build as builder  # noqa: E402
from merrant import classification  # noqa: E402


def smoke(lang: str, pair_root: Path, limit: int) -> dict:
    folder, model_name = builder.LANGUAGES[lang]
    source_path = pair_root / f"{lang}.sentence_pairs.tsv"
    nlp = spacy.load(model_name)
    tokenizer_rules = getattr(nlp.tokenizer, "rules", None) or {}
    whitespace_rules = [key for key in tokenizer_rules
                        if any(character.isspace() for character in key)]
    if whitespace_rules:
        nlp.tokenizer.rules = {key: value for key, value in tokenizer_rules.items()
                               if key not in whitespace_rules}
    for pipe in ("parser", "ner"):
        if pipe in nlp.pipe_names:
            nlp.disable_pipes(pipe)
    classifier = classification.GenericClassifier(aspell_lang=None)
    classifier.initialize()

    rows = []
    for record in builder.read_selected(source_path, limit):
        rows.append((record[2], record[3]))

    error_types = Counter()
    sentences = errored = tokens = incorrect = untyped = 0
    lemma_seen = morph_seen = 0
    crossing = []
    failures = []

    source_docs = list(nlp.pipe([row[0] for row in rows], batch_size=256))
    target_docs = list(nlp.pipe([row[1] for row in rows], batch_size=256))
    for (source, target), source_doc, target_doc in zip(rows, source_docs, target_docs):
        for doc in (source_doc, target_doc):
            crossing.extend(token.text for token in doc
                            if any(character.isspace() for character in token.text))
        lemma_seen += any(token.lemma != 0 for token in source_doc)
        morph_seen += any(len(token.morph) > 0 for token in source_doc)
        try:
            edits = classifier.classify(source_doc, target_doc)
            rich = builder.make_record(lang, folder, source, target,
                                       source_doc, target_doc, edits, 0)
        except Exception as error:  # noqa: BLE001 - the point is to surface it
            failures.append(f"{type(error).__name__}: {error}")
            continue
        sentences += 1
        labels = rich["labels"]
        tokens += len(labels)
        incorrect += labels.count("i")
        errored += "i" in labels
        for edit in rich["edits"]:
            raw = edit["error_type"]
            error_types[raw] += 1
            tag = raw.partition(":")[2] if ":" in raw else raw
            if not tag:
                untyped += 1

    total_edits = sum(error_types.values())
    return {
        "language": lang,
        "model": model_name,
        "pipeline_components": list(nlp.pipe_names),
        "has_lemmatizer": "lemmatizer" in nlp.pipe_names,
        "has_morphologizer": "morphologizer" in nlp.pipe_names,
        "lemmas_populated_pct": round(100 * lemma_seen / max(len(rows), 1), 1),
        "morph_populated_pct": round(100 * morph_seen / max(len(rows), 1), 1),
        "sampled": len(rows),
        "annotated": sentences,
        "failures": failures[:5],
        "failure_count": len(failures),
        "whitespace_crossing_tokens": len(crossing),
        "pct_errored_sentences": round(100 * errored / max(sentences, 1), 1),
        "pct_incorrect_tokens": round(100 * incorrect / max(tokens, 1), 1),
        "edits": total_edits,
        "untyped_edits": untyped,
        "untyped_pct": round(100 * untyped / max(total_edits, 1), 1),
        "top_error_types": dict(error_types.most_common(8)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", default="de,en,es,it,ru,sv,zh")
    parser.add_argument("--pair-root", type=Path, default=HERE.parents[1] / "sources" / "mlcheck" / "pairs")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    for lang in [item.strip() for item in args.langs.split(",") if item.strip()]:
        print(json.dumps(smoke(lang, args.pair_root, args.limit),
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
