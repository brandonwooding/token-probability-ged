#!/usr/bin/env python3
"""Build canonical C4 sentence-pair, M2, GED TSV, and rich JSONL views.

One full mERRANT annotation is the sole edit authority.  Every output view is
written from the same in-memory EditSpan objects; no later realignment or
positional join is permitted.  Overlapping split ranges (the nested scaling
sets) share that annotation pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import spacy

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DATA_ROOT = PROJECT_ROOT / "datasets/synthetic-data"
SOURCES = DATA_ROOT / "sources"
MERRANT_PARENT = SOURCES / "C4_200M/multilingual"
sys.path.insert(0, str(MERRANT_PARENT))
from merrant import classification  # noqa: E402


# Sentence-pair corpora sharing the `<lang>.sentence_pairs.tsv` dialect: column
# one corrupted source, column two clean target. The key tags every output stem,
# so one language can be built from both corpora without the two sets colliding
# in the same language folder.
CORPORA = {
    "c4200": SOURCES / "c4200m_multilingual/pairs",
    "mlcheck": SOURCES / "mlcheck/pairs",
}
PAIR_ROOT = CORPORA["c4200"]

LANGUAGES = {
    "en": ("english", "en_core_web_sm"),
    "de": ("german", "de_core_news_sm"),
    "es": ("spanish", "es_core_news_sm"),
    "ro": ("romanian", "ro_core_news_sm"),
    "ru": ("russian", "ru_core_news_sm"),
    "it": ("italian", "it_core_news_sm"),
    "sv": ("swedish", "sv_core_news_sm"),
    "zh": ("chinese", "zh_core_web_sm"),
}

# Keep the established evaluation and validation sets. All training sizes are
# nested views of one 100k training pool.
DEFAULT_SPLITS = (
    ("eval5k", 0, 5_000),
    ("val5k", 55_000, 60_000),
    ("train10k", 60_000, 70_000),
    ("train20k", 60_000, 80_000),
    ("train50k", 60_000, 110_000),
    ("train100k", 60_000, 160_000),
)

SCHEMA = "c4-merrant-rich-v1"
POLICY = "exclude-noop-unk-punct"
MERRANT_COMMIT = "fc8409b89317efac81dba0ba8788c744d19ffb2c"
INVALID_UNICODE_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}


def normalise(text: str) -> str:
    """Collapse whitespace before tokenisation so every view shares offsets."""
    return " ".join(text.split())


def has_invalid_unicode(text: str) -> bool:
    """Reject replacement, control, formatting, and private-use characters."""
    return "\ufffd" in text or any(
        unicodedata.category(character) in INVALID_UNICODE_CATEGORIES
        for character in text
    )


def escape_tsv(token: str) -> str:
    """MultiGED disk dialect consumed by csv.reader(..., escapechar='\\')."""
    return token.replace("\\", "\\\\").replace('"', '\\"')


def parse_splits(values: list[str] | None) -> list[tuple[str, int, int]]:
    if not values:
        return list(DEFAULT_SPLITS)
    out = []
    for value in values:
        name, start, end = value.split(":")
        start, end = int(start), int(end)
        if not name or start < 0 or end <= start:
            raise ValueError(f"invalid split {value!r}")
        out.append((name, start, end))
    names = [name for name, _, _ in out]
    if len(names) != len(set(names)):
        raise ValueError("split names must be unique")
    return sorted(out, key=lambda item: (item[1], item[2], item[0]))


def is_excluded(error_type: str) -> bool:
    base = error_type.split(":")[-1]
    return error_type in {"noop", "UNK"} or base in {"UNK", "PUNCT"}


def token_record(token) -> dict:
    return {"text": token.text, "start": token.idx, "end": token.idx + len(token.text)}


def edit_record(edit, n_source: int) -> dict:
    ss, se = edit.source_span.start, edit.source_span.end
    ts, te = edit.target_span.start, edit.target_span.end
    error_type = edit.get_label()
    if edit.is_deletion():
        operation, kind = "U", "deletion"
    elif edit.is_insertion():
        operation, kind = "M", "insertion"
    else:
        operation, kind = "R", "replacement"
    included = not is_excluded(error_type)
    sentence_final = bool(edit.is_insertion() and ss >= n_source)
    if not included:
        label_indices = []
    elif ss < se:
        label_indices = list(range(ss, se))
    elif ss < n_source:
        label_indices = [ss]  # MultiGED: missing material marks the next token.
    else:
        label_indices = []  # Official conversion has no right token at EOS.
    return {
        "source_start": ss,
        "source_end": se,
        "target_start": ts,
        "target_end": te,
        "operation": operation,
        "kind": kind,
        "error_type": error_type,
        "included_in_ged": included,
        "label_indices": label_indices,
        "sentence_final_missing": sentence_final,
    }


def make_record(lang: str, folder: str, source: str, target: str,
                source_doc, target_doc, edits, source_line_index: int) -> dict:
    rich_edits = [edit_record(edit, len(source_doc)) for edit in edits]
    labels = ["c"] * len(source_doc)
    for edit in rich_edits:
        for index in edit["label_indices"]:
            labels[index] = "i"
    pair_digest = hashlib.sha256(
        (lang + "\0" + source + "\0" + target).encode("utf-8")).hexdigest()
    return {
        "schema": SCHEMA,
        "pair_id": pair_digest,
        "source_line_index": source_line_index,
        "source": source,
        "target": target,
        # Keep the historical field names because A/D consumes them directly.
        "source_words": [token_record(token) for token in source_doc],
        "target_words": [token_record(token) for token in target_doc],
        "labels": labels,
        "edits": rich_edits,
        "alignment": {"authority": "mERRANT", "policy": POLICY},
        "lang": lang,
        "language_folder": folder,
    }


def m2_block(record: dict) -> str:
    lines = ["S " + " ".join(word["text"] for word in record["source_words"])]
    if not record["edits"]:
        lines.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
    else:
        target_words = record["target_words"]
        for edit in record["edits"]:
            correction = " ".join(
                word["text"] for word in
                target_words[edit["target_start"]:edit["target_end"]])
            lines.append(
                f'A {edit["source_start"]} {edit["source_end"]}|||'
                f'{edit["error_type"]}|||{correction}|||REQUIRED|||-NONE-|||0')
    return "\n".join(lines) + "\n\n"


@dataclass
class SplitWriter:
    name: str
    start: int
    end: int
    out_dir: Path
    prefix: str
    model_name: str
    lang: str
    force: bool
    stack: ExitStack
    handles: dict = field(default_factory=dict)
    temporary: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    error_types: Counter = field(default_factory=Counter)
    excluded_types: Counter = field(default_factory=Counter)

    def __post_init__(self):
        stem = f"{self.prefix}_{self.name}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        suffixes = {"pairs": ".pairs.tsv", "tsv": ".tsv", "m2": ".m2",
                    "jsonl": ".jsonl"}
        for key, suffix in suffixes.items():
            final = self.out_dir / f"{stem}{suffix}"
            if final.exists() and not self.force:
                raise FileExistsError(f"refusing to overwrite {final}; pass --force")
            temp = final.with_suffix(final.suffix + ".tmp")
            if temp.exists():
                temp.unlink()
            self.final[key], self.temporary[key] = final, temp
            self.handles[key] = self.stack.enter_context(temp.open("w", encoding="utf-8"))
        self.final["manifest"] = self.out_dir / f"{stem}.manifest.json"
        if self.final["manifest"].exists() and not self.force:
            raise FileExistsError(
                f"refusing to overwrite {self.final['manifest']}; pass --force")

    def accepts(self, source_line_index: int) -> bool:
        return self.start <= source_line_index < self.end

    def write(self, record: dict):
        self.handles["pairs"].write(record["source"] + "\t" + record["target"] + "\n")
        self.handles["jsonl"].write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.handles["m2"].write(m2_block(record))
        for word, label in zip(record["source_words"], record["labels"]):
            self.handles["tsv"].write(f'{escape_tsv(word["text"])}\t{label}\n')
            self.counts["tokens"] += 1
            self.counts["incorrect_tokens"] += label == "i"
        self.handles["tsv"].write("\n")
        self.counts["sentences"] += 1
        self.counts["edits"] += len(record["edits"])
        self.counts["pairs_written"] += 1
        if not record["edits"]:
            self.counts["noop_sentences"] += 1
        for edit in record["edits"]:
            self.error_types[edit["error_type"]] += 1
            if edit["included_in_ged"]:
                self.counts["included_edits"] += 1
            else:
                self.counts["excluded_edits"] += 1
                self.excluded_types[edit["error_type"]] += 1
            self.counts["sentence_final_missing"] += edit["sentence_final_missing"]

    def finish(self, source_path: Path, skipped_malformed: int,
               skipped_invalid_unicode: int, spacy_version: str,
               aspell_lang: str | None, whitespace_rules_removed: int,
               pipeline_components: list[str]):
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        for key, temp in self.temporary.items():
            temp.replace(self.final[key])
        expected = self.end - self.start
        try:
            source_display = str(source_path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            source_display = str(source_path.resolve())
        manifest = {
            "schema": SCHEMA,
            "dataset_id": f"{self.out_dir.name}/{self.prefix}_{self.name}",
            "language": self.lang,
            "source": source_display,
            "source_line_range": {"start": self.start, "end": self.end},
            "expected_pairs": expected,
            "sampling": (
                "contiguous range after invalid-Unicode filtering of the "
                "MD5-ordered sentence-pair file"
            ),
            "annotator": {
                "name": "mERRANT",
                "implementation_commit": MERRANT_COMMIT,
                "spacy_model": self.model_name,
                "spacy_version": spacy_version,
                "aspell_lang": aspell_lang,
                "whitespace_crossing_tokenizer_rules_removed": whitespace_rules_removed,
                "pipeline_components": pipeline_components,
            },
            "ged_policy": {
                "name": POLICY,
                "exclude": ["noop", "UNK", "*:PUNCT"],
                "span_projection": "all retained source-span tokens",
                "missing_projection": "next source token; none at sentence end",
                "m2_contains_excluded_edits": True,
            },
            "counts": dict(self.counts),
            "error_types": dict(self.error_types.most_common()),
            "excluded_error_types": dict(self.excluded_types.most_common()),
            "malformed_source_lines_seen_before_global_stop": skipped_malformed,
            "invalid_unicode_pairs_seen_before_global_stop": skipped_invalid_unicode,
            "invalid_unicode_filter": {
                "replacement_character": True,
                "unicode_categories": sorted(INVALID_UNICODE_CATEGORIES),
            },
            "files": {key: path.name for key, path in self.final.items()
                      if key != "manifest"},
        }
        temporary = self.final["manifest"].with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(self.final["manifest"])


def read_selected(path: Path, stop: int):
    skipped_malformed = 0
    skipped_invalid_unicode = 0
    pair_index = 0
    with path.open(encoding="utf-8") as stream:
        for source_line_index, line in enumerate(stream):
            if pair_index >= stop:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
                skipped_malformed += 1
                continue
            source, target = normalise(fields[0]), normalise(fields[1])
            if has_invalid_unicode(source) or has_invalid_unicode(target):
                skipped_invalid_unicode += 1
                continue
            yield (pair_index, source_line_index, source, target,
                   skipped_malformed, skipped_invalid_unicode)
            pair_index += 1
    if pair_index < stop:
        raise ValueError(f"only {pair_index} valid pairs available; need {stop}")


def build_language(lang: str, splits: list[tuple[str, int, int]], *,
                   pair_root: Path, data_root: Path, model_override: str | None,
                   batch_size: int, force: bool, aspell_lang: str | None,
                   corpus: str):
    folder, default_model = LANGUAGES[lang]
    prefix = f"{lang}_{corpus}"
    model_name = model_override or default_model
    source_path = pair_root / f"{lang}.sentence_pairs.tsv"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    nlp = spacy.load(model_name)
    # Token-indexed M2 serializes an S line by separating tokens with spaces.
    # A few language models contain special cases such as Spanish `EE. UU.`
    # that make a *single* token span whitespace, which M2 cannot round-trip.
    # Removing only those exceptions makes whitespace a hard token boundary;
    # all ordinary language-specific tokenizer rules remain intact.
    # ChineseTokenizer exposes no `rules` mapping at all; treat that as empty.
    tokenizer_rules = getattr(nlp.tokenizer, "rules", None) or {}
    whitespace_rules = [key for key in tokenizer_rules
                        if any(character.isspace() for character in key)]
    if whitespace_rules:
        nlp.tokenizer.rules = {
            key: value for key, value in tokenizer_rules.items()
            if key not in whitespace_rules
        }
    # Dependency parsing and NER do not enter mERRANT's alignment/classifier;
    # POS, morphology, lemmas, and tokenizer rules do. Avoid paying for the two
    # irrelevant components over 165k pairs per language.
    disabled = [name for name in ("parser", "ner") if name in nlp.pipe_names]
    if disabled:
        nlp.disable_pipes(*disabled)
    classifier = classification.GenericClassifier(aspell_lang=aspell_lang)
    classifier.initialize()
    stop = max(end for _, _, end in splits)
    skipped_malformed = 0
    skipped_invalid_unicode = 0
    with ExitStack() as stack:
        writers = [SplitWriter(name, start, end, data_root / folder, prefix,
                               model_name, lang, force, stack)
                   for name, start, end in splits]
        chunk = []

        def flush():
            nonlocal chunk
            if not chunk:
                return
            source_docs = list(nlp.pipe((row[2] for row in chunk), batch_size=batch_size))
            target_docs = list(nlp.pipe((row[3] for row in chunk), batch_size=batch_size))
            for (pair_index, source_line_index, source, target), source_doc, target_doc in zip(
                    chunk, source_docs, target_docs):
                for side, doc in (("source", source_doc), ("target", target_doc)):
                    crossing = [token.text for token in doc
                                if any(character.isspace() for character in token.text)]
                    if crossing:
                        raise ValueError(
                            f"{lang} line {source_line_index}: {side} tokens cross whitespace: "
                            f"{crossing!r}")
                edits = classifier.classify(source_doc, target_doc)
                record = make_record(lang, folder, source, target, source_doc, target_doc,
                                     edits, source_line_index)
                for writer in writers:
                    if writer.accepts(pair_index):
                        writer.write(record)
            chunk = []

        for (pair_index, source_line_index, source, target, skipped_malformed,
             skipped_invalid_unicode) in read_selected(source_path, stop):
            if any(start <= pair_index < end for _, start, end in splits):
                chunk.append((pair_index, source_line_index, source, target))
                if len(chunk) >= batch_size:
                    flush()
        flush()
        for writer in writers:
            writer.finish(source_path, skipped_malformed, skipped_invalid_unicode,
                          spacy.__version__, aspell_lang, len(whitespace_rules),
                          list(nlp.pipe_names))
            print(json.dumps({"language": lang, "split": writer.name,
                              **dict(writer.counts)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", default="en,de,es,ro,ru")
    parser.add_argument("--split", action="append",
                        help="repeatable NAME:START:END; defaults to the canonical suite")
    parser.add_argument("--corpus", default="c4200", choices=sorted(CORPORA),
                        help="sentence-pair corpus; also tags every output stem")
    parser.add_argument("--pair-root", type=Path, default=None,
                        help="override the selected corpus's pair directory")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--spacy-model",
                        help="override model for a single language (name or local path)")
    parser.add_argument("--aspell-lang",
                        help="optional Aspell code; disabled by default and only valid for one language")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    langs = [item.strip() for item in args.langs.split(",") if item.strip()]
    unknown = sorted(set(langs) - set(LANGUAGES))
    if unknown:
        parser.error(f"unknown languages: {', '.join(unknown)}")
    if (args.spacy_model or args.aspell_lang) and len(langs) != 1:
        parser.error("--spacy-model/--aspell-lang require one --langs value")
    splits = parse_splits(args.split)
    pair_root = args.pair_root or CORPORA[args.corpus]
    for lang in langs:
        build_language(lang, splits, pair_root=pair_root,
                       data_root=args.data_root, model_override=args.spacy_model,
                       batch_size=args.batch_size, force=args.force,
                       aspell_lang=args.aspell_lang, corpus=args.corpus)


if __name__ == "__main__":
    main()
