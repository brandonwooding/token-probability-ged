"""Canonical paths for the multilingual GED pipeline -- the single naming authority.

A *dataset id* selects one corpus split, given either as "<language>/<stem>"
(explicit) or as a bare "<stem>" (the language folder is found by scanning):

    english/en_fce_dev      en_fce_dev      german/de_falko-merlin_dev

Raw inputs live in <corpus root>/<language>/ (tracked). There is more than one
corpus root because not every dataset is MultiGED: the C4_200M-derived synthetic
sets are machine-labelled data of a different provenance and are kept in their
own tree rather than mixed into the shared benchmark. Dataset *ids* are still
just "<language>/<stem>" -- the root a stem lives under is an implementation
detail resolved by scanning, so nothing downstream has to know about it.

    <stem>.tsv              tokens + c/i labels        (required)
    <stem>.m2               error types                (optional; None -> detection-only)
    <stem>.cefr.json        per-unit learner metadata  (optional; None -> no strata)
Generated, co-located, gitignored:
    <stem>.sentences.json   reconstruct_from_tsv.py    (model input text + spans)
    <stem>.gold.json        build_gold.py              (scorer reference)

Per-approach run outputs go under <approach>/outputs/<run>/, results under
standard-eval/results/. Nothing here touches algorithms -- only where files land.
"""

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Corpus roots, searched in order. A stem must be unique across all of them:
# _resolve raises on collisions rather than silently preferring one, because a
# silently preferred root would attach the wrong gold to a run.
MULTIGED_ROOT = REPO_ROOT / "datasets" / "multiged-2023"
SYNTHETIC_ROOT = REPO_ROOT / "datasets" / "synthetic-data"
# Write & Improve 2024: MultiGED-format but not part of the shared task, and the
# only corpus carrying per-text CEFR levels (see <stem>.cefr.json).
WI_ROOT = REPO_ROOT / "datasets" / "write-and-improve"
DATASET_ROOTS = (MULTIGED_ROOT, SYNTHETIC_ROOT, WI_ROOT)

# Back-compat alias: this used to be the single root.
DATASETS_ROOT = MULTIGED_ROOT

RESULTS_DIR = REPO_ROOT / "standard-eval" / "results"

# Unigram frequency tables, one per (model, language), counted from a frozen
# FineWeb/FineWeb-2 sample by secondary/token-frequencies/. Keyed by model_id -- the same
# `<name>.split("/")[-1]` a run folder uses -- so a run and its frequency table
# are named by the same string. Two models sharing a vocabulary still get their
# own table; see secondary/token-frequencies/sources.py for why.
FREQ_DIR = REPO_ROOT / "secondary" / "token-frequencies"
FREQ_TABLES = FREQ_DIR / "tables"
FREQ_SAMPLES = FREQ_DIR / "samples"

# short approach key -> its folder (where outputs/ lives)
APPROACH_DIRS = {
    "encoder-thresh": REPO_ROOT / "secondary" / "encoder-probability-thresh",
    "decoder-thresh": REPO_ROOT / "decoder-probability-thresh",
    "instruction": REPO_ROOT / "GED-by-instruction",
    "linear-probe": REPO_ROOT / "secondary" / "linear-probes",
    "confidence-clf": REPO_ROOT / "secondary" / "confidence-classifiers",
    # Fine-tuned encoder classifier. It emits P(correct) per sub-token in the
    # same data.json shape as the probability-threshold approaches, so it is a
    # threshold approach as far as everything downstream is concerned.
    "ft-encoder": REPO_ROOT / "finetuning",
    # Decoder hidden-state GED classifier (Family D). It emits P(correct) in
    # the same threshold-adapter shape, but is kept in its own experiment tree.
    "decoder-ged-head": REPO_ROOT / "finetuning" / "decoder_ged",
    # Encoder MLM-probability GED (Family E). Trains the masked-LM head itself
    # rather than a classifier, so it emits P(observed token) per sub-token in
    # exactly the shape encoder-thresh does -- low probability means suspicious.
    "encoder-ged-mlm": REPO_ROOT / "finetuning" / "encoder_ged",
}


def _resolve(dataset):
    """Dataset id -> (corpus_root, language, stem).

    Accepts '<lang>/<stem>' or a bare '<stem>'. Resolution is driven by which
    root actually holds the .tsv, so callers never name a corpus root. The .tsv
    is the one file that always exists (everything else is generated from it),
    which is why it -- and not the requested output file -- is what we look for.
    """
    if "/" in dataset:
        lang, stem_ = dataset.split("/", 1)
        hits = [r for r in DATASET_ROOTS if (r / lang / f"{stem_}.tsv").exists()]
        if not hits:
            searched = ", ".join(str(r / lang / f"{stem_}.tsv") for r in DATASET_ROOTS)
            raise ValueError(f"no dataset '{dataset}.tsv'; looked in {searched}")
        if len(hits) > 1:
            raise ValueError(
                f"dataset '{dataset}' exists in more than one corpus root: "
                + ", ".join(str(r) for r in hits)
            )
        return hits[0], lang, stem_

    matches = sorted(
        {(r, p.parent.name) for r in DATASET_ROOTS for p in r.glob(f"*/{dataset}.tsv")},
        key=lambda rl: (str(rl[0]), rl[1]),
    )
    if not matches:
        roots = ", ".join(str(r) for r in DATASET_ROOTS)
        raise ValueError(f"no dataset '{dataset}.tsv' under any of: {roots}")
    if len(matches) > 1:
        where = [f"{r.name}/{lang}" for r, lang in matches]
        raise ValueError(
            f"ambiguous dataset '{dataset}' in {where}; qualify as '<lang>/{dataset}'"
        )
    return matches[0][0], matches[0][1], dataset


def corpus_root(dataset):
    """The datasets/ subtree this dataset lives in."""
    return _resolve(dataset)[0]


def language(dataset):
    return _resolve(dataset)[1]


def stem(dataset):
    return _resolve(dataset)[2]


def tsv(dataset):
    root, lang, s = _resolve(dataset)
    return root / lang / f"{s}.tsv"


def m2(dataset):
    """Path to the co-located m2, or None if this dataset has no error-type file."""
    root, lang, s = _resolve(dataset)
    path = root / lang / f"{s}.m2"
    return path if path.exists() else None


def cefr(dataset):
    """Per-unit learner metadata, or None if this dataset has no level labels.

    Keyed by unit_id -- the positional index of the sentence in the .tsv, which
    is what every stage of standard-eval uses. Only Write & Improve has one.
    """
    root, lang, s = _resolve(dataset)
    path = root / lang / f"{s}.cefr.json"
    return path if path.exists() else None


def sentences(dataset):
    root, lang, s = _resolve(dataset)
    return root / lang / f"{s}.sentences.json"


def gold(dataset):
    root, lang, s = _resolve(dataset)
    return root / lang / f"{s}.gold.json"


def list_datasets(split=None):
    """All '<language>/<stem>' having a .tsv, optionally filtered by split suffix."""
    out = []
    for root in DATASET_ROOTS:
        for path in sorted(root.glob("*/*.tsv")):
            # Canonical synthetic splits co-locate source/target pairs as
            # `<stem>.pairs.tsv`; only the binary `<stem>.tsv` is a GED dataset.
            if path.name.endswith(".pairs.tsv"):
                continue
            if split is None or path.stem.endswith(f"_{split}"):
                out.append(f"{path.parent.name}/{path.stem}")
    return sorted(set(out))


def _run_name(dataset, model, tag=None, timestamp=None):
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [stem(dataset), model] + ([tag] if tag else []) + [ts]
    return "__".join(parts)


def run_dir(approach, dataset, model, tag=None, timestamp=None):
    """<approach>/outputs/<stem>__<model>[__<tag>]__<timestamp>/ for one run."""
    return APPROACH_DIRS[approach] / "outputs" / _run_name(dataset, model, tag, timestamp)


def result_path(approach, dataset, model, tag=None):
    """standard-eval/results/<approach>__<stem>__<model>[__<tag>].json"""
    parts = [approach, stem(dataset), model] + ([tag] if tag else [])
    return RESULTS_DIR / ("__".join(parts) + ".json")


def freqs(model_id, language):
    """Unigram frequency table for one model over one language, or None.

    `language` is the language folder name (as returned by language()), not a
    dataset id, because a table is a property of the language rather than of any
    one split -- en_fce_dev and en_fce_test share English frequencies.
    """
    path = FREQ_TABLES / f"{model_id}__{language}.freqs.json.gz"
    return path if path.exists() else None


def freq_sample(language):
    """The frozen document sample a language's tables were counted from."""
    return FREQ_SAMPLES / f"{language}.jsonl.gz"
