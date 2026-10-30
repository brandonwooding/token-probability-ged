"""Shared loading for the co-located canonical C4/mERRANT datasets.

Every representation of a split has one stem:

    datasets/synthetic-data/<folder>/<prefix>_<name>.tsv
    datasets/synthetic-data/<folder>/<prefix>_<name>.pairs.tsv
    datasets/synthetic-data/<folder>/<prefix>_<name>.m2
    datasets/synthetic-data/<folder>/<prefix>_<name>.jsonl

The files are emitted together by ``datasets/synthetic-data/build/merrant/build.py``.
Consequently the encoder labels, decoder pairs, and A/D edit spans cannot
silently drift through independent sampling or realignment.

**Subsampling is truncation, not sampling.** Both source files are ordered by
the MD5 of the clean target sentence, so their first N rows are already a
uniform random draw. Taking the first N therefore needs no seed *and* makes the
scaling curve nested -- the 20k set is a subset of the 100k set, so a difference
between two points on the curve is the extra data and nothing else. A fresh
random draw per size would confound the two.

**The seed controls mixing only**: once each language's slice is chosen, the
combined multilingual list is shuffled so that a batch is not all-German
followed by all-Russian. Same seed, same order, on any machine.
"""
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.normpath(os.path.join(HERE, "..", "..", ".env"))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
SYNTH_DATASETS = os.path.join(REPO_ROOT, "datasets", "synthetic-data")

# lang code -> dataset-tree language folder. The folder name is also the key
# prompts.py wording is looked up by, so it has to be the long form ("german",
# not "de").
LANGUAGES = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "ro": "romanian",
    "ru": "russian",
    "it": "italian",
    "sv": "swedish",
}

# Synthetic corpora built by c4_merrant/build.py --corpus. Both write into the
# same language folder; the stem prefix is what separates them, so
# `english/en_c4200_train20k.*` and `english/en_mlcheck_train20k.*` coexist.
# Not every language exists in both -- c4200 has ro and no it/sv, mlcheck the
# reverse -- and a missing file raises with the build command to run.
CORPORA = ("c4200", "mlcheck")
DEFAULT_CORPUS = "c4200"


def stem(lang, name, corpus=DEFAULT_CORPUS):
    return "%s_%s_%s" % (lang, corpus, name)

LABELS = ("c", "i")
LABEL_TO_ID = {"c": 0, "i": 1}


def hf_login():
    """Authenticate for gated checkpoints (Gemma), using the same .env the rest
    of the repo reads. Missing token is not fatal -- ungated models still load."""
    try:
        from dotenv import load_dotenv
        from huggingface_hub import login
    except ImportError:
        print("huggingface_login=skipped (huggingface_hub/dotenv not installed)")
        return
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("huggingface_login=skipped token_env_missing")
        return
    try:
        login(token=token, add_to_git_credential=False)
        print("huggingface_login=ok")
    except Exception as exc:
        print("huggingface_login=failed %s: %s" % (type(exc).__name__, exc))


def language_folder(lang):
    return LANGUAGES[lang]


def encoder_tsv(lang, name, corpus=DEFAULT_CORPUS):
    return os.path.join(SYNTH_DATASETS, LANGUAGES[lang],
                        "%s.tsv" % stem(lang, name, corpus))


def pairs_tsv(lang, name, corpus=DEFAULT_CORPUS):
    return os.path.join(SYNTH_DATASETS, LANGUAGES[lang],
                        "%s.pairs.tsv" % stem(lang, name, corpus))


def rich_jsonl(lang, name, corpus=DEFAULT_CORPUS):
    return os.path.join(SYNTH_DATASETS, LANGUAGES[lang],
                        "%s.jsonl" % stem(lang, name, corpus))


def m2_file(lang, name, corpus=DEFAULT_CORPUS):
    return os.path.join(SYNTH_DATASETS, LANGUAGES[lang],
                        "%s.m2" % stem(lang, name, corpus))


def read_label_tsv(path, limit=None):
    """MultiGED-dialect TSV -> [(tokens, labels)].

    The dialect matters: MultiGED writes a literal quote token as `\\"`, so the
    reader must be configured with this exact escapechar/quotechar pair or a
    single quoted token swallows every following line into one field.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"', escapechar="\\")
        tokens, labels = [], []
        for row in reader:
            if row and row[0].strip():
                tokens.append(row[0])
                labels.append(row[1])
            elif tokens:
                out.append((tokens, labels))
                tokens, labels = [], []
                if limit is not None and len(out) >= limit:
                    return out
        if tokens and (limit is None or len(out) < limit):
            out.append((tokens, labels))
    return out


def read_pairs_tsv(path, limit=None):
    """Raw source<TAB>target -> [(source, target)]. No quoting, by design."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                continue
            out.append((fields[0], fields[1]))
            if limit is not None and len(out) >= limit:
                break
    return out


def _mix(per_lang, seed):
    """Flatten {lang: [example, ...]} and shuffle once with `seed`."""
    combined = [ex for lang in sorted(per_lang) for ex in per_lang[lang]]
    random.Random(seed).shuffle(combined)
    return combined


def load_encoder(langs, name, max_per_lang=None, seed=0, corpus=DEFAULT_CORPUS):
    """-> [{tokens, labels, lang}], shuffled across languages."""
    per_lang = {}
    for lang in langs:
        path = encoder_tsv(lang, name, corpus)
        if not os.path.exists(path):
            raise SystemExit(
                "missing %s\n  build it with:\n"
                "  python datasets/synthetic-data/build/merrant/build.py "
                "--corpus %s --langs %s"
                % (path, corpus, lang))
        rows = read_label_tsv(path, limit=max_per_lang)
        per_lang[lang] = [{"tokens": t, "labels": l, "lang": lang} for t, l in rows]
        print("  [%s] %6d sentences from %s" % (lang, len(rows), os.path.basename(path)))
    return _mix(per_lang, seed)


def load_decoder(langs, name, max_per_lang=None, seed=0, corpus=DEFAULT_CORPUS):
    """-> [{source, target, lang, language_folder}], shuffled across languages."""
    per_lang = {}
    for lang in langs:
        path = pairs_tsv(lang, name, corpus)
        if not os.path.exists(path):
            raise SystemExit(
                "missing %s\n  build it with:\n"
                "  python datasets/synthetic-data/build/merrant/build.py "
                "--corpus %s --langs %s"
                % (path, corpus, lang))
        rows = read_pairs_tsv(path, limit=max_per_lang)
        per_lang[lang] = [{"source": s, "target": t, "lang": lang,
                           "language_folder": language_folder(lang)} for s, t in rows]
        print("  [%s] %6d pairs from %s" % (lang, len(rows), os.path.basename(path)))
    return _mix(per_lang, seed)


def align_labels_to_subtokens(encoding, batch_labels, index):
    """Word-level c/i labels -> per-subtoken ids, -100 on everything not scored.

    Only the FIRST subtoken of each word carries the label. The rest are masked
    out of the loss so that a word split into six pieces does not count six
    times as much as a word that stays whole -- the eval unit is the word, so
    the training unit is too.
    """
    word_ids = encoding.word_ids(index)
    out, previous = [], None
    for wid in word_ids:
        if wid is None or wid == previous:
            out.append(-100)
        else:
            out.append(LABEL_TO_ID[batch_labels[wid]])
        previous = wid
    return out
