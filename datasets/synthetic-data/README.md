# Synthetic GED data

Two synthetic corpora of (corrupted, clean) sentence pairs, each annotated once
with multilingual ERRANT (mERRANT) to give token-level `c`/`i` labels:

- **C4_200M** (en, de, es, ro, ru): Google's corruption edits applied to C4 /
  mC4 sentences. Public; on Hugging Face.
- **MLCheck** (en, de, it, sv): Ubisoft's [MultiLingualCheck](https://github.com/ubisoft/ubisoft-laforge-MultiLingualCheck-2M). CC BY-NC-ND 4.0,
  so it must be rebuilt locally.

## Get the C4_200M data

```bash
hf download brandonwooding/token-probability-ged --repo-type dataset \
    --include "synthetic-data/*" --local-dir datasets
```

This gives every split below plus the full source sentence pairs in
`sources/c4200m_multilingual/pairs/`, so the mERRANT step can be rerun without
rebuilding the pairs.

## What a split contains

Each split, e.g. `english/en_c4200_train20k`, has five files from one mERRANT pass:

| File | Contents |
|---|---|
| `.tsv` | tokens + `c`/`i` labels (what scoring and probes read) |
| `.m2` | full mERRANT edits |
| `.jsonl` | token spans and edits (what adaptation training reads) |
| `.pairs.tsv` | the corrupted / clean sentence pair |
| `.manifest.json` | source line range, spaCy/mERRANT versions, counts (tracked in git) |

Labels: every token inside a kept edit is `i`; a missing word marks the token to
its right. `noop`, `UNK` and all punctuation edits are ignored.

Splits are fixed line ranges of the MD5-ordered pair file, identical for every
language. Training sizes are nested:

| Split | Lines | Used for |
|---|---|---|
| `eval5k` | 0–5k | synthetic evaluation and threshold tuning |
| `val5k` | 55k–60k | checkpoint selection |
| `train10k` / `20k` / `50k` / `100k` | from 60k | training (headline adapters: `train20k`) |

## Rebuild from scratch

Set up the dataset environment and the mERRANT code (Google's C4_200M repo,
pinned, plus one small patch making the Aspell import optional):

```bash
python3.12 -m venv .venv-data && .venv-data/bin/pip install -r datasets/synthetic-data/build/merrant/requirements.txt
git clone https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction \
    datasets/synthetic-data/sources/C4_200M
git -C datasets/synthetic-data/sources/C4_200M checkout fc8409b89317efac81dba0ba8788c744d19ffb2c
git -C datasets/synthetic-data/sources/C4_200M apply ../../build/merrant/upstream-classification.patch
```

**1. Sentence pairs** → `sources/<corpus>/pairs/<lang>.sentence_pairs.tsv`
(skip for C4_200M if you downloaded the pairs above).

```bash
PY=.venv-data/bin/python; C4=datasets/synthetic-data/build/c4200m
# English: download the published pairs, then sample 1.5M (seed 0)
$PY $C4/fetch_english.py && $PY $C4/sample_english.py
# de, es, ro, ru: match Google's edits against mC4 (PBS job, ~2 h on 32 cores)
qsub $C4/reconstruct.pbs
# MLCheck: put its per-language Arrow datasets (synthetic_all/) in sources/mlcheck/synthetic_all/, then
$PY datasets/synthetic-data/build/mlcheck/convert_arrow_to_pairs.py
```

The mC4 scan recovers ~58% of the published sentences (the rest are missing
from the Hugging Face mC4 mirror), so the pair files are best taken from Hugging
Face rather than rebuilt.

**2. Annotate and write the splits** (CPU):

```bash
$PY datasets/synthetic-data/build/merrant/build.py --langs en,de,es,ro,ru
$PY datasets/synthetic-data/build/merrant/build.py --corpus mlcheck --langs en,de,it,sv
$PY datasets/synthetic-data/build/merrant/validate_suite.py --data-root datasets/synthetic-data
```

A correct rebuild matches the `counts` in each tracked `.manifest.json`.
