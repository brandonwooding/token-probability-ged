# Write & Improve 2024 → MultiGED-format GED

Token-labelled GED data built from the [Write & Improve Corpus 2024 v2](https://englishlanguageitutoring.com/datasets/write-and-improve-corpus-2024)
(final essay versions), in the same TSV format as MultiGED, plus a CEFR sidecar
for per-level breakdowns.

| Split | Sentences | Tokens | `i` rate | Essays |
|---|---:|---:|---:|---:|
| dev | 4,352 | 88,613 | 7.94% | 506 |
| train | 37,486 | 676,897 | 8.47% | 4,040 |

The corpus's test split has no public corrections, so only an unlabelled test
input is built.

## Build

Register for the corpus, unpack it to
`source/write-and-improve-corpus-2024-v2/` here, and use the dataset environment
(`datasets/synthetic-data/build/merrant/requirements.txt`):

```bash
cd datasets/write-and-improve
PY=../../.venv-data/bin/python
$PY build_wi_ged.py --selftest        # checks the labelling rule against FCE
$PY build_wi_ged.py --split dev       # ~30 s
$PY build_wi_ged.py --split train     # ~3 min
$PY validate.py

# unlabelled test input, from the public original sentences
$PY build_wi_ged.py --originals /path/to/en-writeandimprove-test.orig \
    --out-prefix english/en_wi2024_test_unlabelled
```

Outputs: `english/en_wi2024_<split>.tsv`, `.m2` and `.cefr.json`.

## How labels are made

ERRANT (English, spaCy `en_core_web_sm`) is re-run on each original/corrected
sentence pair. The corpus's own M2 is not reused: it is aligned to whitespace
tokens (`life.` is one token), which marks about 26% too many tokens once split
into spaCy tokens.

A token is `i` if it is inside an edit, except `noop`, `UNK`, non-zero
annotators and every punctuation edit (`*:PUNCT`). A missing word marks the
token to its right.

## CEFR sidecar

`<stem>.cefr.json` gives, per sentence (`unit_id` order): `human_cefr` (A1+…C2),
`human_cefr_coarse` (A/B/C), `auto_cefr`, `l1` and `prompt_id`. Use the human
grade. Score by level with:

```bash
python standard-eval/score.py --dataset english/en_wi2024_dev --pred <run>/predictions.json \
    --group-by human_cefr_coarse
```

Error rates differ a lot by level (A 11.5%, B 8.7%, C 3.3% on dev), so compare
levels with that in mind, and treat essays, not tokens, as the unit for
confidence intervals.

## Hidden test set

`assessor-package/` is a self-contained kit for the test-set holder: it builds
private gold from the test corrections with the same rules and scores a
submitted TSV. See its README.
