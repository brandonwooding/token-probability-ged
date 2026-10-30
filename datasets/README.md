# Datasets

No corpus text is tracked. Each folder holds the scripts that install a corpus
in the exact form the reported numbers were computed on, and `.gitignore` keeps
the installed files out of git.

| Folder | Corpus | Role |
|---|---|---|
| `multiged-2023/` | MultiGED-2023: Czech GECCC, English FCE + REALEC, German Falko-MERLIN, Italian MERLIN, Swedish SweLL-gold | Evaluation target for every reported number |
| `write-and-improve/` | Write & Improve 2024, in the same format plus a CEFR sidecar | Extra English evaluation, per-proficiency breakdowns |
| `synthetic-data/` | C4_200M (en, de, es, ro, ru) and MLCheck (en, de, it, sv), annotated with mERRANT | Adaptation training data and synthetic threshold tuning |

## Dataset ids

[`../paths.py`](../paths.py) resolves a **dataset id**, `<language>/<stem>` or
just `<stem>`, across all three folders. Every script takes `--dataset`.

```
<folder>/<language>/<stem>.tsv             token <TAB> c|i, blank line between sentences
                    <stem>.m2              error types (optional; else detection-only)
                    <stem>.cefr.json       W&I learner metadata (optional)
                    <stem>.sentences.json  generated: model input text + token spans
                    <stem>.gold.json       generated: scorer reference
```

The two generated files rebuild from the TSV with the standard library alone;
`standard-eval/run.py` makes any that are missing (`--rebuild` forces it):

```bash
python reconstruct_from_tsv.py --dataset english/en_fce_dev      # .sentences.json
python standard-eval/build_gold.py --dataset english/en_fce_dev  # .gold.json
```

Labels always come from the TSV exactly as distributed. An `.m2` only attaches
an error type to tokens the TSV already marks `i`, aligned on the flat token
stream (see [`../standard-eval/README.md`](../standard-eval/README.md)).

## 1. MultiGED-2023

```bash
git clone https://github.com/spraakbanken/multiged-2023 /tmp/multiged-2023
# FCE error types: fce_v2.1.bea19.tar.gz from https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
python datasets/multiged-2023/prepare.py --multiged /tmp/multiged-2023 --bea-fce /path/to/fce/m2
```

`prepare.py` copies the train/dev/test TSVs, restores nine tokens upstream
blanks (any token starting with uppercase `NA`: six in FCE train, three in
Falko-MERLIN train), installs the BEA-2019 FCE M2s and upstream's `fm-*.m2` as
`de_falko-merlin_*.m2`, and checks every file against the SHA-256 of the copy
used for the results. Without `--bea-fce` the FCE train repair cannot run.

The linear probes (not in the paper) also used a 90/10 sentence split of the train sets:

```bash
python secondary/linear-probes/split_train_dev.py --input datasets/multiged-2023/german/de_falko-merlin_train.tsv \
    --train-out datasets/multiged-2023/german/de_falko-merlin_train_split.tsv \
    --val-out datasets/multiged-2023/german/de_falko-merlin_val_split.tsv
```

This reproduces the German split exactly. The English split was cut before the
`NA` repair, so re-cutting it gives a split six tokens different from the one
used.

## 2. Write & Improve 2024

Register for the [Write & Improve Corpus 2024 v2](https://englishlanguageitutoring.com/datasets/write-and-improve-corpus-2024),
unpack it to `datasets/write-and-improve/source/write-and-improve-corpus-2024-v2/`,
then build with the dataset environment (see `synthetic-data/` below):

```bash
python datasets/write-and-improve/build_wi_ged.py --selftest
python datasets/write-and-improve/build_wi_ged.py --split dev
python datasets/write-and-improve/build_wi_ged.py --split train
python datasets/write-and-improve/validate.py
```

ERRANT is re-run on the corpus's aligned sentence pairs rather than projecting
its whitespace-token M2; [`write-and-improve/README.md`](write-and-improve/README.md)
explains why, the labelling policy, and how the hidden test set was handled.

## 3. Synthetic data

C4_200M splits for all five languages, with every view (`.tsv`, `.m2`,
`.jsonl`, `.pairs.tsv`) and the full source sentence-pair files, are on Hugging
Face:

```bash
hf download brandonwooding/token-probability-ged --repo-type dataset \
    --include "synthetic-data/*" --local-dir datasets
```

MLCheck cannot be redistributed, and both corpora can be rebuilt from scratch;
see [`synthetic-data/README.md`](synthetic-data/README.md).

## Licences

| Corpus | Terms |
|---|---|
| Czech GECCC, English REALEC, German Falko-MERLIN, Italian MERLIN | CC BY-SA 4.0 |
| Swedish SweLL-gold | MultiGED redistribution CC BY 4.0 (underlying corpus CLARIN-ID/-PRIV/-NORED/-BY) |
| English FCE | Custom Cambridge licence: non-commercial research only, non-transferable |
| Write & Improve 2024 | Provider terms; registration required |
| [MLCheck](https://github.com/ubisoft/ubisoft-laforge-MultiLingualCheck-2M) | CC BY-NC-ND 4.0 |
| C4_200M synthetic | Derived from C4_200M edits (CC BY 4.0) applied to C4 / mC4 text (ODC-BY) |
