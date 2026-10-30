# Write & Improve hidden-test GED package

This folder lets the test-set holder construct private English GED gold from
line-aligned erroneous and corrected sentences, then score a submitted TSV.
The gold corrections, generated M2, and gold TSV can remain private.

## Why the corpus M2 is not converted directly

The released W&I M2 uses whitespace tokens, so punctuation remains attached to
words and its spans are too coarse for our spaCy-token GED data. Reprojecting
those spans marked about 26% more dev tokens than realigning the original and
corrected sentences directly. We therefore rerun English ERRANT on each paired
sentence. An independently built MultiGEC-2025 dev M2 using UDPipe and another
ERRANT implementation produced a very similar aggregate error count, supporting
the realignment decision.

## Convention and assumptions

- `--corrupted` is the erroneous learner/source text; `--clean` is its gold
  correction. Reversing these files reverses ERRANT operations and invalidates
  the gold.
- Each UTF-8 file contains exactly one sentence per line in identical order.
- Whitespace is stripped and collapsed before both tokenisation and alignment,
  matching our W&I train/dev build.
- English ERRANT/spaCy produces the M2 and defines the source token sequence.
- The M2 retains all inferred edits. For binary TSV labels we exclude `noop`,
  `UNK`, nonzero annotators, and every `M:PUNCT`, `R:PUNCT`, and `U:PUNCT` edit.
- `U:DET` and every other non-punctuation deletion are retained.
- Missing material is attached to the source token on the right. A missing item
  beyond sentence end has no source token and therefore creates no GED label.

This is our explicit W&I convention. It is not claimed to be a universal
MultiGED convention, and it differs from the unusually sparse English FCE gold.

## Install and build

Python 3.12 was used. In a fresh virtual environment:

```bash
python -m pip install -r requirements.txt
python build_wi_test.py \
  --corrupted /secure/test.corrupted.txt \
  --clean /secure/test.clean.txt \
  --out-prefix /secure/en_wi2024_test
```

Outputs:

- `en_wi2024_test.m2`: complete post-realignment ERRANT annotations;
- `en_wi2024_test.tsv`: private `TOKEN<TAB>c|i` GED gold;
- `en_wi2024_test.manifest.json`: input/output SHA-256 hashes, exact counts,
  policy, runtime versions, normalization counts, and ERRANT type totals.

Existing outputs are never replaced unless `--force` is supplied. Different
input line counts, blank paired sentences, invalid tokens, or M2/TSV token drift
stop the build.

## Validate and evaluate a submission

The submitted TSV must preserve every gold token and sentence boundary and
replace only the label column:

```bash
python validate_submission.py \
  --hyp /secure/submitted_predictions.tsv \
  --ref /secure/en_wi2024_test.tsv
python eval.py \
  -hyp /secure/submitted_predictions.tsv \
  -ref /secure/en_wi2024_test.tsv
```

`eval.py` is the original MultiGED-2023 evaluation script, unchanged, including
its familiar command-line interface and four-decimal output. The separate
validator first rejects truncation, extra sentences or tokens, token changes,
boundary changes, extra columns, and labels other than `c`/`i`. This prevents a
malformed file from exploiting the original scorer's permissive line-by-line
input handling without changing the official metric implementation.
