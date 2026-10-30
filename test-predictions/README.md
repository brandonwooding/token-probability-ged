# Test-set predictions

Builds the MultiGED-2023 (and W&I 2024) test submissions. Thresholds are chosen
on each corpus's labelled dev set and frozen in `thresholds.json`; test data is
never used to choose them.

## Systems

`thresholds.json` records, per system and per test corpus, the model, prompt,
threshold and the dev run it came from. The systems in the paper's main results table (MultiGED test sets):

| Paper row | System | Test macro F0.5 |
|---|---|---:|
| Instruction Prompted | `gemma-4-12B-it-instruction` | 48.87 |
| Next-token Probability | `gemma-base-fewshot` | 49.21 |
| C4_200M Adapted (dev thresholds) | `a2a20k` | 67.47 |
| MLCheck Adapted (dev thresholds) | `gemma-mlcheck-a2a-multiged` | 68.62 |
| MLCheck Adapted (one synthetic threshold, τ = 0.065) | `mlcheck-a2a-thr-mlcheck-en` | 68.23 |

Scores come from the MultiGED test gold, which is not public.
`test-set-results.txt` has the per-corpus TP/FP/FN, precision, recall and
F0.5 for every system above. `qwen-base-fewshot`,
`gemma-C4200-a2a20k-wi` (Write & Improve), `mlcheck-a2a-thr-c4200-en`,
`mlcheck-a2a-thr-perlang-mlcheck` and the `a2a`, `d1`, `d2*` systems are
extra comparisons not reported in the paper.

## Steps

```bash
# 1. test inputs (CPU): builds the *_test_unlabelled.sentences.json files
python test-predictions/prepare_test.py

# 2. test probabilities (GPU), e.g. for an adapted model
qsub -v ADAPTER="$PWD/finetuning/decoder_ged/models/<run>",TEMPLATE=fewshot decoder-probability-thresh/predict_a2a_test.pbs
qsub decoder-probability-thresh/predict_base_test.pbs      # base Gemma and Qwen
qsub GED-by-instruction/predict_test.pbs                   # instruction

# 3. submission TSVs (CPU), validated before they are written
python test-predictions/submission.py build-all --system gemma-mlcheck-a2a-multiged
python test-predictions/submission.py build-instruction
```

Outputs go to `test-predictions/submissions/<system>/` (gitignored). To freeze
thresholds for a new system, score its dev runs with `standard-eval/run.py`,
then run `python test-predictions/freeze_thresholds.py --system <name>`.

The W&I test input is built from the corpus's public originals; see
[`../datasets/write-and-improve/README.md`](../datasets/write-and-improve/README.md).
Test-set scores are in `test-set-results.txt`.
