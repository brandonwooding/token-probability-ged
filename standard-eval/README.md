# standard-eval

One detection-only evaluation for every approach, so all numbers are directly
comparable. Every token is labelled `c`/`i`; the positive class is `i`.

## How it works

1. **Gold** (`build_gold.py`): the dataset TSV gives the labels; its optional
   `.m2` adds an error type to each `i` token (for per-type recall).
2. **Adapt** (`adapt_thresh.py`, `adapt_instruction.py`, `adapt_linear_probes.py`):
   turn an approach's raw `data.json` into `c`/`i` predictions.
3. **Score** (`score.py`): precision, recall, F1, F0.5, and recall per error type.

A new dataset needs only its TSV; a new approach needs only an adapter.

## Run

```bash
python standard-eval/run.py --dataset english/en_fce_dev czech/cs_geccc_dev
python standard-eval/run.py --dataset english/en_fce_dev --tune-on english/en_c4200_eval5k
```

`run.py` builds any missing `.sentences.json` / `.gold.json`, finds every run in
`<approach>/outputs/<stem>__*/`, adapts and scores it, writes
`results/<approach>__<run>.json`, and prints a ranked table. `--rebuild` forces
the gold files to be regenerated. The scorer and the threshold/instruction
adapters are standard library only.

## Things to know before quoting a number

- **Thresholds.** Probability approaches call a word `i` when its probability
  (the minimum over its sub-tokens) falls below a threshold. By default the
  threshold is the one that maximises F0.5 on the set being scored (an
  optimistic oracle). `--tune-on <dataset>` picks it on another set, normally
  synthetic, and applies it here; this needs a run of the same model and prompt
  on that set. The two are written to separate result files and should not be
  mixed in one table.
- **Labels come from MultiGED as distributed.** The M2 only supplies types. On
  FCE it flags ~966 extra tokens (deletions, missing punctuation) that stay `c`.
- **Per-type scores are recall only**, since a false positive has no gold type.
- **Instruction runs** whose output failed to parse are scored as all `c`.
