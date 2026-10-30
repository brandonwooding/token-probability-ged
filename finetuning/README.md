# Fine-tuning on synthetic data

Adapts models on the synthetic C4_200M or MLCheck data so their token
probabilities flag errors better. Learner annotations are never used for
training. All outputs go through `standard-eval` unchanged.

The paper's adaptation (§3.3) is **Family A** in [`decoder_ged/`](decoder_ged/):
a LoRA trained with Correction + Unlikelihood + Preservation (variant `a2a`),
with `a0` as the correction-only baseline and `a1a`–`a3b` as the objective sweep.

Not in the paper, kept for completeness:

| Where | What it trains | Inference |
|---|---|---|
| `decoder_ged/` Family D | linear GED head over decoder states | `predict_head.py` |
| `train_decoder.py` | an earlier correction-only decoder LoRA | `llm-decoder-threshold.py --adapter` |
| `train_encoder.py` | encoder token classifier (`c`/`i`) | `predict_encoder.py` |
| [`encoder_ged/`](encoder_ged/) | Family E: encoder masked-LM shaped like Family A | `predict_mlm.py` |

Training data is read from `datasets/synthetic-data/` (download or build it
first; see [its README](../datasets/synthetic-data/README.md)). `--corpus`
picks `c4200` (en, de, es, ro, ru) or `mlcheck` (en, de, it, sv); `--name` picks
the split (`train10k`, `train20k`, …). Checkpoints are written to
`<folder>/models/<run>/`, and an adapted run is named after its checkpoint.

## Typical run

```bash
# train (see decoder_ged/README.md for the variants)
python finetuning/decoder_ged/train_objectives.py --variant a2a --corpus c4200 --name train20k

# probabilities on learner dev + the synthetic tuning set, then score
python decoder-probability-thresh/llm-decoder-threshold.py --dataset english/en_fce_dev \
    --model google/gemma-4-12B --template fewshot --adapter finetuning/decoder_ged/models/<run>
python decoder-probability-thresh/llm-decoder-threshold.py --dataset english/en_c4200_eval5k \
    --model google/gemma-4-12B --template fewshot --adapter finetuning/decoder_ged/models/<run>
python standard-eval/run.py --dataset english/en_fce_dev --tune-on english/en_c4200_eval5k
```

On the cluster the same jobs are the `.pbs` scripts in each folder, e.g.
`qsub -v VARIANT=a2a,CORPUS=mlcheck,LANGS=en+de+it+sv,NAME=train20k finetuning/decoder_ged/train_a.pbs`
(join languages with `+`: `qsub -v` splits on commas).

## Note

Runs named `ft200k`, or made before 2026-08-24, used an older synthetic build
and are not comparable with the current data.

`push_to_hub.py --model <models/run> --repo <user>/<name>` uploads a checkpoint
(private by default).
