# Encoder masked-LM objectives (Family E)

*Not part of the paper.*

The encoder counterpart of Family A in [`../decoder_ged/`](../decoder_ged/). It
fine-tunes the masked-LM head itself, so inference is unchanged from
`encoder-probability-thresh`: mask each sub-token in turn, take the minimum
probability per word, and threshold it.

| Variant | MLM | Slot contrast | Unlikelihood | Keep | Decoder analogue |
|---|---:|---:|---:|---:|---|
| `e0` | 1 | 0 | 0 | 0 | `a0` (in-domain MLM only) |
| `e1a` / `e1b` | 1 | 0.5 / 1.0 | 0 | 0 | `a1a` / `a1b` |
| `e2a` / `e2b` | 1 | 0 | 0.5 / 1.0 | 0.5 | `a2a` / `a2b` |
| `e3` | 1 | 0.5 | 0.5 | 0.5 | `a3a` |

```bash
qsub -v VARIANT=e2a finetuning/encoder_ged/train_e.pbs                      # mmBERT, C4_200M
qsub -v VARIANT=e2a,MODEL=FacebookAI/xlm-roberta-base finetuning/encoder_ged/train_e.pbs
qsub -v VARIANT=e2a,CORPUS=mlcheck,LANGS=en+de+it+sv finetuning/encoder_ged/train_e.pbs

qsub -v RUN="$PWD/finetuning/encoder_ged/models/<run>",MODEL_ID=e2a-mmbert finetuning/encoder_ged/predict_dev.pbs
python standard-eval/run.py --dataset english/en_fce_dev --tune-on english/en_c4200_eval5k
```

Checkpoints are selected as in Family A (per-language thresholds on synthetic
`val5k`, best pooled F0.5). Tests: `cd finetuning/encoder_ged && python -m unittest discover -s tests`.
