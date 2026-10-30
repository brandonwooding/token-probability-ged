# Decoder GED objectives (Families A and D)

Both families train on the synthetic `.jsonl` splits, which carry each
sentence's corrupted source, clean target and mERRANT edits.

## Family A: shape the decoder's probabilities (LoRA)

Every variant keeps the normal correction loss on the clean sentence. A second
pass shows the corrupted sentence in the same position and adds:

| Variant | Contrast | Unlikelihood | Keep |
|---|---:|---:|---:|
| `a0` (correction only) | 0 | 0 | 0 |
| `a1a` / `a1b` | 0.5 / 1.0 | 0 | 0 |
| `a2a` / `a2b` | 0 | 0.5 / 1.0 | 0.5 |
| `a3a` / `a3b` | 0.5 | 0.5 / 1.0 | 0.5 |

- **Contrast**: the error span should be less likely than its correction, by a margin of 1.
- **Unlikelihood**: push down the probability of erroneous words, `-log(1 - P)`.
- **Keep**: keep unchanged words likely.

A word's probability is the minimum over its sub-tokens, as at inference. The
paper's adapters are **`a2a`** (Correction + Unlikelihood + Preservation) on
`train20k`, with `a0` as the correction-only baseline. The objective sweep in the
paper's appendix used the older `ft200k__10000` runs of every variant.

```bash
python finetuning/decoder_ged/train_objectives.py --variant a2a \
    --model google/gemma-4-12B --corpus c4200 --langs en,de,es,ro,ru --name train20k
# or: qsub -v VARIANT=a2a,CORPUS=c4200,NAME=train20k finetuning/decoder_ged/train_a.pbs
```

Score with the decoder probability script (`--adapter models/<run>`); see
[`../README.md`](../README.md).

## Family D: linear GED head over decoder states (not in the paper)

A two-class head reads the hidden state after each word:

| Mode | Decoder | Trained |
|---|---|---|
| `d1` | base | head only |
| `d2` | fresh LoRA | LoRA + head |
| `d3a` / `d3b` | a Family-A correction LoRA | head / LoRA + head |

```bash
qsub -v MODE=d2 finetuning/decoder_ged/train_d.pbs
python finetuning/decoder_ged/predict_head.py --run finetuning/decoder_ged/models/<run> --dataset english/en_fce_dev
```

## Checkpoint selection

At each validation step, one threshold per language is chosen on the synthetic
`val5k` split, and the checkpoint with the best pooled F0.5 is kept. No learner
data is used. Training runs for up to three epochs.

Tests: `cd finetuning/decoder_ged && python -m unittest discover -s tests`.
