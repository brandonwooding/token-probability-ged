# Token-probability GED

Code for **Minimally Supervised Multilingual Grammatical Error Detection**
(Brandon Wooding, Andrew Caines, Marek Rei).

A pre-trained decoder should find an erroneous token less likely than a correct
one. This repository turns that into a token-level error detector for five
languages: score each token of a sentence by its next-token probability, and
flag it as an error (`i`) when the probability falls below a threshold. The
signal is strengthened in two steps that use no human-annotated GED data:

1. **Few-shot prompting** puts four correction examples before the sentence.
2. **Adaptation** trains a LoRA on synthetic correction data with three losses:
   Correction, Unlikelihood on erroneous tokens, and Preservation of correct ones.

A single adapted Gemma 4 model with one threshold chosen on synthetic data
reaches 68.2 macro F0.5 on the MultiGED-2023 test sets, within 3 points of the
best supervised language-specific systems.

## Results

F0.5 on the held-out test sets of the six [MultiGED-2023](https://spraakbanken.github.io/multiged-2023/)
corpora ([Volodina et al., 2023](https://aclanthology.org/2023.nlp4call-1.1/)),
the shared task the evaluation data comes from. No system uses human-annotated
GED data for training.

| System | CS | FCE | REALEC | DE | IT | SV | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Instruction prompted | 46.58 | 49.84 | 36.99 | 50.04 | 55.02 | 54.74 | 48.87 |
| Next-token probability ¹ | 56.41 | 50.76 | 44.27 | 53.57 | 45.63 | 44.63 | 49.21 |
| C4_200M adapted ¹ | 71.31 | 64.54 | 53.74 | 76.32 | 68.17 | 70.73 | 67.47 |
| MLCheck adapted ¹ | 74.09 | 63.55 | 52.99 | 78.93 | 69.71 | 72.43 | **68.62** |
| MLCheck adapted ² | 73.46 | 63.14 | 52.10 | 78.86 | 69.80 | 71.99 | 68.23 |

All systems use Gemma 4 (12B): the instruction-tuned variant for instruction
prompting, and the base variant with the few-shot prompt for next-token
probability and adaptation.
¹ Per-corpus thresholds calibrated on the MultiGED development sets.
² One threshold for all corpora, calibrated on synthetic data.

Every development-set number in the paper, for every model, prompt and adapter,
is in [`standard-eval/results/`](standard-eval/results/).

## Paper → code

| Paper | Code |
|---|---|
| §3.1 Instruction-based detection | [`GED-by-instruction/`](GED-by-instruction/) (`ged_inference.py`) |
| §3.2 Next-token probability, prompt templates | [`decoder-probability-thresh/`](decoder-probability-thresh/) (`llm-decoder-threshold.py`, `prompts.py`) |
| §3.3 Adaptation, objective sweep | [`finetuning/decoder_ged/`](finetuning/decoder_ged/) (`train_objectives.py`, variant `a2a`) |
| §4 Corpora and synthetic data | [`datasets/`](datasets/) |
| Scoring and threshold calibration | [`standard-eval/`](standard-eval/) |
| Test-set results | [`test-predictions/`](test-predictions/) |
| Figures | [`paper-figures/`](paper-figures/) |

Every script takes a dataset id such as `english/en_fce_dev`, which
[`paths.py`](paths.py) resolves to files under `datasets/`. Each method writes its
raw output to `<method>/outputs/<run>/data.json`, and `standard-eval/run.py`
turns every run into `c`/`i` predictions and scores it the same way.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # versions as used for every result
cp .env.example .env                     # then add your HF_TOKEN
```

Results were produced with Python 3.12.3 on single A100/L40S/H100 GPUs. The
`.pbs` scripts record each cluster job: submit them from the repository root
(`qsub <dir>/<script>.pbs`), or run the `python` line inside them directly.
Scoring needs only the standard library and no GPU.

## Data

No corpus text is in this repository; [`datasets/README.md`](datasets/README.md)
explains how to install each one in the exact form used.

| Corpus | How |
|---|---|
| [MultiGED-2023](https://spraakbanken.github.io/multiged-2023/) (evaluation; [paper](https://aclanthology.org/2023.nlp4call-1.1/)) | Clone the public release and run `datasets/multiged-2023/prepare.py`; FCE's error-type M2 needs the BEA-2019 FCE download. |
| C4_200M synthetic (en, de, es, ro, ru) | Hugging Face dataset `brandonwooding/token-probability-ged`, or rebuild. |
| MLCheck synthetic (en, de, it, sv) | Download [Ubisoft's MultiLingualCheck](https://github.com/ubisoft/ubisoft-laforge-MultiLingualCheck-2M), then convert and annotate with the included builders. |

## Reproduce

Commands run from the repository root. Each step writes a run folder that
`standard-eval/run.py` picks up.

**Instruction-based detection** (§3.1, GPU, vLLM):

```bash
python GED-by-instruction/ged_inference.py --dataset english/en_fce_dev --model google/gemma-4-12B-it
```

**Next-token probability** (§3.2, GPU). Templates: `none`, `variant_a`,
`variant_b`, `fewshot`.

```bash
python decoder-probability-thresh/llm-decoder-threshold.py \
    --dataset english/en_fce_dev --model google/gemma-4-12B --template fewshot
```

**Adaptation** (§3.3, GPU; 16.5 h for C4_200M and 25 h for MLCheck on one A100):

```bash
hf download brandonwooding/token-probability-ged --repo-type dataset \
    --include "synthetic-data/*" --local-dir datasets          # C4_200M training data
python finetuning/decoder_ged/train_objectives.py --variant a2a --model google/gemma-4-12B \
    --corpus c4200 --langs en,de,es,ro,ru --name train20k       # MLCheck: --corpus mlcheck --langs en,de,it,sv
python decoder-probability-thresh/llm-decoder-threshold.py --dataset english/en_fce_dev \
    --model google/gemma-4-12B --template fewshot --adapter finetuning/decoder_ged/models/<run>
```

Or skip training and download the paper's adapters (Gemma 4, Qwen 3.5 and Tiny
Aya; `a2a`, plus the correction-only `a0`) under their original run names:

```bash
hf download brandonwooding/token-probability-ged --exclude "README.md" \
    --local-dir finetuning/decoder_ged/models
```

**Score** (CPU, standard library):

```bash
DEV="czech/cs_geccc_dev english/en_fce_dev english/en_realec_dev \
     german/de_falko-merlin_dev italian/it_merlin_dev swedish/sv_swell_dev"
python standard-eval/run.py --dataset $DEV                                     # oracle thresholds
python standard-eval/run.py --dataset $DEV --tune-on english/en_mlcheck_eval5k # synthetic threshold
```

A synthetic threshold needs a run of the same model and prompt on the synthetic
set too (`--dataset english/en_mlcheck_eval5k`).

**Test submissions** (the main results table). Thresholds are frozen from dev in
`test-predictions/thresholds.json`:

```bash
python test-predictions/prepare_test.py
python test-predictions/submission.py build-all --system gemma-mlcheck-a2a-multiged
```

**Figures**: in `paper-figures/`, `plot_token_probability_bars.py --unit 299
--words-below --legend-above` (token-probability example),
`plot_fce_combined_hist.py` (probability histograms) and
`plot_threshold_sweep.py` (threshold sensitivity). The prompt-context medians
are in [`paper-figures/gemma-probability-distributions.md`](paper-figures/gemma-probability-distributions.md).

Re-running vLLM inference can move F0.5 by up to ~0.9 points: temperature-0
decoding was not bit-reproducible across repeats. Re-scoring saved outputs is exact.

## Additional experiments (not in the paper)

Kept for completeness; they share the same data and scoring pipeline.

| Path | What |
|---|---|
| [`secondary/encoder-probability-thresh/`](secondary/encoder-probability-thresh/) | Masked-token probability from encoders (XLM-R, mmBERT), thresholded per token. |
| [`secondary/linear-probes/`](secondary/linear-probes/) | Linear probes over frozen hidden states, layer by layer. |
| [`secondary/confidence-classifiers/`](secondary/confidence-classifiers/) | Supervised classifier over encoder confidence features. |
| [`secondary/token-frequencies/`](secondary/token-frequencies/) | FineWeb unigram tables for frequency-normalised (SLOR) scores. |
| `finetuning/`: Family D, `train_decoder.py`, `train_encoder.py`, `encoder_ged/` | Decoder GED heads and encoder fine-tuning; see [`finetuning/README.md`](finetuning/README.md). |
| `datasets/write-and-improve/` | Write & Improve 2024 converted to MultiGED format, with CEFR levels. |

## Not in this repository

| Artefact | Where / how to get it back |
|---|---|
| Learner corpora (MultiGED) | Licensed; fetch and build as above. |
| C4_200M synthetic splits and source sentence pairs | Hugging Face dataset `brandonwooding/token-probability-ged`. |
| MLCheck synthetic splits | Rebuild from MLCheck (CC BY-NC-ND 4.0, not redistributable). |
| The paper's adapters | Hugging Face model `brandonwooding/token-probability-ged`. |
| Raw `outputs/` probability dumps, other checkpoints | Re-run or retrain with the commands above. |

## Citation

```bibtex
@misc{wooding-etal-minimally-supervised-ged,
  title  = {Minimally Supervised Multilingual Grammatical Error Detection},
  author = {Wooding, Brandon and Caines, Andrew and Rei, Marek},
  note   = {Under review}
}
```

## Licences

Code is MIT-licensed ([`LICENSE`](LICENSE)). Each corpus keeps its own licence
(see [`datasets/README.md`](datasets/README.md)); English FCE in particular is
under a non-transferable Cambridge licence and must never be committed here.
Adapters follow their base model: Gemma 4 and Qwen 3.5 adapters are Apache 2.0,
Tiny Aya adapters CC BY-NC 4.0.
