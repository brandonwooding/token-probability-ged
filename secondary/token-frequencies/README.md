# Unigram token frequencies

*Not part of the paper.*

Per-model, per-language unigram counts over web text, used to test whether
frequency-normalising token probabilities (SLOR, after
[jennhu/probability-grammaticality](https://github.com/jennhu/probability-grammaticality))
helps GED. It did not: the few-shot prompt context already provides that
normalisation.

The 50 tables (10 models × 5 languages) are tracked in `tables/`, so nothing
needs rebuilding to use them. Look one up with `paths.freqs(model_id, language)`.

| Language | Source (revisions pinned in `sources.py`) |
|---|---|
| English | `HuggingFaceFW/fineweb`, `sample-10BT` |
| Czech, German, Italian, Swedish | `HuggingFaceFW/fineweb-2` (`ces_Latn`, `deu_Latn`, `ita_Latn`, `swe_Latn`) |

## Rebuild

```bash
python secondary/token-frequencies/sample_fineweb.py --all --docs 150000   # network, ~40 min
python secondary/token-frequencies/count_tokens.py --all                   # CPU, ~45 min
```

The sample is a fixed stride over shards and row groups, recorded in
`samples/<language>.manifest.json`, so it is redrawn exactly. Counts are raw
(no smoothing) and exclude special tokens. Each table records its tokenizer's
`vocab_hash`; check it matches the model you use it with.
