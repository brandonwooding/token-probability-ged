"""Which web corpus each MultiGED language draws its unigram frequencies from.

FineWeb-2 is the multilingual corpus, but it is deliberately the *complement*
of FineWeb: 1,870 language configs and not one of them is English. So English
comes from FineWeb's own `sample-10BT` instead. That is not a workaround -- it
is the same source Hu et al. used for their SLOR frequencies, so English is a
straight replication of their setup and the other four languages extend it.

Revisions are pinned. A frequency table is only interpretable against the
corpus it was counted from, and `main` moves.
"""

# Pinned 2026-09-02 via HfApi().dataset_info(...).sha
FINEWEB_2 = "HuggingFaceFW/fineweb-2"
FINEWEB_2_REV = "af9c13333eb981300149d5ca60a8e9d659b276b9"
FINEWEB = "HuggingFaceFW/fineweb"
FINEWEB_REV = "9bb295ddab0e05d785b879661af7260fed5140fc"

# language folder name (matches datasets/multiged-2023/<language>/) ->
# (repo, revision, shard path prefix). The language names are the ones paths.py
# already resolves, so nothing downstream learns a second naming scheme.
SOURCES = {
    "czech":   (FINEWEB_2, FINEWEB_2_REV, "data/ces_Latn/train/"),
    "english": (FINEWEB,   FINEWEB_REV,   "sample/10BT/"),
    "german":  (FINEWEB_2, FINEWEB_2_REV, "data/deu_Latn/train/"),
    "italian": (FINEWEB_2, FINEWEB_2_REV, "data/ita_Latn/train/"),
    "swedish": (FINEWEB_2, FINEWEB_2_REV, "data/swe_Latn/train/"),
}

LANGUAGES = tuple(SOURCES)

# One table per model, even where two models share a vocabulary. gemma-4-12B and
# gemma-4-12B-it hash identically, as do Qwen3.5-9B and its Base -- so four of
# these files are byte-for-byte duplicates of another four. That is intentional:
# a run names a model, and a lookup that has to know which models are secretly
# the same model is a lookup that will eventually get it wrong.
MODELS = (
    "swiss-ai/Apertus-8B-Instruct-2509",
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-12B-it",
    "CohereLabs/tiny-aya-global",
    "swiss-ai/Apertus-8B-2509",
    "Qwen/Qwen3.5-9B-Base",
    "google/gemma-4-12B",
    "CohereLabs/tiny-aya-base",
    "jhu-clsp/mmBERT-base",
    "FacebookAI/xlm-roberta-base",
)


def model_id(model):
    """'google/gemma-4-12B' -> 'gemma-4-12B'. Same rule the run folders use."""
    return model.split("/")[-1]
