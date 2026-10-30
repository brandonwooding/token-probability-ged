"""The one place the two data.json shapes meet.

A threshold run's data.json exists in two shapes on disk:

  flat (legacy)  [ {word_id, probability, rank, entropy, percentile, ...}, ... ]
  nested         [ {word_id, token, subtokens: [ {probability, ...}, ... ]}, ... ]

Nested is what the threshold approaches write now. It keeps every model
sub-token that overlaps a dataset word, rather than collapsing them at write
time. Pooling itself is not a choice -- it is min probability per word, the same
rule everywhere -- but a *normaliser* applied at model-token level, before
pooling, needs the individual sub-tokens to exist. Collapsing on the GPU threw
them away and made any such normaliser a full re-run away.

flatten() turns either shape into the flat one, so every consumer is written
against a single record layout and keeps working on runs made before the change.

One property to know: a sub-token that straddles two dataset words (gemma's
'wasn' spanning gold 'was' and "n't", or '!!!!' spanning four gold '!') is
listed under BOTH words, so flatten() emits it once per word_id. That is what
min-pooling wants -- each word sees every piece of evidence overlapping it --
but it means the flat rows are not a partition of the sentence. Do not sum over
them to get a corpus statistic.
"""


def is_nested(sentence):
    """True if this sentence uses the nested (sub-token-preserving) shape."""
    return bool(sentence) and "subtokens" in sentence[0]


def flatten(sentence):
    """Either shape -> flat per-sub-token rows, each carrying its word_id.

    A legacy flat sentence passes through untouched, so callers can apply this
    unconditionally.
    """
    if not is_nested(sentence):
        return sentence
    return [
        {**subtoken, "word_id": word["word_id"]}
        for word in sentence
        for subtoken in word["subtokens"]
    ]
