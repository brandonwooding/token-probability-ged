"""Word-level attention features from a single token-level attention matrix.

Pure numpy: no torch, no file I/O, no model. Everything that decides *what the
feature means* lives here, so the variants can be changed without touching the
GPU pass (extract-attention.py) or re-running the model.

BACKGROUND -- why "mean attention" has to be pinned down
--------------------------------------------------------
`A[i, j]` is how much query token i attends to key token j. Rows are softmax
outputs, so every row sums to 1. That makes the *outgoing* mean degenerate: the
mean of a token's row is 1/T for every token, a function of sentence length and
nothing else. The informative direction is the *column* -- how much the rest of
the sentence attends to this token. Because rows sum to 1, column sums over the
whole matrix total exactly T, so a column sum is already scale-free: 1.0 means
"this token receives an average share of the sentence's attention", in a 5-token
sentence or a 100-token one. No extra length normalisation is applied anywhere
below; it comes for free.

Going from tokens to words needs a choice on each axis, and they are not the
same choice:

  * ROWS are normalised distributions (each sums to 1), so they are averaged.
    A word's opinion about salience is the mean of its sub-tokens' opinions --
    dividing by k stops a 3-piece word casting 3 votes on what matters.
  * COLUMNS are unnormalised mass. A word's k columns are k distinct places
    attention landed, and the word received all of it, so they are summed.

Average what is normalised, sum what is not.

THE FOUR SCORES
---------------
All four start from the same head-averaged matrix and differ only in how much
correction is applied. They are deliberately on different scales (see "totals"
below); a threshold tuned on one does not transfer to another.

  token_mean        No row treatment at all: plain token column sums a_j =
                    sum_i A[i, j] over EVERY row (special-token rows included),
                    then the mean over the word's sub-tokens. The most literal
                    reading of "mean attention on this word". Fragmented words
                    still vote k times.

  word_vote         Rows summed per word and divided by k (one vote per word),
                    then columns summed per word. Within-word blocks kept,
                    no renormalisation -- so the mass a word sent to [CLS]/[SEP]
                    is simply lost and rows sum to less than 1.

  word_vote_renorm  As word_vote, then each row rescaled to sum to 1 over the
                    word columns. This recovers the sink mass, redistributing it
                    proportionally over the real words. Self-attention kept.

  word_vote_zeroed  As word_vote, then the within-word block zeroed (for a
                    1-piece word that is just the diagonal, so self-attention is
                    removed for every word), then rows rescaled to 1. "How much
                    does the REST of the sentence attend to this word."

Per-sentence totals: word_vote_zeroed and word_vote_renorm each total exactly W
(every row is a distribution over W words); word_vote totals W minus the sink
share; token_mean totals whatever it totals.

DEGENERATE SENTENCES
--------------------
With one aligned word, word_vote_zeroed leaves an all-zero row -- there is no
"rest of the sentence" -- so the score is 0.0 (the row is left at zero rather
than divided by its zero mass). word_vote_renorm puts the whole row back on the
only word, giving 1.0. With two aligned words, zeroing forces each row entirely
onto the other word, so word_vote_zeroed gives exactly 1.0 to both; the other
three are unconstrained and will vary.

FCE-dev has 18 one-word and 79 two-word sentences, so these are live cases.
"""

import numpy as np

# Order matters: it is the column order everything downstream should assume.
SCORE_NAMES = ("token_mean", "word_vote", "word_vote_renorm", "word_vote_zeroed")


def group_positions(word_ids):
    """word_ids -> (sorted present word ids, {word_id: [token positions]}).

    Tokens with word_id None -- special tokens, and the handful of sub-tokens
    that overlap no gold word -- are excluded from both axes: they are not
    words, so they cannot be a row or a column of the word-level matrix. They
    still contribute rows to `token_mean`, which is defined on the raw matrix.

    A gold word the tokenizer merged into a neighbour owns no sub-token and so
    never appears here; recovering it is the caller's job (the same forward-fill
    the probability features use).
    """
    groups = {}
    for position, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        groups.setdefault(word_id, []).append(position)
    return sorted(groups), groups


def _renormalise_rows(matrix):
    """Rescale each row to sum to 1, leaving all-zero rows at zero.

    An all-zero row is a word with nowhere left to attend (the one-word sentence
    under within-word zeroing). It casts no vote rather than raising a division
    by zero.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    out = np.zeros_like(matrix)
    nonzero = row_sums[:, 0] > 0
    out[nonzero] = matrix[nonzero] / row_sums[nonzero]
    return out


def word_scores(attention, word_ids):
    """Per-word attention scores from one head-averaged (T, T) matrix.

    attention: (T, T) float array, rows summing to 1 (raw softmax output).
    word_ids:  length-T list of int or None, as produced by get_custom_word_ids.

    Returns a dict with 'word_ids' (the present ids, ascending), 'k' (sub-tokens
    per word), 'self_share', and one list per name in SCORE_NAMES -- all aligned
    to 'word_ids'. Empty lists if no token aligned to a word.

    'self_share' is the diagonal of the word-level matrix before any zeroing:
    the fraction of a word's own outgoing attention that lands back on itself.
    It is a diagnostic, not a score -- it is what word_vote_zeroed removes, so
    it lets you check afterwards whether a result is driven by self-attention,
    without needing to re-run anything.
    """
    attention = np.asarray(attention, dtype=np.float64)
    present, groups = group_positions(word_ids)

    if not present:
        return {"word_ids": [], "k": [], "self_share": [],
                **{name: [] for name in SCORE_NAMES}}

    # -- token_mean: no row treatment, so every row of the raw matrix votes at
    # full weight, special-token rows included. Column sums first, then the mean
    # over each word's sub-tokens.
    column_sums = attention.sum(axis=0)
    token_mean = np.array([column_sums[groups[w]].mean() for w in present])

    # -- the word-level matrix, built in two independent steps.
    #
    # Rows: (T, T) -> (W, T). Word w's sub-token rows are summed and divided by
    # k_w. Word w now holds one unit of outgoing attention however many pieces
    # it was split into. The column axis is untouched at this point -- word j
    # still occupies its k_j columns.
    rows = np.stack([
        attention[groups[w], :].sum(axis=0) / len(groups[w]) for w in present
    ])

    # Columns: (W, T) -> (W, W). Only now does each word collapse to a single
    # column, by summing the mass that landed on any of its pieces.
    matrix = np.stack([rows[:, groups[w]].sum(axis=1) for w in present], axis=1)

    # word_vote: column sums as they stand. Rows sum to less than 1 because the
    # mass that went to [CLS]/[SEP] and to unaligned sub-tokens was dropped with
    # those columns and is not given back.
    word_vote = matrix.sum(axis=0)

    # word_vote_renorm: same matrix, rows rescaled to 1, which hands the sink
    # mass back to the real words in proportion to what they already received.
    word_vote_renorm = _renormalise_rows(matrix).sum(axis=0)

    # word_vote_zeroed: matrix[w, w] aggregates exactly the within-word block of
    # the token matrix, so zeroing the diagonal here is equivalent to zeroing
    # A[i, j] wherever word(i) == word(j) before aggregating -- the row scaling
    # in between is a per-row scalar and commutes with it.
    zeroed = matrix.copy()
    np.fill_diagonal(zeroed, 0.0)
    word_vote_zeroed = _renormalise_rows(zeroed).sum(axis=0)

    return {
        "word_ids": [int(w) for w in present],
        "k": [len(groups[w]) for w in present],
        "self_share": np.diag(matrix).tolist(),
        "token_mean": token_mean.tolist(),
        "word_vote": word_vote.tolist(),
        "word_vote_renorm": word_vote_renorm.tolist(),
        "word_vote_zeroed": word_vote_zeroed.tolist(),
    }
