#!/bin/bash
# Train the full grid of confidence classifiers: 2 classifier types x 4 attention
# metrics = 8 comparable systems, all fit on --train and predicting on --eval.
#
# CPU-only sklearn -- runs on the login node; no qsub. The GPU work is generating
# the encoder feature runs (prob/rank/entropy data.json + attention.json) for the
# train and eval splits; this consumes whatever the latest matching runs are.
#
# Usage (from this directory, with the venv active):
#   module load Python/3.12.3-GCCcore-13.3.0
#   source ../../.venv/bin/activate     # adjust to your .venv location
#   ./submit_confidence.sh
#   ./submit_confidence.sh --train english/en_fce_train --eval english/en_fce_dev
#
# Any extra args are forwarded to every run (e.g. --model mmBERT-base --seed 0).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METRICS="token_mean word_vote word_vote_renorm word_vote_zeroed"
CLASSIFIERS="mlp logreg"

for CLF in $CLASSIFIERS; do
    for ATTN in $METRICS; do
        echo "=== clf=$CLF attn=$ATTN ==="
        python "$SCRIPT_DIR/train_confidence_clf.py" --clf "$CLF" --attn "$ATTN" "$@"
    done
done
