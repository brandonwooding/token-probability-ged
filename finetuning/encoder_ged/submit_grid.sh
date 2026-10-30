#!/bin/bash
# Family-E training grid plus its matched supervised baseline:
#   6 variants x 2 corpora x 2 encoders = 24 Family-E runs
#   1 ft-encoder x 2 corpora x 2 encoders =  4 baseline runs
#                                          -----------------
#                                           28 runs
#
# The ft-encoder is the token classifier from ../train_encoder.py, trained on
# the SAME train20k/val5k cuts as Family E. That is what makes it a bar Family
# E has to clear: the existing ft100k checkpoint was trained on different data
# and its cross-tuned number is a known pre-rebuild artefact, so comparing
# against it would confound the objective with the training set.
#
# Prints the qsub lines by default. Pass --submit to actually queue them.
#
#   ./submit_grid.sh                 # dry run
#   ./submit_grid.sh --submit        # queue all 24
#   VARIANTS="e0 e2a" ./submit_grid.sh --submit    # a subset
#
# Language sets match the decoder's a0/a2a runs so the two families stay
# comparable. mlcheck also has ru and es on disk at train20k/val5k, but the
# decoder never trained them there, so they are left out by default -- add them
# with MLCHECK_LANGS if you want the wider set and accept the asymmetry.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

VARIANTS="${VARIANTS:-e0 e1a e1b e2a e2b e3}"
WITH_BASELINE="${WITH_BASELINE:-1}"   # also queue the 4 ft-encoder runs
MODELS="${MODELS:-jhu-clsp/mmBERT-base FacebookAI/xlm-roberta-base}"
C4200_LANGS="${C4200_LANGS:-en+de+es+ro+ru}"
MLCHECK_LANGS="${MLCHECK_LANGS:-en+de+it+sv}"
NAME="${NAME:-train20k}"
VAL_NAME="${VAL_NAME:-val5k}"
TOKEN_BUDGET="${TOKEN_BUDGET:-16384}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
LR="${LR:-1e-5}"

submit=0
[[ "${1:-}" == "--submit" ]] && submit=1

count=0
for corpus in c4200 mlcheck; do
  case "$corpus" in
    c4200)   langs="$C4200_LANGS" ;;
    mlcheck) langs="$MLCHECK_LANGS" ;;
  esac
  for model in $MODELS; do
    for variant in $VARIANTS; do
      count=$((count + 1))
      args="VARIANT=$variant,MODEL=$model,CORPUS=$corpus,LANGS=$langs"
      args="$args,NAME=$NAME,VAL_NAME=$VAL_NAME"
      args="$args,TOKEN_BUDGET=$TOKEN_BUDGET,GRAD_ACCUM=$GRAD_ACCUM,LR=$LR"
      if [[ $submit -eq 1 ]]; then
        qsub -v "$args" "$HERE/train_e.pbs"
      else
        printf '%2d  qsub -v %s %s/train_e.pbs\n' "$count" "$args" "$HERE"
      fi
    done

    if [[ "$WITH_BASELINE" == "1" ]]; then
      count=$((count + 1))
      bargs="MODEL=$model,CORPUS=$corpus,LANGS=$langs"
      bargs="$bargs,NAME=$NAME,VAL_NAME=$VAL_NAME,MAX_PER_LANG=all"
      if [[ $submit -eq 1 ]]; then
        qsub -v "$bargs" "$HERE/../train_encoder.pbs"
      else
        printf '%2d  qsub -v %s %s/../train_encoder.pbs   [ft-encoder baseline]\n' \
          "$count" "$bargs" "$HERE"
      fi
    fi
  done
done
echo "# $count runs ($( [[ $submit -eq 1 ]] && echo submitted || echo 'dry run -- pass --submit to queue'))"
