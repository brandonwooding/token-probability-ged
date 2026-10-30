#!/bin/bash
# Submit one run_ged.pbs job per model, all over a single dataset.
#
# Usage: ./submit_ged.sh <dataset-id> <model> [model ...]
#   e.g. ./submit_ged.sh german/de_falko-merlin_dev \
#            swiss-ai/Apertus-8B-Instruct-2509 Qwen/Qwen3.5-9B
#
# One dataset per invocation, by design: each job loads its own vLLM engine, so
# the unit of work is (dataset, model) and the dataset is what varies least
# often. A dataset id is '<language>/<stem>' or a bare '<stem>' (see paths.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <dataset-id> <model> [model ...]" >&2
    echo "  e.g. $0 english/en_fce_dev google/gemma-4-12B-it" >&2
    exit 1
fi

DATASET="$1"
shift

for MODEL in "$@"; do
    qsub -v MODEL="$MODEL",DATASET="$DATASET" "$SCRIPT_DIR/run_ged.pbs"
done
