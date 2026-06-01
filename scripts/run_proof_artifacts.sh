#!/usr/bin/env bash
set -euo pipefail

PROFILE=${3:-main8_1}
OUTPUT_DIR=${1:-artifacts/proof/$PROFILE}
ITERATIONS=${2:-3}

python3 scripts/generate_proof_artifacts.py \
  --output-dir "$OUTPUT_DIR" \
  --iterations "$ITERATIONS"

python3 scripts/generate_plots.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir results/plots

echo "Proof artifacts written to: $OUTPUT_DIR"
