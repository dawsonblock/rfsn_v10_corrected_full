#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:-artifacts/proof/main8_1}
ITERATIONS=${2:-3}

python3 scripts/generate_proof_artifacts.py \
  --output-dir "$OUTPUT_DIR" \
  --iterations "$ITERATIONS"

echo "Proof artifacts written to: $OUTPUT_DIR"
