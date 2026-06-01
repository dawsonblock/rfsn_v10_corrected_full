#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:-artifacts/proof/main8_1}
BASELINE_DIR=${2:-benchmarks/proof_baselines/main8_1}
ITERATIONS=${3:-3}

python3 scripts/generate_proof_artifacts.py \
  --output-dir "$OUTPUT_DIR" \
  --iterations "$ITERATIONS"

python3 scripts/check_proof_regression.py \
  --baseline-dir "$BASELINE_DIR" \
  --current-dir "$OUTPUT_DIR" \
  --strict-missing \
  --output-json "$OUTPUT_DIR/regression_report.json" \
  --output-md "$OUTPUT_DIR/regression_report.md"

echo "Proof regression gate passed: $OUTPUT_DIR vs $BASELINE_DIR"
