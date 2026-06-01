#!/usr/bin/env bash
set -euo pipefail

PROFILE=${4:-main10}
OUTPUT_DIR=${1:-artifacts/proof/$PROFILE}
BASELINE_DIR=${2:-benchmarks/proof_baselines/$PROFILE}
ITERATIONS=${3:-3}

python3 scripts/generate_proof_artifacts.py \
  --profile "$PROFILE" \
  --output-dir "$OUTPUT_DIR" \
  --iterations "$ITERATIONS"

python3 scripts/generate_plots.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir results/plots

python3 scripts/check_proof_regression.py \
  --profile "$PROFILE" \
  --baseline-dir "$BASELINE_DIR" \
  --current-dir "$OUTPUT_DIR" \
  --strict-missing \
  --output-json "$OUTPUT_DIR/regression_report.json" \
  --output-md "$OUTPUT_DIR/regression_report.md"

echo "Proof regression gate passed: $OUTPUT_DIR vs $BASELINE_DIR"
