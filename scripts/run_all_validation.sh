#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-scripts/outputs}"

python scripts/run_validation_51_controls.py --output-root "${OUTPUT_ROOT}" --run-sanity
python scripts/run_validation_52_robustness.py --output-root "${OUTPUT_ROOT}"
python scripts/run_validation_53_interaction.py --output-root "${OUTPUT_ROOT}"
python scripts/run_validation_54_lambda_sensitivity.py --output-root "${OUTPUT_ROOT}"
python scripts/run_validation_55_diagnostics.py --output-root "${OUTPUT_ROOT}"
python scripts/run_validation_56_rank_stability.py --output-root "${OUTPUT_ROOT}"