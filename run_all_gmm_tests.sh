#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${1:-gmm_proxy_segmentation.py}"
TEST_DATA_DIR="${2:-gmm_test_data_pack}"
OUTPUT_ROOT="${3:-batch_run_outputs}"

python run_all_gmm_tests.py   --script "$SCRIPT_PATH"   --test-data-dir "$TEST_DATA_DIR"   --output-root "$OUTPUT_ROOT"   --fit-on logit   --bootstrap 0   --timeout-seconds 120
