#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${1:-gmm_proxy_segmentation.py}"
TEST_DATA_DIR="${2:-gmm_test_data_pack}"
OUTPUT_ROOT="${3:-batch_run_outputs}"
shift $(( $# >= 3 ? 3 : $# ))

PYTHON_BIN="${PYTHON_BIN:-python}"
FIT_ON="${FIT_ON:-logit}"
BOOTSTRAP="${BOOTSTRAP:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
MIN_COMPONENTS="${MIN_COMPONENTS:-3}"
MAX_COMPONENTS="${MAX_COMPONENTS:-3}"
GRID_POINTS="${GRID_POINTS:-}"
SCORE_COL="${SCORE_COL:-pred_prob}"
BUY_COL="${BUY_COL:-buy}"
GLOB_PATTERN="${GLOB_PATTERN:-*.csv}"

# Legacy extras: the Python runner will forward them only if the target script supports them.
FALLBACK_LOW="${FALLBACK_LOW:-}"
FALLBACK_HIGH="${FALLBACK_HIGH:-}"
MIN_THRESHOLD_GAP="${MIN_THRESHOLD_GAP:-}"

cmd=(
  "$PYTHON_BIN" run_all_gmm_tests.py
  --script "$SCRIPT_PATH"
  --test-data-dir "$TEST_DATA_DIR"
  --output-root "$OUTPUT_ROOT"
  --score-col "$SCORE_COL"
  --buy-col "$BUY_COL"
  --fit-on "$FIT_ON"
  --bootstrap "$BOOTSTRAP"
  --glob "$GLOB_PATTERN"
  --timeout-seconds "$TIMEOUT_SECONDS"
)

if [[ -n "$MIN_COMPONENTS" ]]; then
  cmd+=(--min-components "$MIN_COMPONENTS")
fi
if [[ -n "$MAX_COMPONENTS" ]]; then
  cmd+=(--max-components "$MAX_COMPONENTS")
fi
if [[ -n "$GRID_POINTS" ]]; then
  cmd+=(--grid-points "$GRID_POINTS")
fi
if [[ -n "$FALLBACK_LOW" ]]; then
  cmd+=(--fallback-low "$FALLBACK_LOW")
fi
if [[ -n "$FALLBACK_HIGH" ]]; then
  cmd+=(--fallback-high "$FALLBACK_HIGH")
fi
if [[ -n "$MIN_THRESHOLD_GAP" ]]; then
  cmd+=(--min-threshold-gap "$MIN_THRESHOLD_GAP")
fi

# Forward any extra CLI args the user passes to this wrapper.
cmd+=("$@")

printf 'Running batch test wrapper with command:
  %q' "${cmd[0]}"
for ((i=1; i<${#cmd[@]}; i++)); do
  printf ' %q' "${cmd[i]}"
done
printf '
'

"${cmd[@]}"
