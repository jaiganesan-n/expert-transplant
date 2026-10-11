# Shared settings for the run scripts (sourced, not run).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Ops the MPS backend lacks fall back to the CPU instead of crashing (slower, but logged).
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
# Print every line at once, also when the output goes through `| tee` (Python would otherwise
# hold it back in 8 KB blocks, and a run looks silent for many minutes).
export PYTHONUNBUFFERED=1
BASE="--config configs/base.yaml"
# Keep the Mac awake for long runs (macOS `caffeinate`; ignored elsewhere).
if command -v caffeinate >/dev/null 2>&1; then KEEP_AWAKE="caffeinate -dims"; else KEEP_AWAKE=""; fi
