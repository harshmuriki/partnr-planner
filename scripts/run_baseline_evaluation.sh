#!/usr/bin/env bash
# Launch from any directory; paths in configs are relative to the repository root.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# argparse uses the last --config if the caller supplies an override.
exec python scripts/run_tasks_wrapper.py \
  --config baseline_evaluation_v1/configs/baseline_runs.yaml "$@"
