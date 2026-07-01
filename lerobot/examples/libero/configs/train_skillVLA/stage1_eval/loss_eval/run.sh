#!/usr/bin/env bash
# Offline front-vs-back SKILL loss eval (no sim): for each model in loss_eval_config.yaml's `models`,
# run the flow-matching forward on its dataset and break the per-chunk action MSE down by the chunk's
# within-skill ENDPOINT progress (0-50 / 50-90 / 90-100) — recovering the `action_loss_prog/*` breakdown
# for runs that finished before that wandb panel existed. Prints a per-model table; writes outputs/*.json.
# Needs a GPU in the current shell (run inside salloc, or use ./submit.sh for a Slurm job).
#
#   ./run.sh                       # loss_eval_config.yaml (models + n_batches/batch_size/seed)
#   ./run.sh --n_batches 100       # CLI overrides the yaml sampling knobs
#   ./run.sh --config /path.yaml --batch_size 32
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# project_root = 7 levels up (…/stage1_eval/loss_eval → SBD). Uses SBD/.venv (torch + lerobot + matplotlib,
# same venv the sim eval runs under).
PROJECT_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
PY="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || { echo "[error] project venv python not found: ${PY}" >&2; exit 1; }

export PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/lerobot/examples/libero:${HERE}/../src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH 2>/dev/null || true   # avoid system cuDNN shadowing the bundled one
exec "${PY}" "${HERE}/loss_eval.py" "$@"
