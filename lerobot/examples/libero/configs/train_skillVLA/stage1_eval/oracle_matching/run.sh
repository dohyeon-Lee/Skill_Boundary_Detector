#!/usr/bin/env bash
# Build the per-episode LIBERO init-state map (episode_index → init_state + scene/demo), by content-
# matching each filtered lerobot episode's action trajectory back to its original HDF5 demo. The map is
# FSQ-independent (depends only on the source dataset + original HDF5s) so it is written ONCE at the
# skillvla-dataset parent and shared by every FSQ_xx run under it. Stage-1 oracle eval reads it to reset
# the sim env to the exact scene of the episode whose GT skill sequence is being injected.
#
# Usage (first positional arg = source dataset; defaults to libero_90_full_full):
#   ./run.sh                                  # libero_90_full_full  →  libero_original_dataset/libero_90
#   ./run.sh libero_10_full_full              # reuse on another suite (orig suite auto-derived: libero_10)
#   ./run.sh libero_10_full_full --orig_dataset /abs/path/to/hdf5_dir   # override a derived path
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# project_root = 7 levels up (…/stage1_eval/oracle_matching → SBD). Use its .venv (h5py/pandas/numpy).
PROJECT_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
PY="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || { echo "[error] project venv python not found: ${PY}" >&2; exit 1; }

# Keep the dataset root consistent with every train/eval module.
GLOBAL_CONFIG="${PROJECT_ROOT}/lerobot/examples/libero/configs/global_config.yaml"
DATASET_ROOT_NAME="$("${PY}" - "${GLOBAL_CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent / "train_skills/src"))
from train_skills_config import load_config
print(load_config(sys.argv[1]).get("dataset_root", "dataset"))
PY
)"

SOURCE="${1:-libero_90_full_full}"; [ $# -gt 0 ] && shift   # rest of args forwarded to override defaults
# original LIBERO suite (libero_90 / libero_10 / …) auto-derived from the source name; override via --orig_dataset
SUITE="$(printf '%s' "${SOURCE}" | grep -oE 'libero_[0-9a-z]+' | head -n1 || echo libero_90)"

exec "${PY}" "${HERE}/build_init_states.py" \
  --lerobot_dataset "${PROJECT_ROOT}/${DATASET_ROOT_NAME}/${SOURCE}" \
  --orig_dataset    "${PROJECT_ROOT}/libero_original_dataset/${SUITE}" \
  --out             "${PROJECT_ROOT}/${DATASET_ROOT_NAME}/skillvla_dataset/${SOURCE}/eval_init_states.npz" \
  "$@"
