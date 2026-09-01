#!/usr/bin/env bash
#SBATCH --job-name=oracle_init
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
set -euo pipefail

ORACLE_ROOT="${ORACLE_MATCHING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${ORACLE_ROOT}/src"
CONFIG_PATH="${ORACLE_MATCHING_CONFIG:-${ORACLE_ROOT}/oracle_matching_config.yaml}"
PROJECT_HINT="$(cd "${ORACLE_ROOT}/../../../../../../.." && pwd)"
BOOTSTRAP_PYTHON="${PROJECT_HINT}/.venv/bin/python"

# Direct `sbatch oracle_matching.sbatch` remains supported. The canonical
# submit shell already exports the same frozen settings.
if [ -z "${ORACLE_LEROBOT_DATASET:-}" ]; then
  ORACLE_EXPORTS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/oracle_matching_config.py" \
      --config "${CONFIG_PATH}" --shell
  )"
  eval "${ORACLE_EXPORTS}"
fi

mkdir -p \
  "${ORACLE_ROOT}/../logs" \
  "${PROJECT_ROOT}/.cache/numba" \
  "${PROJECT_ROOT}/.cache/matplotlib"

source "${PROJECT_ROOT}/.venv/bin/activate"
export NUMBA_CACHE_DIR="${PROJECT_ROOT}/.cache/numba"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/tools/lerobot-libero${PYTHONPATH:+:${PYTHONPATH}}"

# A job-local LIBERO config avoids races with other jobs and contains no
# server-specific prefix: all paths are derived from global project_root.
LIBERO_SOURCE_ROOT="${PROJECT_ROOT}/tools/lerobot-libero/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
if [ ! -d "${LIBERO_BENCHMARK_ROOT}/init_files" ]; then
  echo "LIBERO init_files not found: ${LIBERO_BENCHMARK_ROOT}/init_files" >&2
  exit 1
fi
LIBERO_CONFIG_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
LIBERO_LOCAL_CONFIG_DIR="$(mktemp -d "${LIBERO_CONFIG_TMP_ROOT}/libero_cfg.XXXXXX")"
trap 'rm -rf "${LIBERO_LOCAL_CONFIG_DIR}"' EXIT
mkdir -p "${LIBERO_LOCAL_CONFIG_DIR}"
cat > "${LIBERO_LOCAL_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
datasets: ${LIBERO_SOURCE_ROOT}/datasets
assets: ${LIBERO_BENCHMARK_ROOT}/assets
EOF
export LIBERO_CONFIG_PATH="${LIBERO_LOCAL_CONFIG_DIR}"

"${ORACLE_ROOT}/run_oracle_matching.sh" --config "${CONFIG_PATH}"
