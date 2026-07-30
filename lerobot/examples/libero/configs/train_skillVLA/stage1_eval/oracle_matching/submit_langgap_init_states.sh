#!/usr/bin/env bash
# Submit LangGap episode-exact init-state reconstruction on one EGL-capable GPU.
# Usage: ./submit_langgap_init_states.sh [source] [builder options...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
PY="${PROJECT_ROOT}/.venv/bin/python"
SOURCE="${1:-langgap_56_full_firsthalf}"
[ $# -eq 0 ] || shift

mapfile -t SLURM_SETTINGS < <("${PY}" - "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "lerobot/examples/libero/configs/train_skills/src"))
from train_skills_config import as_list, load_config
cfg = load_config(root / "lerobot/examples/libero/configs/global_config.yaml")
print(",".join(as_list(cfg.get("train_partition", []))))
print(str(cfg.get("train_qos", "base_qos")))
print(str(cfg.get("train_nodelist", "")))
print(",".join(as_list(cfg.get("train_exclude_nodes", []))))
PY
)
PARTITION="${SLURM_SETTINGS[0]}"
QOS="${SLURM_SETTINGS[1]}"
NODELIST="${SLURM_SETTINGS[2]}"
EXCLUDE="${SLURM_SETTINGS[3]}"

SBATCH_ARGS=(--partition="${PARTITION}" --qos="${QOS}")
[ -z "${NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${NODELIST}")
[ -z "${EXCLUDE}" ] || SBATCH_ARGS+=(--exclude="${EXCLUDE}")

cd "${HERE}"
mkdir -p ../logs
sbatch "${SBATCH_ARGS[@]}" "${HERE}/build_langgap_init_states.sbatch" "${SOURCE}" "$@"
