#!/usr/bin/env bash
# Encode the FSQ skill latents that a replay run is still missing.
#
# Encoding is the only GPU work in the GT replay pipeline: the replay itself
# just decodes dataset video and writes PNGs on CPU. Doing it here, once per
# run, lets the replay array run entirely GPU-free -- so a three-model
# comparison needs three GPUs instead of one per checkpoint.
#
# Env:
#   FSQ_GT_REPLAY_DIR          skill_eval directory (submit dir)
#   FSQ_GT_REPLAY_CONFIG       snapshotted config path
#   FSQ_GT_REPLAY_RUN          run name to encode for
#   FSQ_LATENTS_CHECKPOINTS    space-separated checkpoints to encode
#SBATCH --job-name=FSQ_LAT
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${FSQ_GT_REPLAY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FSQ_GT_REPLAY_CONFIG:-${SUBMIT_DIR}/fsq_gt_replay_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

: "${FSQ_GT_REPLAY_RUN:?fsq_latents_prepass needs FSQ_GT_REPLAY_RUN}"
: "${FSQ_LATENTS_CHECKPOINTS:?fsq_latents_prepass needs FSQ_LATENTS_CHECKPOINTS}"
read -r -a CHECKPOINTS <<< "${FSQ_LATENTS_CHECKPOINTS}"

echo "START FSQ LATENTS PREPASS"
echo "node       : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "run        : ${FSQ_GT_REPLAY_RUN}"
echo "checkpoints: ${FSQ_LATENTS_CHECKPOINTS}"
nvidia-smi || true

FIRST_SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
    --config "${CONFIG_PATH}" --run-name "${FSQ_GT_REPLAY_RUN}" \
    --checkpoint "${CHECKPOINTS[0]}" --shell
)"
eval "${FIRST_SETTINGS}"
cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

# Encoding on CPU takes roughly an hour per checkpoint, so a node whose CUDA is
# broken must requeue rather than silently fall back.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  SETTINGS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
      --config "${CONFIG_PATH}" --run-name "${FSQ_GT_REPLAY_RUN}" \
      --checkpoint "${CHECKPOINT}" --shell
  )"
  eval "${SETTINGS}"
  if [ -f "${FSQ_LATENTS_PATH}" ]; then
    echo "[${FSQ_EPOCH_TAG}] latents already present; skipping"
    continue
  fi
  # The same lock the replay path uses, so a concurrently submitted replay job
  # never encodes the same checkpoint twice.
  LATENTS_LOCK="${FSQ_LATENTS_PATH}.lock"
  (
    flock 9
    if [ -f "${FSQ_LATENTS_PATH}" ]; then
      echo "[${FSQ_EPOCH_TAG}] latents appeared while waiting for the lock; skipping"
      exit 0
    fi
    TEMP_LATENTS="${FSQ_LATENTS_PATH%.npz}.tmp.${SLURM_JOB_ID:-$$}.npz"
    echo "[${FSQ_EPOCH_TAG}] encoding -> ${FSQ_LATENTS_PATH}"
    PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero" \
      "${PROJECT_ROOT}/.venv/bin/python" \
        "${LEROBOT_ROOT}/examples/libero/encode_FSQ_skills.py" \
        --model-path "${FSQ_MODEL_PATH}" \
        --skills-dir "${FSQ_SKILLS_DIR}" \
        --output-path "${TEMP_LATENTS}" \
        --device cuda
    mv "${TEMP_LATENTS}" "${FSQ_LATENTS_PATH}"
  ) 9>"${LATENTS_LOCK}"
done

echo "DONE FSQ LATENTS PREPASS: ${FSQ_GT_REPLAY_RUN}"
