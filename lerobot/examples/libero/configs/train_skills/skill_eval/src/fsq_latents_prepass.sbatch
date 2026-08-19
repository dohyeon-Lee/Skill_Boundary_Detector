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

# Resolve every checkpoint first, then encode them in ONE process: reading the
# 11k-file skillset dominates a single-checkpoint run (~40s of encoding against
# many minutes of loading), so it must be paid once per run, not once per
# checkpoint.
PLAN_FILE="$(mktemp "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/fsq_latents_plan.XXXXXX.json")"
trap 'rm -f "${PLAN_FILE}"' EXIT
SKILLS_DIR=""
PLAN_ENTRIES=()
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
  SKILLS_DIR="${FSQ_SKILLS_DIR}"
  PLAN_ENTRIES+=("{\"model_path\": \"${FSQ_MODEL_PATH}\", \"output_path\": \"${FSQ_LATENTS_PATH}\"}")
  echo "[${FSQ_EPOCH_TAG}] queued -> ${FSQ_LATENTS_PATH}"
done

if [ "${#PLAN_ENTRIES[@]}" -eq 0 ]; then
  echo "Nothing to encode; every requested checkpoint already has latents."
else
  printf '[%s]\n' "$(IFS=,; echo "${PLAN_ENTRIES[*]}")" > "${PLAN_FILE}"
  echo "Encoding ${#PLAN_ENTRIES[@]} checkpoint(s) in one process"
  # encode_FSQ_skills.py writes each output atomically and skips ones that
  # already exist, so a requeued job resumes where it stopped.
  PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero" \
    "${PROJECT_ROOT}/.venv/bin/python" \
      "${LEROBOT_ROOT}/examples/libero/encode_FSQ_skills.py" \
      --skills-dir "${SKILLS_DIR}" \
      --plan-path "${PLAN_FILE}" \
      --device cuda
fi

echo "DONE FSQ LATENTS PREPASS: ${FSQ_GT_REPLAY_RUN}"
