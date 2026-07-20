#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${PRETRAIN_CONFIG:-${SCRIPT_DIR}/pretrain_config.yaml}"
PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3
eval "$("${PYTHON}" "${SCRIPT_DIR}/src/pretrain_config.py" --config "${CONFIG_PATH}" --shell)"

if [ -e "${FAST_TOKENIZER}" ] && [ -f "${PRETRAIN_TARGETS}" ]; then
  echo "FAST tokenizer and full-skill target pack already exist:"
  echo "  ${FAST_TOKENIZER}"
  echo "  ${PRETRAIN_TARGETS}"
  exit 0
fi

mkdir -p "$(dirname "${FAST_TOKENIZER}")"
cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"

echo "Preparing variable-length full-skill FAST targets"
echo "  tokenizer : ${FAST_TOKENIZER}"
echo "  targets   : ${PRETRAIN_TARGETS}"
echo "The first run downloads the tokenizer source from lerobot/fast-action-tokenizer if it is not cached."
"${PYTHON}" "${SCRIPT_DIR}/src/prepare_pretrain_targets.py" \
  --repo_id="${REPO_ID}" \
  --dataset_root="${SKILLVLA_DATASET_DIR}" \
  --transition_pack="${TRANSITION_PACK}" \
  --tokenizer_dir="${FAST_TOKENIZER}" \
  --target_pack="${PRETRAIN_TARGETS}" \
  --encoded_dims="${TOKENIZER_ENCODED_DIMS}" \
  --vocab_size="${FAST_VOCAB_SIZE}" \
  --scale="${TOKENIZER_SCALE}" \
  --max_fast_tokens="${MAX_FAST_TOKENS}"
