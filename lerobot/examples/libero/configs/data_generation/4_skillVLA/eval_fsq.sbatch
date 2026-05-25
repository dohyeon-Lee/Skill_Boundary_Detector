#!/usr/bin/env bash
# SkillVLA FSQ eval — GPU 노드에서 eval.py 실행
#
# 사용법:
#   sbatch eval_fsq.sbatch
#
# 환경변수로 오버라이드 가능:
#   DATA=libero_90 RUN_NAME=<run_name> sbatch eval_fsq.sbatch
#   MAX_EVAL_SKILLS=500 sbatch eval_fsq.sbatch   # 빠른 테스트용 (0 = 전체)

#SBATCH --job-name=FSQ_eval
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --partition=suma_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}

# pipeline_config.py 위치 찾기
CONFIG_PY=${SUBMIT_DIR}/examples/libero/configs/data_generation/pipeline_config.py
if [ ! -f "${CONFIG_PY}" ]; then
  CONFIG_PY=${SUBMIT_DIR}/../pipeline_config.py
fi
if [ ! -f "${CONFIG_PY}" ]; then
  echo "Could not find pipeline_config.py from ${SUBMIT_DIR}." >&2
  echo "Submit from lerobot root or from 4_skillVLA/." >&2
  exit 1
fi

# Python 실행 파일 찾기
BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../.venv/bin/python
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../../../../../../.venv/bin/python
fi
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --shell)"

cd "${HOMEDIR}${PROJDIR}/lerobot"
source "${HOMEDIR}${PROJDIR}/.venv/bin/activate"
unset LD_LIBRARY_PATH

# 파라미터 (환경변수 오버라이드 가능)
DATA=${DATA:-${DATA:-libero_90}}
RUN_NAME=${RUN_NAME:-${FSQ_RUN_NAME:-}}
MAX_EVAL_SKILLS=${MAX_EVAL_SKILLS:-0}
N_ACTION_STEPS=${N_ACTION_STEPS:-5}

# RUN_NAME이 없으면 pipeline_config 기본값 사용 (eval.py 내부에서 처리)
if [ -n "${RUN_NAME}" ]; then
  RUN_NAME_ARG="--run_name ${RUN_NAME}"
else
  RUN_NAME_ARG=""
fi

EVAL_SCRIPT="${HOMEDIR}${PROJDIR}/lerobot/examples/libero/configs/data_generation/4_skillVLA/eval.py"
OUTPUT_HTML="${HOMEDIR}${PROJDIR}/lerobot/examples/libero/configs/data_generation/4_skillVLA/html/${DATA}_${RUN_NAME:-default}_eval.html"

echo "SkillVLA FSQ eval (GPU)"
echo "  data            : ${DATA}"
echo "  run_name        : ${RUN_NAME:-<pipeline_config default>}"
echo "  max_eval_skills : ${MAX_EVAL_SKILLS} (0=전체)"
echo "  n_action_steps  : ${N_ACTION_STEPS} (chunk plot stride)"
echo "  device          : cuda"
echo "  output          : ${OUTPUT_HTML}"
echo "  job id          : ${SLURM_JOB_ID}"
echo "  node            : ${SLURMD_NODENAME}"
echo ""

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python "${EVAL_SCRIPT}" \
    --data              "${DATA}" \
    ${RUN_NAME_ARG} \
    --max_eval_skills   "${MAX_EVAL_SKILLS}" \
    --n_action_steps    "${N_ACTION_STEPS}" \
    --device            cuda

echo "DONE — HTML: ${OUTPUT_HTML}"
