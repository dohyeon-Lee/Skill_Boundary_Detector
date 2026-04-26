#!/usr/bin/env bash
#SBATCH --job-name=DP_skill

# SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=base_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# ── 설정 ──────────────────────────────────────────────────────────────────────
CHUNK_LEN=40
DATASET=libero_90

DATASET_DIR=${HOMEDIR}${PROJDIR}/libero_dataset/${DATASET}
OUTPUT_DIR=${HOMEDIR}${PROJDIR}/libero_dataset/${DATASET}_skillset_fixed${CHUNK_LEN}
# ─────────────────────────────────────────────────────────────────────────────

# 각 array job이 5개 task 처리 (0-14 × 5 = task 0~74, 72 초과는 skip)
TASKS_PER_JOB=5
START_TASK=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))
END_TASK=$(( START_TASK + TASKS_PER_JOB - 1 ))

TASK_IDS=""
for TASK_ID in $(seq ${START_TASK} ${END_TASK}); do
    if [ ${TASK_ID} -gt 72 ]; then
        break
    fi
    TASK_IDS="${TASK_IDS} ${TASK_ID}"
done

echo "Processing tasks:${TASK_IDS}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/build_skill_dataset_fixed.py \
    --dataset_dir ${DATASET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --chunk_len ${CHUNK_LEN} \
    --task_ids ${TASK_IDS} \
    --resume

echo "DONE array ${SLURM_ARRAY_TASK_ID}"
