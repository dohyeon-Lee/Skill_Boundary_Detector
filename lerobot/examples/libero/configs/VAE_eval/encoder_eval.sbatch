#!/usr/bin/env bash
#SBATCH --job-name=VAE_eval

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# HOMEDIR=/scratch/mdorazi
# PROJDIR=/Skill_Boundary_Detector

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"

# ── 설정 ──────────────────────────────────────────────────────────────────────
CHECKPOINT_EPOCH=10000   # 평가할 체크포인트 epoch (예: 0100, 0200, ...)
NO_HTML=false            # true: interactive HTML 생성 스킵 (업로드 빠름), false: 생성
VAE_TYPE=spline   # 'raw' (BiLSTM) or 'spline' (MLP)

DATA=libero_90
LATENT_DIM=64
# SKILLSET=${DATA}_skillset
SKILLSET=${DATA}_skillset_fixed50


EXP="${DATA}_lat${LATENT_DIM}_${VAE_TYPE}"
SWEEP_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP}"

LATENTS_PATH="${SWEEP_DIR}/${VAE_TYPE}_vae_latents_epoch${CHECKPOINT_EPOCH}.npz"
OUTPUT_DIR="${SWEEP_DIR}/latent/${VAE_TYPE}_latents_epoch${CHECKPOINT_EPOCH}"
# ─────────────────────────────────────────────────────────────────────────────

echo "Experiment : ${EXP}"
echo "Checkpoint : epoch${CHECKPOINT_EPOCH}"
echo "Latents    : ${LATENTS_PATH}"

rm -rf ${OUTPUT_DIR}/wandb

WANDB_MODE=online \
WANDB_DIR=${OUTPUT_DIR} \

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/visualize_latents.py \
    --latents_path ${LATENTS_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --color_by all \
    --n_clusters 15 \
    --dataset_dir ${HOMEDIR}${PROJDIR}/libero_dataset/libero_90 \
    --skills_per_cluster 5 \
    --wandb_project VAE_eval \
    --wandb_run_name ${EXP}_epoch${CHECKPOINT_EPOCH} \
    $([ "${NO_HTML}" = "true" ] && echo "--no_html")

echo "DONE"
