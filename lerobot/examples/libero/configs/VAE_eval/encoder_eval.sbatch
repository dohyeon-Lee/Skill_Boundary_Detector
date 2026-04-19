#!/usr/bin/env bash
#SBATCH --job-name=vae_eval
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31,node25
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot
source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate


export WANDB_CACHE_DIR=/scratch/mdorazi/.wandb_cache
export WANDB_DATA_DIR=/scratch/mdorazi/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"

# ── 설정 ──────────────────────────────────────────────────────────────────────
CHECKPOINT_EPOCH=3800   # 평가할 체크포인트 epoch (예: 0100, 0200, ...)
NO_HTML=true            # true: interactive HTML 생성 스킵 (업로드 빠름), false: 생성


DATA=libero_90
LATENT_DIM=16
EXP="${DATA}_lat${LATENT_DIM}"
SWEEP_DIR="/scratch/mdorazi/Skill_Boundary_Detector/outputs/${DATA}_skill_latents/${EXP}"

LATENTS_PATH="${SWEEP_DIR}/skill_vae_latents_epoch${CHECKPOINT_EPOCH}.npz"
OUTPUT_DIR="${SWEEP_DIR}/latent/skill_vae_latents_epoch${CHECKPOINT_EPOCH}"
# ─────────────────────────────────────────────────────────────────────────────

echo "Experiment : ${EXP}"
echo "Checkpoint : epoch${CHECKPOINT_EPOCH}"
echo "Latents    : ${LATENTS_PATH}"

rm -rf ${OUTPUT_DIR}/wandb

WANDB_MODE=online \
WANDB_DIR=${OUTPUT_DIR} \
PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  python examples/libero/visualize_latents.py \
    --latents_path ${LATENTS_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --color_by all \
    --n_clusters 15 \
    --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90 \
    --skills_per_cluster 5 \
    --wandb_project VAE_eval \
    --wandb_run_name ${EXP}_epoch${CHECKPOINT_EPOCH} \
    $([ "${NO_HTML}" = "true" ] && echo "--no_html")

echo "DONE"
