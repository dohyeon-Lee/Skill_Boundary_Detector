#!/usr/bin/env bash
#SBATCH --job-name=tsne_sweep
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot
source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# ── Experiment list (must match train_vae_sweep) ───────────────────────────────
EXPS=(
  "vae90_lat8_hid128"
  # "vae_lat32_hid128"
  # "vae_lat64_hid128"
  # "vae_lat32_hid256"
  # "vae_lat64_hid256"
)

IDX=$SLURM_ARRAY_TASK_ID
EXP=${EXPS[$IDX]}
SWEEP_DIR="/scratch/mdorazi/Skill_Boundary_Detector/outputs/vae_90/${EXP}"

echo "Experiment: ${EXP}"

# ── Run t-SNE for each available latents file ──────────────────────────────────
for LATENTS_PATH in ${SWEEP_DIR}/skill_vae_latents_epoch*.npz; do
    [ -f "${LATENTS_PATH}" ] || continue

    BASENAME=$(basename ${LATENTS_PATH} .npz)
    OUTPUT_DIR="${SWEEP_DIR}/tsne/${BASENAME}"

    echo "  Processing: ${LATENTS_PATH}"

    PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
      python examples/libero/visualize_latents.py \
        --latents_path ${LATENTS_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --color_by all \
        --n_clusters 10 \
        --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90 \
        --skills_per_cluster 10 \
        --wandb_project SBD_vae_viz \
        --wandb_run_name ${EXP}_${BASENAME}
done

echo "DONE ${EXP}"
