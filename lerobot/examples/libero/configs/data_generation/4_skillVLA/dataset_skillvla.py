#!/usr/bin/env python3
# Stage 4b: add FSQ skill latents to a LeRobot dataset for SkillVLA training.
#
# Inputs (all under ${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data/):
#   - ${DATA}_for_skillVLA/${OUTPUT_NAME}/skill_latents.npz  (from encode_fsq_skills.sbatch)
#   - ${DATADIR}/${DATA}/                                     (raw LeRobot dataset)
#
# Optional:
#   - ${DATA}_for_FSQ/${IMAGE_FEATURE_TAG}_tokens.npz        (precomputed DINO features)
#
# Output:
#   ${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data/${DATA}_for_skillVLA/${DATA}_skillvla/
#
# Run directly : python dataset_skillvla.py
# Submit to SLURM: sbatch dataset_skillvla.py

#SBATCH --job-name=skillvla_dataset
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

import os
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONFIG_DIR))

from pipeline_config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    latents_path = Path(cfg.skill_latents_path)
    src_dataset_dir = Path(cfg.raw_dataset_dir)
    dst_dataset_dir = Path(cfg.skillvla_dataset_dir)
    dst_repo_id = f"dohyeon/{cfg.skillvla_dataset}"
    print(f"SkillVLA dataset creation")
    print(f"  dataset   : {cfg.data}")
    print(f"  src       : {src_dataset_dir}")
    print(f"  dst       : {dst_dataset_dir}")
    print(f"  latents   : {latents_path}")
    print(f"  fsq_levels: {list(cfg.fsq_levels)}")
    print(f"  max_order : {cfg.max_order}  max_length: {cfg.max_length}")
    print(f"  dino      : (skipped — load separately at train time via npz mmap)")

    if not latents_path.exists():
        print(f"ERROR: skill_latents.npz not found: {latents_path}", file=sys.stderr)
        print("Run configs/data_generation/4_skillVLA/encode_fsq_skills.sbatch first.", file=sys.stderr)
        sys.exit(1)

    if not src_dataset_dir.exists():
        print(f"ERROR: source dataset not found: {src_dataset_dir}", file=sys.stderr)
        sys.exit(1)

    lerobot_root = Path(f"{cfg.homedir}{cfg.projdir}/lerobot")
    script = lerobot_root / "examples/libero/add_skill_latents_to_dataset.py"

    cmd = [
        sys.executable, str(script),
        "--src_dataset_dir", str(src_dataset_dir),
        "--dst_dataset_dir", str(dst_dataset_dir),
        "--dst_repo_id",     dst_repo_id,
        "--latents_path",    str(latents_path),
        "--fsq_levels",      *[str(l) for l in cfg.fsq_levels],
        "--max_order",       str(cfg.max_order),
        "--max_length",      str(cfg.max_length),
    ]

    # DINO features are NOT embedded in parquet.
    # The _tokens.npz stores (N, 65, 384) patch tokens — 2D per frame, incompatible
    # with parquet column format. Load separately at train time via np.load(mmap_mode='r')
    # and index by (episode_id, frame_index). See load_dino_feature_map() for the mapping.

    env = os.environ.copy()
    env["PYTHONPATH"] = str(lerobot_root / "src")

    subprocess.run(cmd, env=env, check=True)

    print(f"\nDONE  →  {dst_dataset_dir}")


if __name__ == "__main__":
    main()
