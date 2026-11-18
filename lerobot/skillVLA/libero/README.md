# LIBERO workflows

LIBERO is a sibling of ABC under `lerobot/skillVLA`. This tree contains dataset
preparation, DP/SBD/FSQ workflows, SkillVLA PT/FT and evaluation, and the PI0.5,
GR00T and MolmoAct2 baselines.

## Directory guide

| Path | Purpose |
| --- | --- |
| [configs/global_config.yaml](configs/global_config.yaml) | Dataset/output roots and shared defaults. |
| [../servers/](../servers/) | Shared ABC/LIBERO storage, scheduler and Slurm settings. |
| [configs/generate_training_dataset/](configs/generate_training_dataset/) | Dataset downloads, filtering, splitting, conversion and previews. |
| [configs/train_skills/DP/](configs/train_skills/DP/) | Diffusion Policy training. |
| [configs/train_skills/build_data/](configs/train_skills/build_data/) | DP-based SBD curves, boundary calibration and segmented skillsets. |
| [configs/train_skills/FSQ/](configs/train_skills/FSQ/) | Skill tokenizer training and frame/DINO feature caches. |
| [configs/train_skills/skill_eval/](configs/train_skills/skill_eval/) | SBD analysis, FSQ codebook/decoder evaluation, EEF projection and reports. |
| [configs/train_skillVLA/build_data/](configs/train_skillVLA/build_data/) | SkillVLA dataset construction: segmentation, FSQ encoding, skill annotations and spatial targets. |
| [configs/train_skillVLA/PT/](configs/train_skillVLA/PT/) | VSA, Predictor and Terminator post-training; shared settings in `stage1_common_config.yaml`. |
| [configs/train_skillVLA/FT/](configs/train_skillVLA/FT/) | VSA, Predictor and Terminator fine-tuning; shared settings in `newtask_ft_common_config.yaml`. |
| [configs/train_skillVLA/PT/eval/](configs/train_skillVLA/PT/eval/) | PT-task evaluation and retention/forgetting experiments. |
| [configs/train_skillVLA/FT/eval/](configs/train_skillVLA/FT/eval/) | FT-task evaluation, Predictor diagnostics, videos and reports. |
| [configs/train_pi05/](configs/train_pi05/) | PI0.5 PT/FT and their evaluation entrypoints. |
| [configs/train_groot/](configs/train_groot/) | GR00T PT/FT and evaluation using its isolated model environment. |
| [configs/train_molmoact2/](configs/train_molmoact2/) | MolmoAct2 PT/FT and evaluation using its isolated model environment. |
| [configs/train_cyclic/pi05/](configs/train_cyclic/pi05/) | PI0.5 IID/block-cyclic training variants, FT and evaluation. |
| [configs/paper_vis/](configs/paper_vis/) | Preprocessing and post-training paper visualizations. |
| [configs/src/](configs/src/) | Shared configuration, runtime, GPU protection, submission, baseline evaluation and HF synchronization helpers. |

## SkillVLA workflow

1. Prepare the dataset under `generate_training_dataset/`.
2. Train a DP with `train_skills/DP/submit_train_dp.sh`.
3. Build its SBD skillset with `train_skills/build_data/submit_build_data.sh`.
4. Train FSQ with `train_skills/FSQ/submit_train_fsq.sh`.
5. Build PT/FT SkillVLA datasets with `train_skillVLA/build_data/submit_build_all.sh`.
6. Train VSA, Predictor and Terminator using their `PT/<component>/submit_train.sh` entrypoints.
7. Fine-tune each component using `FT/<component>/submit_train.sh` and the selected PT checkpoints.
8. Evaluate the chosen checkpoint combination with `PT/eval/submit_eval.sh` or `FT/eval/submit_eval.sh`.

All paths in these steps are relative to `configs/`; component paths in steps
6–8 are under `train_skillVLA/`. Run each launcher from its own directory after
setting its YAML. The baseline families also keep separate `PT/`, `FT/` and
stage-local `eval/` entrypoints.

Dataset/checkpoint locations come from the global config and server overlay;
the current roots are `dataset_filtered/` and `outputs_filtered/` at the project
root. SkillVLA component checkpoints are grouped under
`outputs_filtered/skillVLA_PT/` and `outputs_filtered/skillVLA_FT/`.
Evaluation outputs are written under the corresponding `eval/outputs/` folder.
Model weights and generated data are separate from this code tree. Submission
scripts preserve per-run config snapshots.
