# ABC workflows

ABC is a sibling of LIBERO under `lerobot/skillVLA`. This tree contains ABC
real-world/simulator data preparation, DP and FSQ training, SkillVLA data building
and VSA PT, and PI0.5 training/evaluation. ABC wrappers reuse shared LIBERO
implementations where appropriate while keeping ABC configs and data contracts.

## Directory guide

| Path | Purpose |
| --- | --- |
| [configs/global_config.yaml](configs/global_config.yaml) | Dataset/output roots, task groups and shared defaults. |
| [../servers/](../servers/) | Shared ABC/LIBERO storage, scheduler and Slurm settings. |
| [configs/generate_training_dataset/](configs/generate_training_dataset/) | Real/sim downloads, native dataset views, episode selection and previews. |
| [configs/train_skills/DP/](configs/train_skills/DP/) | Diffusion Policy training and native action-statistics preparation. |
| [configs/train_skills/build_data/](configs/train_skills/build_data/) | DP-based SBD curves, global boundary thresholds and segmented skillsets. |
| [configs/train_skills/FSQ/](configs/train_skills/FSQ/) | Skill tokenizer training from the selected skillset. |
| [configs/train_skills/DP/eval/](configs/train_skills/DP/eval/) | SBD analysis/ablations, boundary results and interactive reports. |
| [configs/train_skills/FSQ/eval/](configs/train_skills/FSQ/eval/) | FSQ codebook and decoder evaluation. |
| [configs/train_skillVLA/build_data/](configs/train_skillVLA/build_data/) | Encode skills and build native SkillVLA annotations, initial states and bimanual spatial targets. |
| [configs/train_skillVLA/PT/](configs/train_skillVLA/PT/) | ABC VSA post-training; the launcher is under `VSA/`. |
| [configs/train_pi05/PT/](configs/train_pi05/PT/) | PI0.5 post-training on ABC native data. |
| [configs/train_pi05/PT/eval/](configs/train_pi05/PT/eval/) | Closed-loop ABC Simulator evaluation, videos and reports. |
| [configs/src/](configs/src/) | Shared ABC configuration, runtime and submission helpers. |
| [configs/src/simulator/](configs/src/simulator/) | Pinned ABC Simulator environment, asset installation and environment checks. |

## SkillVLA workflow

1. Prepare the native ABC dataset under `generate_training_dataset/`.
2. Train a DP with `train_skills/DP/submit_train_dp.sh`.
3. Build its SBD skillset with `train_skills/build_data/submit_build_data.sh`.
4. Train FSQ with `train_skills/FSQ/submit_train_fsq.sh`.
5. Build SkillVLA training annotations with `train_skillVLA/build_data/submit_build_all.sh`.
6. Train VSA with `train_skillVLA/PT/VSA/submit_train.sh`.

The paths in these steps are relative to `configs/`. Run each launcher from its
own component directory after setting its YAML. Skill-boundary and FSQ analysis
use the separate `train_skills/skill_eval/` entrypoints. PI0.5 uses its own
`train_pi05/PT/submit_train.sh` and `train_pi05/PT/eval/submit_eval.sh`.

Dataset/checkpoint locations come from the global config and server overlay;
the default roots are `dataset_ABC/` and `outputs_ABC/` at the project root.
Model weights and generated data are separate from this code tree. Submission
scripts preserve per-run config snapshots.
