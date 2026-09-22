# Stage-1 training

The three Stage-1 components are independent jobs. Edit shared dataset/model
inputs in [`stage1_common_config.yaml`](stage1_common_config.yaml), then edit
the component-specific YAML and run its launcher:

| Component | Config | Launcher | New output location |
| --- | --- | --- | --- |
| VSA | `VSA/vsa_train_config.yaml` | `VSA/submit_train.sh [architecture]` | `outputs_root/skillVLA_stage1/VSA/<run>` |
| Skill Predictor | `Predictor/predictor_train_config.yaml` | `Predictor/submit_train.sh` | `outputs_root/skillVLA_stage1/Predictor/<run>` |
| Terminator | `Terminator/terminator_train_config.yaml` | `Terminator/submit_train.sh` | `outputs_root/skillVLA_stage1/Terminator/<run>` |

Each launcher snapshots its own YAML, the shared YAML, and the global cluster
YAML before submission. Nested fields in a component config override shared
fields. For example, Terminator can set `dataset.relabeled` without changing
VSA or Predictor. Predictor PT training always uses original skill labels.

`VSA/submit_train.sh` is the VSA launcher itself (shared code in `src/`); the old
`stage1/submit_train.sh` + `stage1_train_config.yaml` pair was removed.
`terminator/submit_train.sh` still backs the Predictor and Terminator launchers.
No existing checkpoints or logs are moved. Downstream resolvers accept both old
and new run locations by run name.
