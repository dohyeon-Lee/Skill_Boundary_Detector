# SkillVLA PT training

The three PT components are independent jobs. Edit shared dataset/model
inputs in [`stage1_common_config.yaml`](stage1_common_config.yaml), then edit
the component-specific YAML and run its launcher:

The code directory was renamed from `stage1` to `PT`. Existing YAML names,
`STAGE1_*` environment variables, and snapshot formats remain compatible.
Training now writes to `outputs_root/skillVLA_PT/`. Existing `outputs/` and
`outputs_filtered/` PT groups were renamed in place; checkpoints and run names
are unchanged. Saved paths containing `skillVLA_stage1` remain readable, and
run-name selectors also accept old layouts on servers not yet migrated.
Evaluation results, logs, and snapshots moved with this
directory, so closed-loop evaluation now runs from `PT/eval`.

| Component | Config | Launcher | New output location |
| --- | --- | --- | --- |
| VSA | `VSA/vsa_train_config.yaml` | `VSA/submit_train.sh [architecture]` | `outputs_root/skillVLA_PT/VSA/<run>` |
| Skill Predictor | `Predictor/predictor_train_config.yaml` | `Predictor/submit_train.sh` | `outputs_root/skillVLA_PT/Predictor/<run>` |
| Terminator | `Terminator/terminator_train_config.yaml` | `Terminator/submit_train.sh` | `outputs_root/skillVLA_PT/Terminator/<run>` |

Each launcher snapshots its own YAML, the shared YAML, and the global cluster
YAML before submission. Nested fields in a component config override shared
fields. For example, Terminator can set `dataset.relabeled` without changing
VSA or Predictor. Predictor PT training always uses original skill labels.

`VSA/submit_train.sh` is the VSA launcher itself (shared code in `src/`); the old
`PT/submit_train.sh` + `stage1_train_config.yaml` pair was removed.
`Terminator/src/submit_auxiliary.sh` backs the Predictor and Terminator launchers,
including NewTask FT; its config helpers are shared with FT VSA and Joint.
The old top-level `terminator/` directory was removed. Its historical logs and
snapshots are under `Terminator/logs/legacy_auxiliary/` and
`Terminator/.config_snapshots/legacy_auxiliary/`. The historical flat
`skillVLA_terminator/` group stays in place. Downstream resolvers accept both
old and new run locations by run name.

## Direct / Residual layer split

For `direct_patch_1..6`, `residual_patch_1..6`, and their `_dGD` variants,
`architecture.visual_bridge_last_n_layers` accepts 1..17. The first `18-N`
Expert layers predict the 4H motion base; the last `N` layers process its
current H-token slice and each read their corresponding Cond bottleneck.
For example, `1` keeps the existing 17+1 split and run name, while `9` selects
9+9 and adds `_layer9` after the architecture label in the run name.
`_dGD` detaches the action path at the selected split. FT and evaluation
inherit the split from the checkpoint without another setting.

## Checkpoint sweeps in Stage-1 eval

In `eval/stage1_eval_config.yaml`, use
`model_defaults.checkpoint: [10000, 20000, 50000]` to evaluate every model at
those checkpoints, or set a model's `checkpoint` list. Numeric steps are
normalized to six-digit folder names. Multiple models use the same ordered
checkpoint list for side-by-side evaluation.

The result folder retains the existing bar charts and also writes these sweep
line charts automatically: `checkpoint_success_rates.png` overlays the models'
overall mean SR; `checkpoint_task_success_rates/<index>_<model>.png` contains
one chart per model with a separate line for each task. Thus one model produces
two line charts, and N models produce N+1. Curves sort checkpoints numerically,
and unavailable task results are gaps. Every finished array worker refreshes
the charts using the metrics available so far.
