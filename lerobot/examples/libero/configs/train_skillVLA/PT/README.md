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

## Top-UV Fullskill Bspline ablations

Choose one name in `VSA/vsa_train_config.yaml` under `architecture.name`:

* `UV_fullskill_bspline_3_lastlayer_lastvis`
* `UV_fullskill_bspline_3_lastlayer_lastvis_current`
* `UV_fullskill_bspline_3_lastlayer_lastvis_current_none`
* `UV_fullskill_bspline_3_lastlayer_lastvis_current_preskill`

Names are case insensitive. Saved architecture labels are lowercase; output
run names retain `UV_`. Splits still follow `visual_bridge_last_n_layers`
(9 means 9+9). These are new checkpoint identities, not reinterpretations of
the existing XYZ models.

Cond and the prefix Expert receive continuous **absolute Top-view endpoint
UV**, centered at zero in [-1,1] (u right, v down), through the same MLP/AdaRMS
connections as the old end XYZ. The Expert retains grounded raw skill-start
XYZ, but has no skill-start UV or end XYZ input. Invalid UV masks the entire
goal-adapter contribution, not the action/plan loss. The action suffix, 30x8
normalized proprio target, 1+1+98 patch interface, auxiliary losses/weights,
and start/current observation rules are unchanged. `0.01` in the Notion
patch tags means one query out of 100, NOT a new loss weight.

The preskill variant carries the previous occurrence's UV/validity alongside
its code/start XYZ and last observation. Its detached tau=1 prefix encoding
remains non-recursive and occurrence-cached; BOS reads remain fully masked.

PT and standalone FT use datasets with a valid Top-camera `skill_focus_uv.npz`.
PT/FT eval use GT UV for GT panels or a UV-enabled external Predictor for
predicted panels (Mode3 already inverse-transforms its center crop back to
original-image UV). An XYZ-only predictor is not silently replaced with GT UV.
End XYZ metadata is still retained for training-only patch geometry; it does
not enter either VSA conditioning adapter. Joint FT remains unsupported, as
for the original Bspline models.

Each of the four UV names also accepts `_diffaux` as its final suffix. This
replaces **only** the Top patch classification head with a LayerNorm + Linear
8D regression head on its dedicated bottleneck token: skill-end EEF XYZ3
(episode-grounded), axis-angle3, and measured finger positions2. The Wrist
patch head, 1+1+98 interface, gates and all plan/vision routes are unchanged.
The saved `chunk_end_state_q01/q99` (observation.state statistics) normalize
all eight target dimensions; MSE averages over dimensions and valid samples.
Invalid endpoint states are masked independently of Top UV visibility.

The existing `wrist_patch_align_loss_weight` multiplies
`wrist_patch_loss + end_state_mse` (no extra weight or 0.01 multiplier).
Held-start models retain `0.5 * (start_aux + current_aux)`; `_current` models
compute the shared current auxiliary once. `_preskill` memory has no extra
auxiliary loss. W&B adds `agent_end_state/raw_loss`, `valid_fraction`, and
normalized XYZ/rotation/gripper MAE (plus `start_...` for held-start models).
The endpoint prediction is training-only: eval inputs remain UV + start XYZ,
never GT end pose/gripper or a newly injected predicted state. Model/output
names and checkpoint revisions distinguish all four diffaux architectures.
Select these through `architecture.name` only; keep `end_pose_mode: xyz`.
That legacy input-selector does not select or truncate the diffaux 8D target.

## Mode3 Predictor spatial targets

All four `mode3_preskill`, `mode3_preskill_wristX`,
`mode3_preskill_wristX_proprioX`, and `mode3_preskill_proprioX` support
`skill_predictor.spatial_target: xyz` **or** `uv`. Previous-skill/BOS conditioning
and the selected wrist/proprio ablations are unchanged.

* `xyz`: predict canonical skill-end XYZ minus the actual jittered skill-start
  XYZ. Add the measured start origin back at inference, as before.
* `uv`: predict **absolute** skill-end UV in the augmented Top image, never
  end-minus-start UV. Dataset build must provide `skill_focus_uv.npz` from
  `focus_uv: {enabled: true, camera: agentview}`. Canonical endpoint/temporal
  jitter semantics remain the same as the original UV dataset.

Both keep the 1.1x Top zoom plus per-sample uniform crop. For UV, one explicit
crop draw is shared by image pixels and labels; its transform also includes
the actual integer zoom dimensions, bilinear half-pixel geometry, and final
224 resize/padding. Targets outside the resulting image (or already invalid)
do not contribute UV loss, but skill loss continues for **every** sample.
W&B reports `focus_uv_valid_fraction`, `focus_uv_original_valid_fraction`, and
`focus_uv_crop_masked_fraction` under `skill_predictor/`.

UV inference always uses a center crop and returns UV **in the original image
coordinate system** after the exact inverse transform. A UV-based VSA therefore
receives its ordinary normalized `[-1, 1]` coordinates, not crop coordinates;
no XYZ/depth reconstruction is involved. Existing UV model contracts and
`predictor_uv...<mode>` names distinguish these runs from XYZ. Standalone FT
inherits the UV contract and uses the same training/eval path. The separate
offline `FT/eval/submit_pred_eval.sh` tool remains an XYZ-error diagnostic;
use the normal PT/FT closed-loop eval with a UV-based VSA for UV predictors.

Legacy non-mode3 UV models do **not** receive this new zoom/crop or coordinate
change. Existing mode3 XYZ images and relative targets keep their old behavior.

## Predictor modes sharing one GPU

`Predictor/predictor_train_config.yaml` keeps the original single-model behavior
with a scalar `skill_predictor.mode: mode3_preskill`. To train independent models
simultaneously on one large GPU, explicitly select a list instead:

```yaml
skill_predictor:
  mode: [mode3_preskill, mode3_preskill_proprioX]
training:
  dataloader: {batch_size: 16, workers: 6, gpus: 1}
slurm: {gres: "gpu:1", cpus: 16, memory: 192G, time: "48:00:00"}
```

Run the same `Predictor/submit_train.sh`. This submits **one allocation**, not a
job array: both trainers inherit the same `CUDA_VISIBLE_DEVICES`, have their own
weights/optimizer/W&B run, and use the usual mode-specific output folders. The
dataset is staged only once. Batch size and workers apply **per model**; Slurm
CPU and host-memory limits are **totals**. The example resources are not a VRAM
guarantee: model/optimizer/activation memory adds up on the shared GPU, and GPU
compute is shared too. No MPS or fractional GPU reservation is enabled.

The allocation log reports each worker's status; separate stdout/stderr and
exit codes are under `Predictor/logs/predictor_group_<job id>/`. A failed worker
does not cancel healthy ones, but the allocation exits nonzero if any failed.
Cancellation stops the whole group, including dataloader workers. Resubmitting
uses each model's own `checkpoints/last`, just like a scalar run. Duplicate modes
and multi-GPU list configurations are rejected before submission.

Use `SKILL_AUX_DRY_RUN=1 bash Predictor/submit_train.sh` to inspect the snapshot,
resolved output paths, and one-GPU request without submitting. Lists are PT-only;
FT continues to inherit its architecture from its warm-start checkpoint.

## Direct / Residual layer split

For `direct_patch_1..6`, `residual_patch_1..6`, and their `_dGD` variants,
`architecture.visual_bridge_last_n_layers` accepts 1..17. The first `18-N`
Expert layers predict the 4H motion base; the last `N` layers process its
current H-token slice and each read their corresponding Cond bottleneck.
For example, `1` keeps the existing 17+1 split and run name, while `9` selects
9+9 and adds `_layer9` after the architecture label in the run name.
`_dGD` detaches the action path at the selected split. FT and evaluation
inherit the split from the checkpoint without another setting.

## Bspline current-observation memory ablations

Both models retain `fullskill_bspline_3_lastlayer_lastvis_current`'s live
current Z18, H-token main path, current proprio plan auxiliary loss, and one
current patch auxiliary pass. Select either under `architecture.name`; use
`visual_bridge_last_n_layers: 9` for the 9+9 split (the split remains configurable).

- `fullskill_bspline_3_lastlayer_lastvis_current_preskill`: the first late layer
  reads the previous skill's 30 plan hidden tokens instead of the current plan.
  Training uses frame `s-1`, where `s` is the **same causal jittered current
  start**; previous skill code/start/end XYZ accompany that observation. A
  previous start later than this observation is causally clipped too. Extraction
  uses a fixed Gaussian template at tau=1, runs only Cond and the Expert prefix
  under no-grad/eval, and stops **before** any plan read: there is no recursion.
  Eval snapshots every observed frame, not only action replans, and caches the
  previous plan per skill occurrence. Episode BOS has a masked whole reader
  update; reset clears history. Past inputs use the saved state normalization.
- `fullskill_bspline_3_lastlayer_lastvis_current_none`: bypass the entire
  30-token cross-attention reader. The prefix's H hidden tokens pass directly to
  the late layers, while 30-token proprio auxiliary supervision remains active.
  It needs no prior observation encoding or cache.

Existing architectures retain their original memory and gradient behavior.
FT inherits these contracts from its warm-start checkpoint; PT and FT eval use
the same occurrence-aware runtime wrapper. External callers of the preskill
model must provide the explicit `preskill_*` observation contract (including a
validity mask and occurrence id), not a recursively generated action history.

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
