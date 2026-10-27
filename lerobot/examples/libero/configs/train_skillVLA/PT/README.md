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

All components use the shared trainer, which never uploads checkpoint weights
or adapters to W&B. Old resume configs cannot re-enable model uploads with
`wandb.disable_artifact=false`. W&B metrics, plots/videos, local saves and
server-to-server checkpoint archival are unchanged. This applies at the next
trainer launch; running processes and existing cloud artifacts are untouched.

## NewTerm token-input ablations

Set `fsq_terminator.architecture` in `Terminator/terminator_train_config.yaml`
to `Term_Token_1` through `Term_Token_6` (optionally ending in `_both`) and run
`./submit_train.sh` from `Terminator/`. Names are case insensitive.

| Name | Context beyond skill and current Top-view image |
| --- | --- |
| Term_Token_1 | skill-start proprio (identical to Newterm8) |
| Term_Token_2 | skill-end XYZ (identical to Newterm9) |
| Term_Token_3 | skill-start proprio + skill-end Top-view UV |
| Term_Token_4 | skill-end Top-view UV |
| Term_Token_5 | none: skill only (identical to Newterm1) |
| Term_Token_6 | skill-start proprio + skill-end XYZ (identical to Newterm7) |

All retain Newterm7's frozen DINO, token conditioning, raw skill skip,
termination-only loss and ten-frame post-end supervision. No current proprio
token, history, patch auxiliary or extra start jitter is added. Token1/2 reuse
the existing profiles but produce separate `terminator_Term_Token_<n>` run names.
Existing Newterm1/7/8/9 checkpoints do not need retraining just to change the label.
The six `_both` profiles change **only** camera selection to Top + Wrist; UV
coordinates still refer exclusively to Top view. They have distinct output names.

Existing checkpoint run paths on this server use the same aliases:
`Newterm1 -> Term_Token_5`, `Newterm7 -> Term_Token_6`,
`Newterm8 -> Term_Token_1`, `Newterm9 -> Term_Token_2`.
Legacy paths remain usable through relative compatibility symlinks. The FT
`Newterm7_own_lab` run likewise maps to `Term_Token_6_own_lab`. The PT Newterm7
real directory is named `Term_Token_6`; its old `Newterm7` path is a relative
compatibility symlink to that directory.
Saved checkpoint JSON, training-state metadata, historical snapshots and evaluation
outputs are not rewritten. There is no second weight copy or extra model storage.

For XYZ and UV goals, train-time noise is the same independent uniform
`[-0.01, +0.01]` metres on each XYZ axis. UV variants project the **perturbed
grounded XYZ** into the recorded Top camera, including the original image's
horizontal mirror, then center-normalize UV to `[-1,1]`. No UV-space noise or
XYZ-statistics normalization is applied to the UV token. Current and past-skill
conditions use their respective canonical skill endpoints and start states.
Noise is disabled for evaluation. Out-of-view UV follows the dataset's clipping
convention; nonfinite/behind-camera goals get a finite neutral token.

UV training requires a focus artifact with embedded per-episode camera matrices
or the existing episode-exact HDF5/XML sources used to build that artifact.
Recorded calibration respects `episode_start_xyz`; it never guesses a shared
camera pose. Standalone FT inherits this representation from the PT checkpoint.
PT/FT rollout evaluation reads the external Terminator's UV contract independently
of the VSA: GT goals require `oracle.episode_exact: true`, while predicted UV
goals require a UV-capable Predictor. An XYZ VSA does not acquire a UV input
merely because its Terminator uses UV.

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

## Final-layer-only UV vision ablations

These four names append `_last` to their corresponding UV baseline:

* `UV_Fullskill_Bspline_3_lastlayer_lastvis_last`
* `UV_Fullskill_Bspline_3_lastlayer_lastvis_diffaux_last`
* `UV_Fullskill_Bspline_3_lastlayer_lastvis_current_preskill_diffaux_last`
* `UV_Fullskill_Bspline_3_lastlayer_lastvis_current_preskill_last`

Only the suffix vision bridge changes: current Z18 enters L18 alone instead
of L10–L18. Keep `visual_bridge_last_n_layers: 9` for the 9+9 split; setting
it to 1 would also change the plan split. Prefix vision still enters L9,
using held skill-start Z18 or shared current Z18 according to the baseline.
Current proprio still conditions every suffix layer. Plan generation/read,
detached preskill memory and BOS masking, the separate 1+1+98 bottleneck
self-attention groups, and patch/diffaux targets and loss weights are retained.
The `_diffaux_last` names keep the 8D auxiliary head; they do not predict XYZ
for the Expert. PT, standalone FT and evaluation retain distinct labels and
checkpoint revisions for the variants. Select a name through
`architecture.name` or the existing `submit_train.sh` architecture argument.

## Joint bottleneck self-attention ablation

`UV_Bspline_lastlayer_current_preskill_diffaux` inherits
`UV_fullskill_bspline_3_lastlayer_lastvis_current_preskill_diffaux`, with only
the bottleneck self-attention scope changed. At each Cond layer, Top query 0
still cross-attends only Top memory, Wrist query 1 only Wrist memory, and queries
2–99 both cameras. Their cross-attention residuals are concatenated before one
existing self-attention/MLP update over all 100 queries. No new weights or gate
are added; the existing Expert bridges and their gates remain unchanged.
The same update applies to the detached preskill encoder. UV/start-XYZ inputs,
8D endpoint auxiliary, Wrist patch loss, Bspline targets and memory rules are
inherited. The saved architecture label/revision and run name distinguish it
from the original partitioned model.

Select `architecture.name: UV_Bspline_lastlayer_current_preskill_diffaux`, or
run `./submit_train.sh UV_Bspline_lastlayer_current_preskill_diffaux` from `PT/VSA`.

## Predicted XYZ Expert input (diffaux2)

Set `architecture.name` to one of these names; no additional option is required:

| Name | Predicted-XYZ gradient into Expert | Top XYZ loss coefficient |
| --- | --- | --- |
| `UV_Bspline_lastlayer_current_preskill_diffaux2_detach` | detached | 0.1 |
| `UV_Bspline_lastlayer_current_preskill_diffaux2_attach` | attached | 0.1 |
| `UV_Bspline_lastlayer_current_preskill_diffaux2_detach_topw1` | detached | 1.0 |
| `UV_Bspline_lastlayer_current_preskill_diffaux2_attach_topw1` | attached | 1.0 |

Cond keeps endpoint UV and joint 100-token self-attention. The Top token predicts
only normalized skill-end XYZ3. Both training and inference inverse-normalize
that prediction to episode-grounded metres before the prefix Expert's goal
adapter, matching its unchanged skill-start XYZ input. There is no GT XYZ input
or fallback in this path. The same prediction path is used by detached preskill
re-encoding. Wrist patch supervision and 30x8 Bspline planning are unchanged.
With the default `spatial_loss_weight: 0.1`, Wrist has coefficient 0.1 in all four
variants; `_topw1` fixes the XYZ coefficient at 1.0 independently. Other variants
use the same coefficient as Wrist. Detach stops only the predicted goal's Expert
input branch; the XYZ head remains supervised by XYZ MSE. Component raw losses,
weights and weighted losses are logged separately. Each name has its own saved
architecture revision and run directory.

## GT XYZ variants: diffaux3 and diffaux4

Select any of these fourteen names through `architecture.name` or the submission argument:

| Name | XYZ readout | Prefix vision / plan |
| --- | --- | --- |
| `UV_Bspline_lastlayer_diffaux3` | Bottleneck query 2 | Held start / current plan |
| `UV_Bspline_lastlayer_diffaux3_last` | Bottleneck query 2 | Held start / current plan |
| `UV_Bspline_lastlayer_current_preskill_diffaux3` | Bottleneck query 2 | Current / detached previous plan |
| `UV_Bspline_lastlayer_current_preskill_diffaux3_last` | Bottleneck query 2 | Current / detached previous plan |
| `UV_Bspline_lastlayer_diffaux4` | Learned Gemma token + UV AdaRMS | Held start / current plan |
| `UV_Bspline_lastlayer_diffaux4_uvtoken` | UV Gemma token; UV AdaRMS removed | Held start / current plan |
| `UV_Bspline_lastlayer_current_preskill_diffaux4` | Learned Gemma token + UV AdaRMS | Current / detached previous plan |
| `UV_Bspline_lastlayer_current_preskill_diffaux4_uvtoken` | UV Gemma token; UV AdaRMS removed | Current / detached previous plan |
| `UV_Bspline_lastlayer_diffaux4_last` | Learned Gemma token + UV AdaRMS | Held start / current plan |
| `UV_Bspline_lastlayer_current_preskill_diffaux4_last` | Learned Gemma token + UV AdaRMS | Current / detached previous plan |
| `UV_Bspline_lastlayer_diffaux4_start` | Learned Gemma token, start-only XYZ loss | Held start / current plan |
| `UV_Bspline_lastlayer_current_preskill_diffaux4_start` | Learned Gemma token, start-only XYZ loss | Current / detached previous plan |
| `UV_Bspline_lastlayer_diffaux4_start_last` | Learned Gemma token, start-only XYZ loss; L18 late vision | Held start / current plan |
| `UV_Bspline_lastlayer_current_preskill_diffaux4_start_last` | Learned Gemma token, start-only XYZ loss; L18 late vision | Current / detached previous plan |

During training, the prefix Expert receives raw GT skill-end XYZ in episode-grounded metres.
Each axis receives independent uniform
noise in `[-0.01, +0.01]` metres. The SAME noisy endpoint supplies Expert XYZ
and the calibrated Top UV projection. Never add noise in UV space. Auxiliary
labels stay clean. Evaluation disables this noise. Training requires recorded
per-episode Top calibration, loaded through the existing camera-artifact path.

Diffaux3 retains Top/Wrist/XYZ/action queries 1+1+1+97. Diffaux4 uses
1+1+98 bottleneck queries and adds one token to Cond Gemma (outside that 100).
All 100 bottleneck queries mix via joint self-attention. The Gemma XYZ token
participates in bidirectional Gemma attention but is excluded from the direct
bottleneck memory and camera patch indexing. Its final hidden state predicts
normalized XYZ3 through a separate head. `_uvtoken` initializes that token
from projected UV and removes only UV AdaRMS; proprio/skill AdaRMS remains.

The objective is action + Bspline + 0.1 Top patch + 0.1 Wrist patch + 0.1 XYZ
with the default `spatial_loss_weight: 0.1`; the patch pair is not averaged.
XYZ validity remains independent of UV visibility. Held-start models average
start/current auxiliary losses. Plan length is 30 and the Expert split is 9+9.
Suffix vision reaches L10–L18, except diffaux3/4 `_last` variants use only L18;
current proprio still reaches all suffix layers. Preskill memory is prefix-only,
non-recursive and detached, with the BOS reader update fully masked.

The diffaux4 `_start` and `_start_last` variants supervise XYZ only from the current skill's
start images. Held-start models retain the existing average of start/current
Top and Wrist patch losses; current-preskill models retain current-only patch
losses and add a start-image encoding for the training XYZ loss. That extra pass
shares the encoder/readout weights, carries gradients, and never replaces the
Expert's current vision or detached previous-skill plan. XYZ is added AFTER
patch averaging with its full coefficient 0.1. The current-image XYZ head is
not evaluated, and W&B records XYZ under `train/start_agent_end_state/*` only.
Diffaux3/4 inference, including current-preskill variants, encodes the skill-start
observation once for the XYZ goal and caches it until the next skill occurrence.
`_start` does not select a different visual bridge layer or UV token route.
`_start_last` additionally restricts direct suffix vision to L18; the prefix L9
vision, all 18 Expert layers, suffix proprio and plan/preskill paths remain.

All diffaux3/4 variants use their OWN start-image XYZ prediction for the
Expert at inference: bottleneck query 2 for diffaux3, and the Gemma readout
token for diffaux4. Quantile-normalized XYZ is inverse-normalized to grounded
metres, with no training noise and no GT endpoint fallback. Held-start vision
reuses that same encoding. Current-preskill vision remains current, and its
previous-skill plan retains the previous skill's cached predicted XYZ. A new
occurrence refreshes the cache even when consecutive FSQ codes are equal;
episode reset clears it. `focus_source: auto | gt | predictor` still selects UV
independently. `pose_source` remains relevant to other pose consumers; it does
not replace these models' own XYZ with a GT or external Predictor goal.
Training objectives and checkpoint revisions are unchanged.

### Offline diffaux3/4 XYZ-head evaluation

`eval/submit_xyz_head_eval.sh` measures the deploy-time start-image XYZ head
without MuJoCo, action sampling, or a Terminator. The default
`eval/xyz_head_eval_config.yaml` selects two deterministic recorded episodes
for each of tasks 0--9, evaluates every canonical skill occurrence with GT
skill and GT focus UV, and compares the prediction with the recorded canonical
endpoint in episode-grounded metres.

```bash
cd lerobot/examples/libero/configs/train_skillVLA/PT/eval
./submit_xyz_head_eval.sh --dry-run
./submit_xyz_head_eval.sh
```

Each model writes `metrics.json`, CSV/JSONL predictions and an HTML report under
`eval/outputs_xyz_head/<output_name>/`. The HTML includes overall and per-task
mean/median/RMSE/P90 3D error, per-axis MAE, and every skill-start Top/Wrist
image. The fixed Top camera overlays GT XYZ as a green circle, predicted XYZ as
a red cross, and their pixel displacement as a yellow line. The evaluator calls
only `predict_expert_xyz`; the Expert action flow is never sampled.

The old 16 diffaux3 detach/attach/xyzw1 names are retired. New GT variants have
separate checkpoint revisions and must not load those predicted-XYZ weights as
if they were the same architecture. Existing diffaux2 models keep their original
predicted-XYZ behavior. Diffaux4 `_last` is available only for the learned Gemma
token + UV AdaRMS variants. No `_uvtoken_last` or `_xyzw1` variant is registered.

## Mode3 Predictor spatial targets

`mode3`, `mode3_preskill`, `mode3_preskill_wristX`,
`mode3_preskill_wristX_proprioX`, and `mode3_preskill_proprioX` support
`skill_predictor.spatial_target: xyz` **or** `uv`. `mode3` keeps Top/Wrist/proprio
inputs and the same zoom/crop, but has no previous-skill/BOS conditioning.
The `_preskill` variants retain previous-skill/BOS conditioning and their selected
wrist/proprio ablations.

* `xyz`: predict canonical skill-end XYZ minus the actual jittered skill-start
  XYZ. Add the measured start origin back at inference, as before.
* `uv`: predict **absolute** skill-end UV in the augmented Top image, never
  end-minus-start UV. Dataset build must provide `skill_focus_uv.npz` from
  `focus_uv: {enabled: true, camera: agentview}`. Canonical endpoint/temporal
  jitter semantics remain the same as the original UV dataset.

Both keep the 1.1x Top zoom plus per-sample uniform crop. For UV, one explicit
crop draw is shared by image pixels and labels; its transform also includes
the actual integer zoom dimensions, bilinear half-pixel geometry, and final
224 resize/padding. Finite targets outside the original image or sampled crop
are clamped to the nearest visible border and still contribute UV loss. Only
non-finite targets are masked; skill loss continues for **every** sample. W&B
reports `focus_uv_valid_fraction`, `focus_uv_original_valid_fraction`,
`focus_uv_clipped_supervision_fraction`, `focus_uv_crop_clipped_fraction`, and
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
  mode: [mode3, mode3_preskill]
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

## Terminator architectures sharing one GPU

The same independent-process sharing is available for PT Terminators:

```yaml
fsq_terminator:
  architecture: [Term_Token_3_both, Term_Token_4_both]
training:
  dataloader: {batch_size: 32, workers: 12, gpus: 1}
slurm: {gres: "gpu:1", cpus: 32, memory: 192G, time: "24:00:00"}
```

Run the normal `Terminator/submit_train.sh`: one GPU allocation launches every
listed architecture **concurrently**, with independent model/optimizer/W&B run,
architecture-specific output folder and automatic checkpoint resume. Three or
more names also work if the combined VRAM fits; these are not sequential jobs.
The dataset is staged once, not once per model. No parameters, losses or batches
are shared. Batch size and dataloader workers remain **per model**; Slurm CPU/RAM
are job totals, not automatically multiplied. Shared GPU compute can reduce each
model's speed, and the sample resources above are not a memory guarantee.

The parent logs show each worker's status; individual logs and exit codes are in
`Terminator/logs/terminator_group_<job id>/`. A failed model does not stop healthy
models; the job reports failure if any model fails. Cancelling stops all workers.
A scalar architecture keeps the original single-model/DDP path. Lists require
exactly one GPU and Terminator-only PT; standalone FT still inherits one model
from its warm-start checkpoint. Duplicate/unknown names and mixed Predictor +
Terminator lists are rejected before any job is submitted. A one-item list also
uses the ordinary single-model path. `SKILL_AUX_DRY_RUN=1 ./submit_train.sh` from
`Terminator/` shows every output and the resource request without submitting.

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

## UV_Bspline_skill independent start planners

Set `architecture.name` in `VSA/vsa_train_config.yaml` to one of:

| Name | Planner input | Fullskill readout / objective |
| --- | --- | --- |
| `UV_Bspline_skill_learnable` | 30 learned tokens | Direct 30×8D proprio MSE |
| `UV_Bspline_skill_learnable_wovis` | 30 learned tokens, no prefix vision | Direct 30×8D proprio MSE |
| `UV_Bspline_skill_denoising` | 30 independent noisy 8D proprio tokens | Direct velocity readout, proprio flow matching |
| `UV_Bspline_skill_denoising_none` | 30 independent GT-noised proprio tokens (training only) | One FM forward; no Action plan reader or rollout planner |
| `UV_Bspline_skill_denoising_none_wovis` | Same one-pass FM, no prefix vision | One FM forward; no Action plan reader or rollout planner |
| `UV_Bspline_skill_learnable_query` | 10 learned tokens (`chunk_size`) | Existing 30 queries + generator + proprio MSE |

All six use ordinary aux3 (no current/preskill or `_last` variant): 100-token
joint self-attention, Top/Wrist/XYZ auxiliary heads, shared 9+9 Expert, start
vision at L9 except `_wovis`, and current vision/proprio at L10–L18. No aux5
Gemma XYZ head or extra XYZ loss is added. All three auxiliary targets remain
clean and keep their original weights and start/current averaging.

`learnable`, `denoising_none` and their `_wovis` variants condition the prefix
Expert on **end UV + start XYZ**, never end XYZ. Training perturbs GT endpoint
XYZ by Uniform(-0.01,+0.01) metres per axis, projects it through calibrated
Top camera metadata and clips UV. The same UV conditions vision and Expert;
cached UV is not jittered directly. At inference, the selected `focus_source`
supplies UV; finite clipped coordinates are retained even if the original
projection was out of frame. No GT endpoint or predicted XYZ is fed to these
four Experts. Bottleneck XYZ prediction remains an auxiliary objective.
Their saved revisions end in `expert_uv_v2`, rejecting older XYZ-input weights
under the same name rather than silently reusing an incompatible adapter.

The unmodified `denoising` and `learnable_query` models keep GT end XYZ during
training and their own cached start-bottleneck XYZ prediction at inference.
In Stage-1 eval, UV-only models use `expert_xyz_source: auto` (resolves to
`native`) or `native`; override an inherited explicit `predicted`/`gt` setting.

`_wovis` disables the start-vision bridge in **both Planner and Action L1–L9**.
It leaves every late L10–L18 vision bridge and current proprio unchanged.
Start images still train the shared auxiliary heads; inference skips their
unused encoding. Planner conditions still include Skill, start XYZ and end
UV, so this removes direct visual-token input rather than all scene-dependent
information (UV may itself come from a predictor).

The planner runs a **separate prefix forward** from start observations and
skill/goal conditions (without the start images for `_wovis`). It never reads current observations, action noise or
action hidden states. Learned inputs use fixed time=1. Direct modes remove
the separate output query/generator, projecting 30 prefix states to plan width;
the query variant retains that generator. Action still denoises its own H-token
chunk. The four reader models read the generated 30-token plan before L10;
their shared prefix weights and plan latents receive Action gradients without
an extra detach. Input length differs between the two learned variants,
so their comparison does not isolate only the output-query mechanism.

During training, start-only planning and current-observation Action learning
both run on every batch. Loss is `Action + Plan + 0.1*(Top + Wrist + XYZ)`;
the three auxiliaries retain aux3's start/current averaging. `Plan` is MSE for
learned modes, or FM loss for denoising, using `bspline.weight` (default 1).
FM operates on quantile-normalized 30×8D proprio, not action commands. Its
teacher-noised GT is used only for the separate FM loss. In the denoising reader
model, Action reads a freely generated plan: training samples the planner using
`policy.num_inference_steps` (default 10), then re-encodes the clean state at
t=0. This costs more prefix forwards and memory than the one-pass learned
variants. `planner/flow_loss` or `planner/regression_loss` identifies the loss
in metrics; `bspline/mae_normalized` measures generated-plan error.

For the reader models, inference creates the plan once per skill occurrence, holds it across all
Action replans, updates only changed batch elements, and clears it on reset.
Repeated identical FSQ codes use the occurrence ID to distinguish skills.
The `denoising_none` models instead pass the H-token Action hidden directly
from L9 to L10 with no plan-reader update. Its fullskill branch does **one**
GT-noised FM forward per supervised batch and no iterative sampling or clean
re-encoding. Only the prefix weights/vision conditioning are shared; Action
gradients do not directly reach the fullskill-specific projection/velocity head.
Action-only/disabled-FT-plan training and normal inference skip this branch
entirely, with no plan cache. Fullskill video overlay is skipped for these models
(Input UV overlay remains available). Metrics omit generated-plan MAE rather
than presenting teacher-noised velocities as a clean plan.
PT warm-start allowlists, saved architecture revisions, Stage-1 eval/video
capture and standalone FT all use these names. `submit_train.sh <name>` also
works. Selecting a name does not submit a job by itself.

## GPU allocation in Stage-1 eval

The existing `eval_num_gpus` and `eval_max_workers_per_gpu` settings also apply
to multi-model comparisons and checkpoint sweeps. Work is split over **task x
panel** pairs, where one panel is one model/checkpoint combination. For example,
ten tasks and nine panels with `eval_num_gpus: 10` and
`eval_max_workers_per_gpu: 5` request ten one-GPU array jobs, each running five
independent evaluation processes and covering nine task/panel pairs in total.
GPU count is an upper bound; a smaller evaluation uses fewer GPUs. The per-GPU
process cap is 1–10. Inside an existing allocation, evaluation uses one GPU
and processes its assigned task/panel groups there.

Every worker keeps the full panel order for comparison videos and evaluates
only its assigned panels. Existing seeds, resume checks, panel directories,
merged success-rate tables/charts and side-by-side video layouts are preserved.
This distributes more independent work; throughput still depends on available
GPUs and the CPU/VRAM resources shared by each GPU's processes.

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
