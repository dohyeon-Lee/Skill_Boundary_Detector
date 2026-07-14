# Codex handoff: SkillVLA / FSQ redesign

Updated: 2026-07-14 (Asia/Seoul)

This document transfers the design context from the long Codex thread that was
run across `/scratch/mdorazi` and `/scratch2/mdorazi`. The active workspace is
now `/scratch2/mdorazi`.

## 0. 2026-07-14 continuation — read this first

The code and config are authoritative. The current branch is
`splitVLA_lora_ABC`, HEAD is `d188bd8` at this update. The worktree is
intentionally dirty with the Stage-1 eval/assets changes described below;
preserve them. Do not assume these facts remain true; run the checks in section
1 before editing.

### 0.1 Path/cache portability is fixed at Slurm entry points

`lerobot/examples/libero/configs/runtime_env.sh` is sourced by the active
train/eval/build sbatch scripts. It forces all transient storage below the
configured `${PROJECT_ROOT}` (currently `/scratch2/...`):

- Hugging Face: `${PROJECT_ROOT}/.cache/huggingface/{datasets,hub}`
- W&B cache/data: `${PROJECT_ROOT}/.cache/wandb/{cache,data}`
- Matplotlib: `${PROJECT_ROOT}/.cache/matplotlib`

This was necessary because FSQ job `1800407` crashed while `datasets` tried to
create a parquet lock under the old default
`/scratch/mdorazi/.cache/huggingface/datasets` (`Errno 28: No space left on
device`). Do not remove the runtime-env source lines or revert to `$HOME` cache
fallbacks. `.cache/` is deliberately gitignored.

### 0.2 Boundary construction has two selectable modes

`skillset_boundary_threshold_mode` selects the segmentation threshold:

```yaml
episode_mean  # legacy: each episode's SG-smoothed divergence mean
global_mean   # one mean over every SG-smoothed replanning point in all episodes
```

- Set it in `train_skills/build_data/build_data_config.yaml` to choose what
  `submit_build_data.sh` builds. At this update it is `global_mean`.
- A global skillset is isolated as
  `seg_{dp}_ck{checkpoint}_globalmean/skillset`; legacy episode-local data stays
  in `seg_{dp}_ck{checkpoint}/skillset`. Never mix or resume across them.
- `skillset_global_threshold_source: ""` means compute a fresh target-dataset
  global mean. Set it to a prior `global_boundary_threshold.json` to hold one
  taxonomy across a new task: the output is instead isolated as
  `seg_{dp}_ck{checkpoint}_globalref/skillset`, target curves are still
  collected, and cached-curve segmentation uses the referenced `global_mean`.
  The reference file is checked before Slurm submission and is recorded in the
  resulting `.complete` marker. Do not copy the JSON into the new output: its
  path is intentional provenance. Newly reduced JSONs also record DP and curve
  parameter provenance; older schema-v1 JSONs remain valid sources.
- `global_mean` is a 2-pass Slurm pipeline: (1) curve-only DP/VF array,
  (2) `compute_global_boundary_threshold.sbatch`, which writes
  `skillset/global_boundary_threshold.json`, and (3) cached-curve segmentation
  array. Stage 3 does **not** rerun DP/VF inference.
- `compute_global_boundary_threshold.py` verifies that it received one valid
  curve for every episode and computes a replanning-point-weighted mean of
  `sg_vals`, not an average of array/task means.
- Initial reducer job `1802399` failed because it used the Slurm spool path;
  `compute_global_boundary_threshold.sbatch` now anchors on `SLURM_SUBMIT_DIR`.
  The first global curve pass completed all 3921 curves successfully.

FSQ and DP-eval selection is now explicit too:

- `FSQ/fsq_config.yaml`: `skillset_boundary_threshold_mode`
- `skill_eval/eval_config.yaml`: `skillset_boundary_threshold_mode`

Those module values override the build-config fallback, intentionally allowing
an old/local and a new/global skillset to coexist. At this update both FSQ and
skill-eval configs are set to `episode_mean`, while build-data is set to
`global_mean`; change the module config intentionally before consuming a global
skillset. FSQ metadata records the mode used for training, and FSQ reconstruction
eval reuses that metadata so it cannot accidentally evaluate a global FSQ on a
local skillset.

### 0.3 Naming, W&B, checkpoints, and eval behavior

- A global FSQ run has `_global` immediately after `dp_tag`, e.g.
  `..._state_obs20_global_fsq555_...`. Local runs omit it. DP boundary-eval HTML
  similarly becomes `state_obs20_global_ck100000.html`.
- `FSQ.pt` is overwritten whenever `val/select` reaches a new minimum, not at a
  fixed cadence. `FSQ_epochNNNN.pt` is the exact-resume checkpoint and is saved
  every `fsq_checkpoint_every` epochs (currently 25).
- `val/select` defaults to the same weights as the train loss:
  `1.0*action + 0.1*progress + 0.1*termination`. It selects `FSQ.pt`; it does
  not affect the gradient.
- Standard W&B `train/*` and `val/*` use `optimizer_step` as x; duplicated
  `train_epoch/*` and `val_epoch/*` curves use epoch as x. `action_objective`
  equals `action` while `weighted_loss: false` (current default).
- New FSQ jobs log `train/` and `val/` codebook utilization and active-entry
  count per epoch. This is an on-device boolean accumulation of the already
  computed encoder indices; it has no extra encode/decode pass. Train coverage
  is approximate because weights change during the epoch; validation coverage
  is for the fixed validation split/current model.
- `end_recall` is a termination classification metric, not a model parameter:
  prediction is sigmoid(logit) >= `end_threshold` (0.5); target is >= 0.5. With
  current `end_target_sigma: 1.0`, the end and roughly one preceding frame are
  positive under that threshold.
- `fsq_eval.py` defaults to `decoder_scope: samples`, configured through
  `fsq_eval_decoder_scope`. It encodes all skills (exact codebook counts/top
  entries) but only live-decodes the plotted samples. `decoder_scope: all`
  restores full-skill metrics. In sample mode W&B metrics are under
  `decoder_sample/*`, with `decoder/evaluated_skills`; do not compare them as
  full-dataset metrics. The current eval YAML has max 10 entries × 10 samples.
- FSQ action reconstruction in eval uses 10 Euler flow/denoising steps per
  frame microbatch (`model.decode(..., num_steps=10)`). The image terminator is
  evaluated once after that action integration. The eval CLI must retain its
  `--image_key` argument; it was fixed after sample-mode first exposed the
  thumbnail path.

### 0.4 LIBERO MuJoCo assets must be committed with the repository

The required simulator assets live at
`tools/lerobot-libero/libero/libero/assets`. They are static files bundled with
the upstream `huggingface/lerobot-libero` repository (scene/object XML, meshes,
and texture PNGs); they are not generated by FSQ, SkillVLA data build, or eval.
LIBERO also resolves this directory directly relative to
`libero/envs/bddl_base_domain.py`, so a config path cannot compensate for
missing files in that tree.

The root `.gitignore` acquired a global `*.png` rule in commit `6cb270d`, before
`tools/lerobot-libero` was converted from a gitlink/submodule to ordinary
vendored files in commit `d8c2d5a`. Consequently the vendoring commit omitted
all 119 required asset PNGs, and a normal clone/pull on another server could
never restore them. `tools/lerobot-libero` is currently neither a submodule nor
a nested Git checkout (`.gitmodules` and `tools/lerobot-libero/.git` are both
absent). The root `.gitignore` now explicitly unignores the complete assets
tree:

```gitignore
!tools/lerobot-libero/libero/libero/assets/
!tools/lerobot-libero/libero/libero/assets/**
```

On 2026-07-14 the complete assets tree was copied from the intact
`/scratch/mdorazi/Skill_Boundary_Detector/.../assets` into the active
`/scratch2` workspace. Verification found 119 PNGs, zero checksum differences
against the intact source, and successful direct MuJoCo parsing of
`libero_kitchen_tabletop_base_style.xml`. The PNGs total about 209 MiB and the
largest single file is about 12.2 MiB, so they fit normal GitHub file limits;
they are intentionally normal Git files, not Git LFS. They are currently
untracked until the surrounding work is committed, so **include the restored
asset files and `.gitignore` exception in the next commit/push**.

### 0.5 Current running-job snapshot (ephemeral)

Latest local-W&B snapshot: 2026-07-14 16:20 KST. Six FSQ jobs were running. All used batch size 64;
local skillsets have 148 train updates/epoch (10,497 skills) and global
skillsets 143 (10,157 skills). This is only a status snapshot; inspect Slurm and
logs again on a new server/session.

| Job | Run | Latest observed progress | FSQ.pt best epoch |
|---|---|---:|---:|
| 1802357 | local zero-grounded `state_skill` | e125 / step 18,500 | 124 |
| 1802358 | local raw-state `state_skill` | e242 / step 35,816 | 167 |
| 1802359 | local raw-state `state` | e63 / step 9,324 | 62 |
| 1802766 | global raw-state `state_skill` | e143 / step 20,449 | 105 |
| 1802768 | global zero-grounded `state_skill` | e143 / step 20,449 | 107 |
| 1803656 | local zero-grounded `state` | e77 / step 11,396 | 70 |

For the currently queried global zero-grounded `state_skill` run
`libero_90_full_full_state_obs20_global_fsq555_dino_frozen_small_vsa_state_skill`
(job `1802768`): `FSQ.pt` is epoch 107, with best `val/select=0.140005`.
At epoch 143 its selection score is `0.140693`. Its best-checkpoint full
encoding (`outputs_filtered/FSQ/<run>/skill_latents.npz`) covers 10,157 skills:
98/125 codes used, 27 empty, and 20 nonempty codes occur only 1–19 times.
`fsq_snap_min_code_freq: 20` therefore retains 78 codes and treats 47 as
unsupported (27 empty + 20 rare). Support sizes at thresholds 1/5/10/20 are
98/90/86/78 respectively; use 10 as the less-aggressive ablation.

## 1. Repository state to start from

- Active repository root: `/scratch2/mdorazi/Skill_Boundary_Detector`
- Main code directory: `/scratch2/mdorazi/Skill_Boundary_Detector/lerobot`
- Branch: `splitVLA_lora_ABC`
- HEAD at this update: `d188bd8` (do not rely on this after transfer)
- The worktree is intentionally dirty with the changes documented in section
  0 and the Stage-1 eval section. Preserve all of them, including restored
  untracked LIBERO asset PNGs.

On the new filesystem, first verify:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

Do not discard user changes if the new copy is dirty. Read the current code and
compare it to this document before editing; the code is authoritative if it has
advanced past the commit above.

## 2. Project objective and terminology

The overall target is a task-agnostic SkillVLA:

- Stage 1 learns motion from a VSA (vision-skill-action) policy.
- Stage 2 adds VLM/language information to supply residual information that a
  discrete skill code cannot fully represent.
- The long-term goal is zero-shot transfer to a new task when a valid skill
  sequence is supplied.
- Therefore Stage 2 must avoid learning task shortcuts. The intended trainable
  path is the language-to-action bridge/LoRA path; the language understanding
  LLM and the Stage-1 VSA backbone are otherwise frozen.

Important distinction:

- The VSA action expert itself is image-free. Its inputs are current state,
  time/noisy action, and skill conditioning.
- Images are consumed by the condition/vision side and by the FSQ terminator.
- Do not conflate the VLM-side image contract with the VSA/condition-side image
  contract; they are intentionally different in the Stage-2 pipeline.

## 3. Current end-to-end pipeline

Conceptually:

```text
DP / boundary construction
  -> per-skill trajectory NPZ files
  -> joint FSQ training
       encoder: one pass per complete trajectory B
       action expert + terminator: M sampled timesteps per trajectory (B*M)
  -> encode all skills with the trained FSQ encoder
  -> build SkillVLA dataset
  -> Stage 1 VSA fine-tuning
  -> Stage 2 language residual/bridge fine-tuning
  -> rollout evaluation
```

The new FSQ format is version 3. Legacy decoder/fallback code was deliberately
removed rather than preserved.

Boundary construction is intentionally **state-only**:

- DP boundary checkpoints must have `_state` in their run name and assert
  `state_only` when loaded.
- The old DINO feature staging/precompute, visual-DP mode, and associated build
  config fields were removed from the active `train_skills` path.
- Standalone divider/eval code may still decode frames for visualizations, but
  images are not a DP/boundary-policy input.

## 4. FSQ v3 architecture

Primary implementation:

- `lerobot/examples/libero/FSQ.py`
- `lerobot/examples/libero/train_FSQ.py`

### 4.1 Trajectory encoder

- Training unit is a complete skill trajectory.
- The encoder runs once for each of the B trajectories.
- The source is full `observation.state` for the skill trajectory (LIBERO:
  six EEF pose dimensions plus two observed gripper-state dimensions).
- A B-spline maps the variable-length trajectory to a fixed number of control
  points, plus a length token.
- Transformer/query pooling produces the pre-quantized vector, followed by FSQ.
- Current FSQ levels are `[5, 5, 5]`, therefore 125 codes.

Encoder input is configurable:

```yaml
fsq_encoder_input_mode: zero_grounded  # zero_grounded | raw_state
```

- `zero_grounded`: subtract the first state from the pose dimensions; preserve
  the two gripper-state dimensions as absolute values.
- `raw_state`: retain the original absolute state trajectory.
- The selected convention is applied consistently to normalization statistics,
  spline control points, training, `encode_numpy`, `encode_index`, and FSQ eval.
- It is stored in the FSQ checkpoint and `action_stats.npz`.
- A resume attempt with a different mode fails explicitly.
- A raw-state run gets `_rawstate` in its run name to prevent accidental resume
  from a zero-grounded checkpoint. The historic zero-grounded name is preserved.

Research caveat: `raw_state` may improve reconstruction by encoding absolute
start/location information, but it can leak layout/task information and weaken
task-agnostic transfer. Treat it as an ablation.

### 4.2 Action reconstructor / VSA flow expert

- The old FSQ dense reconstructor is gone.
- Reconstruction uses the same image-free Gemma/AdaRMS VSA flow expert format
  that Stage 1 later warm-starts.
- Inputs are current state and `z_q`, plus the normal flow-matching time/noisy
  action inputs.
- Output is a Stage-1-compatible action chunk.
- This expert is deliberately architecturally identical to the downstream
  Stage-1 action expert so weights transfer directly.

Two skill-conditioning modes must remain supported:

```yaml
fsq_state_cond_mode: state       # state | state_skill
```

- `state`: AdaRMS condition is time + state; `z_q` remains an attended prefix
  token. This is the lighter skill influence and leaves more room for Stage-2
  language modulation.
- `state_skill`: AdaRMS condition is time + state + `z_q`; there are no skill
  prefix tokens. This is the strongest skill influence and is useful if FSQ
  training otherwise ignores the skill.

The action expert remains Gemma even when the terminator architecture is
`small`. It is controlled independently by:

```yaml
fsq_action_expert_variant: gemma_300m
```

### 4.3 Terminator

Terminator inputs and outputs:

- Inputs: third-person raw image, wrist raw image, current state, and `z_q`.
- Both camera streams are always used; the old per-camera enable flags were
  removed.
- State and skill condition the query branch through AdaRMS-style modulation.
- Outputs are one progress value and one termination logit, generated from two
  separate learned query tokens.

Attention direction is intentional:

- Image tokens may read image tokens only.
- Progress query reads image tokens and itself.
- Termination query reads image tokens and itself.
- The two queries do not read one another.
- Image tokens never read the learned queries.

There is no DINO 8x8 pooling and no camera positional embedding. The live
vision frontend retains unpooled tokens and matches the Stage-1 condition-side
token contract as closely as possible:

- DINO: CLS plus all patch tokens, dropping register tokens.
- SigLIP: all patch tokens.
- The same vision tower processes both cameras and both streams share one
  `image_proj`.

Terminator architecture is selectable:

```yaml
fsq_terminator_arch: small       # small | cond
fsq_cond_encoder_variant: gemma_300m
```

- `small`: lightweight `QueryTerminatorLayer` stack. It does **not** construct
  a terminator Gemma; `cond_encoder` is `None`.
- `cond`: plain-RMS Gemma compatible with the Stage-1 condition encoder. The
  state+skill modulation remains external on the query input/output so image
  tokens remain image-only and the Gemma state dict remains transferable.
- `fsq_cond_encoder_variant` is ignored when `fsq_terminator_arch: small`; it is
  present only so one config can switch between the two architectures.

Vision is separately selectable:

```yaml
fsq_vision_backbone: dino        # dino | siglip
fsq_freeze_vision_encoder: true  # false = end-to-end vision fine-tuning
```

### 4.4 FSQ training batch semantics

For one optimization step:

- `B = fsq_batch_size` complete skill trajectories are selected.
- The trajectory encoder processes those B trajectories once.
- `M = fsq_samples_per_skill` timesteps are randomly sampled per trajectory for
  the action expert and terminator.
- Thus the decoder-side effective sample count is `B*M`, but all M samples from
  one trajectory share the same encoder result and FSQ code.
- Training M sampling is random without replacement when the skill is long
  enough and with replacement otherwise.
- Validation uses deterministic evenly spaced samples.
- The DataLoader shuffles trajectories across epochs, so the dataset is covered
  in the normal epoch sense.

M does not need its own deterministic coverage schedule; random sampling was
considered acceptable. Increasing M improves within-trajectory coverage but
raises live vision compute and memory roughly proportionally.

### 4.5 Current image/data loading

There is no disk DINO-token precompute and no episode-wide in-RAM warm pass in
the current FSQ training path.

- Each worker lazily owns a LeRobotDataset instance.
- Only the M selected raw frames are requested for each skill.
- Both third-person and wrist frames are decoded.
- DINO or SigLIP runs live on the resulting B*M frames.
- The earlier DINO 8x8 disk/cache contract was removed from the core FSQ path.

This is preferable for the new sampled-M training because an episode-wide warm
pass would compute/store many frames that the update never uses. It avoids the
old roughly 27 GB per-camera in-RAM token cache. Repeated video random access and
live vision compute are the tradeoffs.

Current loader choices:

- train workers: configured `fsq_num_workers`, persistent, prefetch factor 1.
- validation workers: at most 2 and non-persistent, to avoid simultaneously
  retaining a second large worker pool.
- `_sample_images` now batches the M requested timestamps into one
  `decode_video_frames` call per camera (two calls per skill), rather than
  issuing 2*M random decoder requests. It intentionally uses the worker-local
  LeRobot reader's private metadata/video helpers; a raw-dataset smoke test
  confirmed batched decode matches individual-frame decode exactly.
- CUDA DataLoaders use pinned memory and the existing non-blocking transfer.
  These changes reduce CPU/video starvation but do not eliminate it; PyAV
  random decode remains the dominant source of GPU-utilization/power sawtooth.
- TorchCodec was tested but could not load on this server because its FFmpeg
  shared libraries/PyTorch build were incompatible. Do not enable it without a
  working decoder smoke test.
- Episode-aware sampling/LRU raw-frame cache is not implemented. It could
  reduce repeated decoding further but trades off batch diversity and can add
  task/episode correlation. The batched-M decode above is the safe baseline.

## 5. Current FSQ config and naming

Main config:

`lerobot/examples/libero/configs/train_skills/FSQ/fsq_config.yaml`

At handoff time the important values are:

```yaml
fsq_levels: [5, 5, 5]
fsq_encoder_input_mode: zero_grounded
skillset_boundary_threshold_mode: episode_mean  # explicit local/global selector

fsq_terminator_arch: small
fsq_vision_backbone: dino
fsq_freeze_vision_encoder: true
fsq_dino_model_path: models/dinov3-vitl16  # resolved relative to project_root
fsq_dino_image_size: 224
fsq_siglip_image_size: 224
fsq_cond_encoder_variant: gemma_300m  # only used for arch=cond

fsq_batch_size: 64
fsq_num_workers: 8
fsq_num_epochs: 500
fsq_checkpoint_every: 25
fsq_chunk_size: 10
fsq_samples_per_skill: 5
fsq_action_expert_variant: gemma_300m
fsq_state_cond_mode: state
fsq_expert_dtype: bfloat16

fsq_encoder_lr: 3.0e-4
fsq_terminator_lr: 3.0e-4
fsq_expert_lr: 2.5e-5

fsq_action_loss_weight: 1.0
fsq_progress_loss_weight: 0.1
fsq_end_loss_weight: 0.1
fsq_end_target_sigma: 1.0
weighted_loss: false
```

The run-name template is:

```text
{target_dataset}_{dp_tag}_{fsq_tag}_{fsq_vision_tag}_{fsq_terminator_tag}
{fsq_encoder_input_suffix}{fsq_state_cond_suffix}{weighted_suffix}{fsq_exp_suffix}
```

Without the visual line break, the current default resolves to approximately:

```text
libero_90_full_full_state_obs20_fsq555_dino_frozen_small_vsa_state
```

Tags:

- `fsq_vision_tag`: e.g. `dino_frozen`, `dino_tuned`, `siglip_frozen`, or
  `siglip_tuned`.
- `fsq_encoder_input_suffix`: empty for zero-grounded; `_rawstate` for raw state.
- `fsq_encoder_input_mode` additionally supports `optimal`: it keeps the
  zero-grounded spline trajectory but adds exactly one learned encoder token
  projected from the absolute EEF pose at the skill start. Its run-name suffix
  is `_optimal`; it is a new encoder architecture and cannot resume a
  zero-grounded/raw checkpoint. The current implementation is intentionally
  single-arm: it excludes LIBERO's trailing two gripper-state dimensions from
  the start-pose token. This same preprocessing is used by train,
  `encode_FSQ_skills.py`, and `fsq_eval.py`.
- `fsq_state_cond_suffix`: `_vsa_state` or `_vsa_state_skill`.
- `weighted_suffix`: empty when `weighted_loss: false`; `_weighted` when true.
- `dp_tag`: adds `_global` immediately after the DP tag for a global-mean
  skillset, so local and global FSQ outputs cannot collide.

The Slurm script records architecture, vision, freeze state, encoder input mode,
and `skillset_boundary_threshold_mode` in `fsq_meta.json`.

Project-relative model paths are resolved in `train_skills_config.py`; use those
instead of server-specific absolute `/scratch*` paths.

## 6. Stage-1 warm start from FSQ

Implementation:

- `lerobot/src/lerobot/policies/skill_expert/modeling_skill_expert.py`

Behavior:

- The FSQ action expert always warm-starts the Stage-1 action expert.
- If FSQ terminator architecture is `small`, no terminator backbone is
  transferred into the Stage-1 condition encoder.
- If it is `cond`, the loader transfers the compatible `cond_encoder`,
  `image_proj`, and active vision tower state.
- Learned progress/termination queries, their heads, and query-only state/skill
  modulation are deliberately excluded from the condition warm start.
- Loading is strict/fail-fast for architecture mismatches.

Critical compatibility warning:

- The current FSQ default DINO path is `dinov3-vits16`.
- Some Stage-1 configs previously used DINO ViT-L/16.
- This is fine for `small`, because no terminator-to-cond transfer occurs.
- For `cond`, Stage 1 must use the same vision architecture/model dimensions,
  image size, and condition variant, or the warm start will fail on tensor
  shapes. Do not weaken this check merely to make loading pass.

When an in-policy terminator is trained downstream, code re-freezes its vision
tower after broad `requires_grad_(True)` calls so a frozen FSQ vision encoder
does not accidentally become trainable.

## 7. Stage-2 context and earlier evaluation findings

Stage 2 was designed to learn only the language-to-action residual path while
keeping the language understanding backbone and Stage-1 VSA frozen. Several
Stage-2 jobs were compared using W&B.

Observed training-loss behavior:

- Flow loss dropped to roughly 0.08-0.10 within about 500-1000 steps.
- Larger historical runs also remained near roughly 0.085-0.093 even after
  tens of thousands of steps.
- This is consistent with a conditional-variance/multimodality floor, especially
  because many motions alias into 125 skill codes.
- Therefore a flat train flow loss is not evidence that language is being used
  correctly. Rollout success and controlled ablations are the decisive metrics.

Important evaluation concern:

- In partial rollout results, `full` often looked worse than VSA-only.
- Expert LoRA disabled appeared qualitatively better than expert LoRA enabled in
  some runs.
- Quantitative comparison should separate:
  - `full`
  - `drop_vlm`
  - `VSA`
  - `stage1`
- This distinguishes language-path degradation from bridge/expert-LoRA drift.

VSA vs Stage-1 should be open-loop equivalent when all Stage-2 additions are
disabled and the same state, skill, noise, and weights are used. Closed-loop
rollouts can still diverge from noise and chaotic feedback, but large success
gaps need an explicit open-loop forward equivalence test.

The Stage-2 equivalence utility was updated previously to account for expert
LoRA and transition timing. Relevant path:

`lerobot/examples/libero/configs/train_skillVLA/stage2_eval/tools/verify_vsa_stage1_equiv.py`

Evaluation transition semantics should remain:

1. Observe the current post-action environment frame/state.
2. Let the terminator update progress/termination and decide whether to
   transition the skill.
3. Generate the next action under the resulting active skill.
4. Step the environment and repeat.

Do not accidentally generate an extra action under a skill that the current
observation has already terminated.

## 8. Proposed but not yet established as implemented

The leading Stage-2 improvement idea was a hard-residual sampler/weighting
scheme:

- Compute frozen-VSA error per sample.
- Upweight or oversample samples where the frozen VSA has large residual error,
  especially cross-skill transitions.
- Apply this primarily to connected/language-visible batches.
- Keep task sampling balanced so the sampler cannot create a task-frequency
  shortcut.

This idea addresses language neglect by increasing the share of examples for
which the skill alone is insufficient. It should be preferred over merely
raising LoRA rank or unfreezing vision.

Do **not** assume this sampler is already implemented. Search the new tree and
verify first. Also do not add teacher/distillation losses without a new design
decision: the user objected that action supervision already trained Stage 1 and
that reliably labeling language-unnecessary samples is difficult.

## 9. Verification already performed for FSQ v3

Before the handoff, the following checks passed on the old workspace:

- Python compilation for the changed FSQ/config/eval modules.
- `bash -n` for the FSQ Slurm script.
- `git diff --check`.
- Actual raw LeRobot single-process frame loading.
- Actual small+DINO terminator forward:
  - frozen vision parameters had no gradients;
  - progress/termination shapes were correct;
  - image-to-query attention was blocked;
  - query-to-image attention was allowed;
  - progress-query to termination-query attention was blocked.
- Cond terminator state dict matched the Stage-1 plain-RMS condition Gemma for
  the transferable component.
- Config generation was checked for small/cond and DINO/SigLIP combinations.
- The zero-grounded/raw-state spline test verified:
  - zero-grounded first six pose values become zero;
  - the two gripper-state values remain absolute;
  - raw-state mode leaves the full trajectory unchanged.
- Zero-grounded output naming stayed backward compatible.
- Raw-state config emitted `_rawstate` in the output folder.

Continuation checks completed on `/scratch2`:

- Python compile and shell syntax checks for the state-only build, global
  threshold reducer, FSQ training/eval config emitters, and runtime-env script.
- `global_mean` resolver output was checked for both global and local paths.
- The global reducer verified all 3921 staged curves before writing its mean.
- Legacy `episode_mean` peak selection was compared against the explicit
  threshold implementation across randomized signals and matched exactly.
- FSQ eval sample scope was smoke-tested through the decoder progress bar:
  sample mode decoded 9 microbatches in about 19 seconds instead of the old
  full-eval ~8700 microbatches. The subsequent missing `--image_key` parser
  argument was fixed.

A local multi-worker DataLoader smoke test could not run in the previous sandbox
because multiprocessing resource-sharing sockets were denied. This was a
sandbox restriction, not a demonstrated code failure. A short Slurm smoke run
on the new server is still recommended before launching a long FSQ job.

Suggested new-server smoke checks:

```bash
python -m py_compile \
  lerobot/examples/libero/FSQ.py \
  lerobot/examples/libero/train_FSQ.py \
  lerobot/examples/libero/fsq_eval.py \
  lerobot/examples/libero/configs/train_skills/src/train_skills_config.py

bash -n lerobot/examples/libero/configs/train_skills/FSQ/src/train_fsq.sbatch

python lerobot/examples/libero/configs/train_skills/src/train_skills_config.py \
  --config lerobot/examples/libero/configs/train_skills/FSQ/fsq_config.yaml
```

Then submit a short, low-epoch Slurm run and verify:

- raw frame loading from both cameras;
- B*M tensor shapes;
- DINO/SigLIP model paths;
- GPU memory;
- worker CPU RAM;
- W&B run name/config;
- checkpoint save/load and exact resume.

## 10. Important files

FSQ core and training:

- `lerobot/examples/libero/FSQ.py`
- `lerobot/examples/libero/train_FSQ.py`
- `lerobot/examples/libero/encode_FSQ_skills.py`
- `lerobot/examples/libero/fsq_eval.py`
- `lerobot/examples/libero/decoder_eval.py`

FSQ config/Slurm:

- `lerobot/examples/libero/configs/train_skills/FSQ/fsq_config.yaml`
- `lerobot/examples/libero/configs/train_skills/FSQ/src/train_fsq.sbatch`
- `lerobot/examples/libero/configs/train_skills/FSQ/submit_train_fsq.sh`
- `lerobot/examples/libero/configs/train_skills/src/train_skills_config.py`

State-only boundary build / global-threshold implementation:

- `lerobot/examples/libero/build_skill_dataset.py`
- `lerobot/examples/libero/skill_divider.py`
- `lerobot/examples/libero/configs/train_skills/build_data/build_data_config.yaml`
- `lerobot/examples/libero/configs/train_skills/build_data/submit_build_data.sh`
- `lerobot/examples/libero/configs/train_skills/build_data/src/build_skillset.sbatch`
- `lerobot/examples/libero/configs/train_skills/build_data/src/compute_global_boundary_threshold.py`
- `lerobot/examples/libero/configs/train_skills/build_data/src/compute_global_boundary_threshold.sbatch`
- `lerobot/examples/libero/configs/runtime_env.sh`

SkillVLA data construction:

- `lerobot/examples/libero/configs/train_skillVLA/build_data/`
- `lerobot/examples/libero/configs/train_skillVLA/build_data_eval/`

`build_data/train_skillVLA_config.yaml` has an explicit
`skillvla_data_mode: pt | ft | ft_own`. It is added to the final `run_tag`, so
PT and FT data builds cannot silently share final artifacts. All three construct
global-boundary skillsets: `pt` reduces the PT curves; `ft` reuses the one
matching PT `global_boundary_threshold.json` and writes an isolated
`seg_*_globalref`; `ft_own` reduces the FT source curves into its own
`seg_*_globalmean` for the new-motion ablation. There is no user-facing
`fsq_snap_reference`: with snapping enabled, `pt` automatically uses its own
raw code distribution (`self`) for vocabulary pruning, while `ft` automatically
finds the one matching PT build's `skill_latents.npz` (same FSQ checkpoint and
snap threshold). The same PT-code reference applies to `ft_own`. FT modes fail
before submission if that PT reference or (for `ft`) PT threshold is absent or
ambiguous, rather than silently making snapping a no-op.

The SkillVLA build now really executes those threshold semantics in both
`build_data/submit_build_all.sh` and `build_data/src/submit_build_skillset.sh`:
curve-only array → reducer or PT-reference dependency → cached-curve segment
array → verification. The VLA-local reducer wrapper is
`build_data/src/compute_global_boundary_threshold.sbatch`; it invokes the shared
Python reducer with VLA-resolved paths. Do not restore the old one-pass,
episode-mean invocation in `build_skillset.sbatch` / `verify_skillset.sbatch`.

### Stage-1 eval can run the FSQ action expert before Stage-1 training

`stage1_eval` now accepts heterogeneous `models` entries:

```yaml
- {kind: fsq_expert, run_tag: <SkillVLA data run>, advance_mode: gt, label: FSQ_GT}
- {kind: fsq_expert, run_tag: <same run>, advance_mode: terminator, label: FSQ_TERM}
- {kind: stage1, model_dir: <trained S1 run>, checkpoint: "015000",
   advance_mode: gt, label: Stage1_GT}
```

The `fsq_expert` backend (`stage1_eval/src/fsq_expert_policy.py`) loads only
`action_expert.*` plus FSQ/state/action metadata from the data run's `FSQ.pt`.
It does not construct policy vision or cond modules. It maps the injected GT
code back to normalized FSQ grid coordinates and uses the unchanged Stage-1
quantile pre/post processors, so actions are returned in LIBERO units. The raw
FSQ terminator is loaded separately only for skill timing/progress traces.
`advance_mode` is per panel: `gt` uses demo duration; `terminator` uses the
trained FSQ progress/end signals. FSQ-only panels and later trained Stage-1
panels reuse the existing episode-exact alignment and labelled video stitcher.

GT injection timing was audited against the current Stage-2 implementation and
is already identical: evaluate the old skill's boundary on current `obs_t`,
clear its queued actions if it ends, inject the next GT code on that same
`obs_t`, then generate the new-skill action. `set_forced_skill_token_sequences`
also runs before rollout reset and retains the sequence across that reset. A
mock timing smoke test covered both GT and terminator transitions, and the real
FSQ expert-only loader produced `(1, 10, 7)` without image/cond modules.

The checked-out default eval YAML currently compares the downloaded
`FSQ555_dino_frozen_small_vsa_state_skill_125_pt_snap10` in GT and terminator
modes. The required
`skillvla_dataset/libero_90_full_full/eval_init_states.npz` now exists (built
2026-07-14). If it is missing after transfer, rebuild it with
`stage1_eval/oracle_matching/run.sh libero_90_full_full` before submitting.

Stage-1 `eval_num_gpus > 1` submission now uses one Slurm job array, matching
Stage-2 naming/management: scheduler entries are `JOBID_0`, `JOBID_1`, ... and
`scancel JOBID` cancels the whole evaluation. `submit_eval.sh` exports a frozen
`EVAL_FANOUT` table; each `eval.sbatch` array element applies its task chunk and
`TASK_TAG` after loading the config snapshot. Per-chunk summaries, W&B names,
scratch folders, and log files remain disjoint (`logs/S1eval_%A_%a.*`).

Array job `1809318_[0-9]` confirmed that the fanout itself works, but all ten
elements failed before policy evaluation while MuJoCo opened the LIBERO scene:
the vendored assets tree had zero PNG files, producing `ValueError: Error
opening file .../assets/...png`. This was not an FSQ/Stage-1/model-path failure.
The full assets tree has since been restored and the exact kitchen XML now
parses successfully, but no Stage-1 eval result was produced by `1809318`; a
fresh `submit_eval.sh` run is still required. The LeRobot parser warning `No
pretrained path ... random weights` is expected for the FSQ-expert-only primary
backend because `run_eval.py` replaces the parser placeholder with the FSQ
expert context; it was not the crash cause.

Stage 1 / Stage 2 / FT:

- `lerobot/src/lerobot/policies/skill_expert/modeling_skill_expert.py`
- `lerobot/src/lerobot/policies/skill_expert/configuration_skill_expert.py`
- `lerobot/src/lerobot/policies/skillVLA/modeling_skillVLA.py`
- `lerobot/examples/libero/configs/train_skillVLA/stage1/`
- `lerobot/examples/libero/configs/train_skillVLA/stage2/`
- `lerobot/examples/libero/configs/train_skillVLA/FT/`
- `lerobot/examples/libero/configs/train_skillVLA/stage1_eval/`
- `lerobot/examples/libero/configs/train_skillVLA/stage2_eval/`
- `lerobot/examples/libero/configs/train_skillVLA/FT_eval/`

Obsolete legacy utilities deleted in the redesign include:

- `lerobot/examples/libero/encode_skills.py`
- `lerobot/examples/libero/eval_latent_space.py`
- `lerobot/examples/libero/fsq_zswap_eval.py`

Do not recreate the old fallback architecture unless explicitly requested.

## 11. First instruction for the next Codex session

Use this prompt in the `/scratch2/mdorazi` workspace:

```text
Read CODEX_HANDOFF.md completely. Verify the current branch, commit, worktree,
and all server-specific paths. Then inspect the referenced FSQ/Stage1/Stage2
files and report any mismatch between the handoff and the actual code before
making changes. Preserve unrelated user edits and do not restore deleted legacy
fallbacks.
```
