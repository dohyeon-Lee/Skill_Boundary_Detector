# FSQ_new Design and Implementation Handoff

Updated: 2026-07-21 (Asia/Seoul)

This document transfers the current `FSQ_new` design and implementation state.
It is deliberately separate from the older repository-wide `CODEX_HANDOFF.md`.
Read the current code before editing; this snapshot describes branch `skillVLA`
at commit `344b81f4a90b4302d25401ee63858fa7f5e97b08`.

## 0. Read This First

`FSQ_new` is an independent experiment. The original implementation remains at:

```text
lerobot/examples/libero/FSQ.py
lerobot/examples/libero/train_FSQ.py
lerobot/examples/libero/configs/train_skills/FSQ/
```

Do not replace or modify those files while continuing this experiment. The new
implementation lives entirely under:

```text
lerobot/examples/libero/configs/train_skills/FSQ_new/
```

Current status:

- The architecture, dataset path, A/B/C forward passes, losses, YAML resolver,
  Slurm submission, checkpointing, and W&B metrics are implemented.
- Python compilation, CLI construction, config resolution, and Bash syntax have
  been checked.
- A real model forward/backward with DINO + the 300M expert has **not** yet been
  run. Do a one-batch GPU smoke test before submitting a long job.
- Existing FSQ reconstruction eval and Stage-0 do **not** yet consume the new
  image/goal context path. That downstream integration remains future work.
- A/B/C are three regimes evaluated for every trajectory in every batch. There
  is no separate "ABC mode" flag and one must not be added.

On a new server, start with:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

Preserve unrelated user changes. The code is authoritative if it has advanced
past the commit recorded above.

## 1. Objective

Earlier ablations showed that an expert trained only by FSQ motion
reconstruction was less plastic than the original pi05 expert when Stage-0 later
introduced image/language information. Weakening skill conditioning alone would
undermine the purpose of the skill code.

`FSQ_new` therefore exposes one shared expert to three information regimes during
FSQ training:

```text
A: state + skill
B: state + skill + current image
C: state + skill + current image + endpoint goal image
```

The endpoint image is a proxy for language during FSQ training. The long-term
Stage-0 plan is to replace modality-specific context models while preserving the
fixed context-token interface read by the expert.

The intended capability ordering is:

```text
loss(C) < loss(B) < loss(A)
```

The FSQ codebook itself must still be determined by the image-free A action
path. B/C train the shared expert and context machinery, but their skill vectors
are detached from the FSQ encoder.

## 2. Files

```text
FSQ_new/fsq_new_config.yaml
  User-facing experiment configuration.

FSQ_new/submit_train_fsq_new.sh
  Resolves paths, snapshots YAML, and submits Slurm.

FSQ_new/src/train_fsq_new.sbatch
  Slurm runtime entry point. Writes outputs below outputs_filtered/FSQ_new/.

FSQ_new/src/resolve_fsq_new_config.py
  Resolves the nested context/conditioning/loss YAML fields into shell exports.

FSQ_new/src/train_FSQ_new.py
  CLI, skill NPZ loading, normalization stats, config construction, and training
  entry point.

FSQ_new/src/FSQ_new.py
  FSQ encoder, vision/terminator, context resamplers, expert, data loader, loss,
  optimization, checkpointing, and W&B logging.
```

The implementation began as an independent copy of FSQ v3 and was then changed
locally. Do not import the original `examples/libero/FSQ.py` into this path as a
shortcut; that would couple the experiment back to the production FSQ code.

## 3. Architecture

### 3.1 FSQ trajectory encoder

The trajectory encoder remains the FSQ v3 spline encoder:

- One complete skill trajectory produces one quantized skill vector `z_q`.
- The current YAML uses `fsq_levels: [3, 3, 3]` and
  `fsq_encoder_input_mode: optimal`.
- `optimal` uses the zero-grounded trajectory plus an absolute start-EEF token.
- The expert receives the normalized quantized vector.

### 3.2 Expert initialization

Both original FSQ and `FSQ_new` warm-start their expert from
`models/pi05_base/model.safetensors` when starting a fresh run.

Loaded from pi05:

```text
gemma_expert.* excluding lm_head
action_in_proj.*
action_out_proj.*
time_mlp_in.* / time_mlp_out.*
```

Newly initialized:

```text
state_proj.*
skill_proj.*
context_gates.*
image_context.*
goal_context.*
FSQ encoder and terminator-specific heads
```

The expert is not frozen during FSQ training. It is optimized with
`fsq_expert_lr` (currently `2.5e-5`). Exact resume loads the periodic FSQ
checkpoint instead of warm-starting pi05 again.

### 3.3 State, time, and skill conditioning

Current YAML uses:

```yaml
fsq_state_cond_mode: broadcast
```

- Time and state are summed and passed as the expert AdaRMS condition.
- The projected skill vector is added to the normalized hidden tokens before
  Q/K/V projection in every expert layer.
- `skill_scale` multiplies this broadcast vector per regime.

This is not the old `state_skill` mode where the skill enters AdaRMS.

### 3.4 Shared image backbone

The terminator owns the vision encoder and exposes `image_features()` so the
same encoder is reused by the context path.

- Current and endpoint images both use third-person + wrist views.
- The current config uses DINO at `models/dinov3-vitl16`.
- `fsq_freeze_vision_encoder: true` freezes DINO by default.
- Current and goal features share DINO, but use separate context resamplers.

If vision is later unfrozen, terminator and B/C context gradients update the
same shared vision backbone. This has not yet been smoke-tested.

### 3.5 Context resamplers

There are two independent `ContextResampler` instances:

```text
current DINO tokens -> image_context -> fixed context tokens
goal DINO tokens    -> goal_context  -> fixed context tokens
```

Current defaults:

```yaml
context: {queries: 16, layers: 4, heads: 8, dropout: 0.0}
```

Each layer performs:

```text
learned-query self-attention
source cross-attention
MLP
```

Third and wrist source tokens receive separate learned view embeddings. The
resampler projects the DINO width into the pi05 expert width and emits
`[batch, 16, expert_width]` final context tokens.

### 3.6 Expert reads context

Context is not inserted into the expert input sequence. In every expert layer:

1. The ordinary expert self-attention is computed.
2. Action-token Q reads image context K/V in a separate attention softmax.
3. Action-token Q reads goal context K/V in another separate attention softmax.
4. Scaled image/goal updates are summed and added to self-attention output.
5. The existing expert output projection and gated residual path are used.

The expert's existing Q/K/V/O projections are shared for self, image-context,
and goal-context attention. No new per-modality Q/K/V/O adapter was added.

There is one zero-initialized scalar `context_gate_l` per expert layer:

```text
context residual *= tanh(context_gate_l)
```

The gate is shared by image and goal at that layer; regime-specific image/goal
scales are applied before it. Only action chunk tokens read context. Context Q/K
currently do not use positional RoPE.

Because this is a custom layer loop, the first GPU smoke test must verify:

- tensor shapes through eager attention;
- BF16 dtype compatibility;
- finite gradients for context gates/resamplers;
- numerical A-path parity with the context-free expert at zero context gate.

## 4. Regimes and Gradient Routing

Current configuration:

```yaml
conditioning:
  A: {skill: 1.0, image: 0.0, goal: 0.0}
  B: {skill: 0.5, image: 1.0, goal: 0.0}
  C: {skill: 0.5, image: 0.5, goal: 1.0}
```

All regimes use the same shared expert and are evaluated on the same batch.
They also use the same GT action, sampled flow time, and noise, making their
per-sample losses directly comparable.

Gradient routing:

```text
A action loss:
  FSQ encoder/codebook + shared expert

B/C action loss:
  detached z + shared expert + context resampler
  (+ shared vision encoder only if it is unfrozen)

progress/termination loss:
  FSQ encoder + terminator once per batch
```

B/C do not have separate experts. Because they update the shared expert, they
can indirectly change A behavior; the direct A loss anchors that behavior. This
tradeoff was accepted for the first implementation.

## 5. Image Data Flow

The existing skill NPZ files are sufficient. No skill dataset rebuild is needed.
They must contain at least:

```text
actions, states, episode_id, skill_index, frame_start, frame_end
```

The raw LeRobot dataset with `videos/` and `meta/` must remain available.

For each trajectory, the loader:

- samples `M=fsq_samples_per_skill` current timesteps;
- reads current third+wrist frames from the raw videos;
- reads one endpoint third+wrist frame;
- decodes the sorted current frames and endpoint in one request per camera;
- repeats the single goal context over the M sampled action targets.

The current endpoint index is:

```text
dataset episode start + skill frame_start + skill length - 1
```

It is the final frame included in the current skill, not the first frame after
the boundary. Images are not copied into NPZ files, so storage does not grow,
but online video decode cost increases.

## 6. Losses

### 6.1 Direct flow loss

For each regime, flow-matching residual MSE is averaged over chunk steps/action
dimensions, then over M samples and trajectories:

```text
LA, LB, LC
Ldirect = (wA*LA + wB*LB + wC*LC) / (wA + wB + wC)
```

Current direct weights are all `1.0`.

### 6.2 Optional A-only progress weighting

Current YAML deliberately has:

```yaml
weighted_loss: false
weighted_loss_end_weight: 2.0
```

If enabled, only A action samples are weighted from `1.0` at skill start to
`weighted_loss_end_weight` at skill end. B/C direct loss, ranking, wrong-goal,
progress, and termination remain uniformly weighted.

`train/action_A` is always plain A MSE. `train/objective_A` is the possibly
weighted A objective.

### 6.3 Adjacent ranking

```text
R_AB = relu(LB - (1-rank_margin) * stopgrad(LA))
R_BC = relu(LC - (1-rank_margin) * stopgrad(LB))
R = R_AB + R_BC
```

With `relative_margin: 0.05`, B is asked to beat A by 5%, and C to beat B by
5%. The predecessor is detached in each ranking term so ranking does not
directly raise the baseline loss.

`ranking.weight: 0.1` multiplies `R` in the final action objective. It controls
gradient strength, not the required percentage.

### 6.4 Wrong-goal negative

The current implementation uses an in-batch cyclic shift:

```python
wrong_goal = goal_context_per_skill.roll(shifts=1, dims=0)
```

Training batches are shuffled, so this is effectively another random skill's
goal and never the same sample when batch size is greater than one. It does not
currently require the negative to have the same FSQ code.

```text
R_wrong = relu((1 + wrong_margin) * LC - Lwrong_goal)
```

With margin `0.05`, the wrong goal must produce at least 5% more action loss
than the correct goal. Unlike adjacent ranking, both correct-C and wrong-goal
graphs currently receive gradient from this term.

An unresolved possible improvement is:

```text
same-code in-batch negative first -> random in-batch fallback
```

This would better force endpoint information to resolve ambiguity left by the
skill code, but can create false negatives when two trajectories genuinely have
the same endpoint semantics. Do not change this silently; confirm the intended
negative policy first.

### 6.5 Progress, termination, and total

Progress uses Smooth L1 and termination uses BCE-with-logits. They are computed
once, not once per A/B/C regime.

```text
Laction = Ldirect + ranking_weight*R + wrong_goal_weight*R_wrong

Ltotal = action_loss_weight*Laction
       + progress_loss_weight*Lprogress
       + end_loss_weight*Ltermination
```

`FSQ.pt` selection intentionally uses plain A reconstruction plus progress and
termination according to the selection weights. B/C ranking does not determine
the canonical codebook checkpoint.

## 7. Batch Size

Current YAML uses `fsq_batch_size: 16`, not the original FSQ value 64.

Batch 16 is not split among A/B/C. Every one of the 16 trajectories runs all
three regimes. With `samples_per_skill: 5` and wrong-goal enabled, one update
roughly evaluates:

```text
16 trajectories * 5 timesteps * (A + B + C + wrong-goal C)
```

The expert therefore retains up to four forward graphs before one backward.
`16 * 4` was chosen as a conservative activation load analogous to the old
single-branch batch 64. On B200, test 16 first and then 32. Batch 64 is an
aggressive setting in this implementation.

Larger batches help in-batch negatives. If same-code hard negatives become
mandatory, a cross-batch queue is likely more efficient than increasing batch
size solely to find matching codes.

## 8. Current YAML Snapshot

Important current values:

```yaml
fsq_levels: [3, 3, 3]
fsq_encoder_input_mode: optimal
fsq_state_cond_mode: broadcast
fsq_batch_size: 16
fsq_samples_per_skill: 5
fsq_num_epochs: 500

context: {queries: 16, layers: 4, heads: 8, dropout: 0.0}

conditioning:
  A: {skill: 1.0, image: 0.0, goal: 0.0}
  B: {skill: 0.5, image: 1.0, goal: 0.0}
  C: {skill: 0.5, image: 0.5, goal: 1.0}

weighted_loss: false
weighted_loss_end_weight: 2.0

context_loss:
  direct: {A: 1.0, B: 1.0, C: 1.0}
  ranking: {weight: 0.1, relative_margin: 0.05}
  wrong_goal: {enabled: true, weight: 0.1, relative_margin: 0.05}

fsq_encoder_lr: 3.0e-4
fsq_expert_lr: 2.5e-5
fsq_context_lr: 3.0e-4
fsq_terminator_lr: 3.0e-4
```

Read `fsq_new_config.yaml` for dataset/boundary settings. The current resolved
run name is:

```text
libero_90_full_full_state_obs20_std_global_ms1_fsq333_dino_frozen_small_optimal_vsa_broadcast_context_q16_d4
```

## 9. Running and Outputs

From the new config directory:

```bash
cd lerobot/examples/libero/configs/train_skills/FSQ_new
./submit_train_fsq_new.sh
```

The common resolver initially creates the legacy FSQ output path, but both the
submit script and sbatch script intentionally override it to:

```text
outputs_filtered/FSQ_new/<FSQ_RUN_NAME>/
```

Outputs:

```text
FSQ.pt                 best validation-selection component checkpoint
FSQ_epochNNNN.pt       resumable model + optimizer + scheduler checkpoint
action_stats.npz       normalization statistics
fsq_meta.json          dataset/skillset provenance
```

The sbatch script automatically resumes the latest `FSQ_epoch*.pt` in the same
output folder. `FSQ.pt` is not an exact-resume checkpoint because it omits
optimizer/scheduler state.

The checkpoint still reports format version 3, like original FSQ, but contains
additional context parameters/config fields. Keep it under `FSQ_new`; do not
assume every original v3 consumer can strict-load it.

## 10. W&B Metrics

Epoch metrics are written under both optimizer-step and epoch x-axes:

```text
train/*, val/*
train_epoch/*, val_epoch/*
```

Important action keys:

```text
action                 plain A MSE; legacy checkpoint-selection metric
action_objective       direct + weighted ranking/wrong-goal auxiliaries
action_A/B/C           plain regime MSE
objective_A/B/C        direct objectives; A may be progress-weighted
ranking_AB/BC          active ranking violation
wrong_goal             active hinge violation
action_wrong_goal      plain wrong-goal action MSE
```

Progress/termination, codebook utilization, per-group LR, and end classification
metrics are also logged.

## 11. Stage-0 Integration Contract (Not Implemented Yet)

The agreed future design is source-specific context models with a fixed expert
interface:

```text
FSQ training:
  DINO current -> image_context_fsq -> expert context tokens
  DINO endpoint -> goal_context_fsq -> expert context tokens

Stage-0:
  cond/current image -> image_context_stage0 -> expert context tokens
  VLM/language       -> goal_context_stage0  -> expert context tokens
```

The Stage-0 context models are new modules; they are not expected to reuse the
FSQ context-resampler weights. Their output contract must match:

```text
[batch, context_queries, expert_width]
```

The planned Stage-0 experiment freezes the FSQ-trained expert and its learned
per-layer context gates, then trains the Stage-0 source/context side. The goal is
interface compatibility, not equality of DINO endpoint and language content.

Current gaps:

- `sample_action_chunks()` and `decode()` still expose only the image-free A
  inference path.
- Original FSQ eval has no B/C context modes for this checkpoint.
- Stage-0 loaders currently target the original FSQ implementation/checkpoint
  contract and need explicit `FSQ_new` loading support.
- There is no dedicated FSQ_new eval config/folder yet.

## 12. Required Validation and Known Issues

Do these in order before a long experiment:

1. Instantiate `SplineFSQAE` on one GPU and run a batch size 1 forward with
   A/B/C and wrong-goal disabled.
2. Run batch size 2 with wrong-goal enabled; assert finite loss and gradients.
3. Verify A at zero context gate matches the context-free expert numerically.
4. Verify gradient routing: B/C action loss must not reach the FSQ encoder,
   while it must reach the shared expert and context modules.
5. Verify frozen DINO has no gradients and stays in eval mode.
6. Verify all context gates and both resamplers get nonzero gradients after the
   first optimizer steps. Zero-init gates mean resampler gradients may be zero
   on the very first step while gate gradients open the path.
7. Save/resume a one-step periodic checkpoint and compare parameters/optimizer.
8. Measure B200 VRAM at batch 16 before trying 32.

Known cleanup items:

- In the training step, `third = moved["third"].reshape(...)` appears twice.
  This is harmless but should be removed.
- If a final batch has size 1, wrong-goal forward is skipped and
  `action_wrong_goal` is currently logged as NaN. It does not enter the loss,
  but can poison an epoch aggregate. Replace it with a validity-aware aggregate
  or zero plus a valid-count metric before production training.
- `fsq_meta.json` records base FSQ provenance but not all context scales/loss
  fields. The full config is present in the checkpoint; metadata should still
  be expanded for easier artifact inspection.
- No unit test currently covers the custom expert attention path or the new
  loss formulas.
- Wrong-goal is random in-batch, not same-code hard negative.

## 13. Validation Already Performed

The following checks passed on 2026-07-21:

```text
Python py_compile:
  FSQ_new.py
  train_FSQ_new.py
  resolve_fsq_new_config.py

Bash syntax:
  submit_train_fsq_new.sh
  train_fsq_new.sbatch

Config resolution:
  common train_skills resolver
  FSQ_new nested resolver

CLI construction:
  train_FSQ_new.py --help
```

These checks do not validate tensor shapes, memory use, numerical behavior, or
Slurm execution with real data.

## 14. Related Stage0-pretrain Logging Fix

During the same work session, a separate W&B bug was fixed for
`skill_vla_stage0_pretrain`:

- The policy already returned `ar/skill_ce`, `ar/skill_token_acc`, and
  `ar/skill_exact_acc`, but the common trainer whitelist dropped `ar/*`.
- `ar/*` is now retained and averaged over each `log_every` interval.
- It is routed to `train_autoregressive/*`.
- `fast_context/*` is routed to `train_fast_context/*`.
- FAST/structure CE appears only when `loss.autoregressive.fast: true`.

Files:

```text
lerobot/src/lerobot/scripts/lerobot_train.py
lerobot/src/lerobot/rl/wandb_utils.py
lerobot/tests/scripts/test_lerobot_train_metrics.py
```

The focused metric tests pass (`2 passed`). Existing/running processes do not
pick up this source change, and historical W&B runs cannot be backfilled.
