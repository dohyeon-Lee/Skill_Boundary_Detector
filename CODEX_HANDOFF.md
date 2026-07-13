# Codex handoff: SkillVLA / FSQ redesign

Updated: 2026-07-13 (Asia/Seoul)

This document transfers the design context from the long Codex thread that was
run in `/scratch/mdorazi`. The next workspace is expected to be
`/scratch2/mdorazi`.

## 1. Repository state to start from

- Old repository root: `/scratch/mdorazi/Skill_Boundary_Detector`
- Main code directory: `/scratch/mdorazi/Skill_Boundary_Detector/lerobot`
- Branch: `splitVLA_lora_ABC`
- Authoritative commit: `6f966c9b45201b8dc553b1b227f7e18d09e22ebe`
- Commit subject: `save`
- The worktree was clean immediately before this handoff file was added.

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

## 5. Current FSQ config and naming

Main config:

`lerobot/examples/libero/configs/train_skills/FSQ/fsq_config.yaml`

At handoff time the important values are:

```yaml
fsq_levels: [5, 5, 5]
fsq_encoder_input_mode: zero_grounded

fsq_terminator_arch: small
fsq_vision_backbone: dino
fsq_freeze_vision_encoder: true
fsq_dino_model_path: /scratch/mdorazi/Skill_Boundary_Detector/models/dinov3-vits16
fsq_dino_image_size: 224
fsq_siglip_image_size: 224
fsq_cond_encoder_variant: gemma_300m  # only used for arch=cond

fsq_batch_size: 64
fsq_num_workers: 8
fsq_num_epochs: 300
fsq_checkpoint_every: 50
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
{target_dataset}_{dp_tag}_{fsq_tag}_{fsq_vision_tag}_{fsq_terminator_arch}
{fsq_encoder_input_suffix}{weighted_suffix}{fsq_exp_suffix}
```

Without the visual line break, the current default resolves to approximately:

```text
libero_90_full_full_state_obs20_fsq555_dino_frozen_small_v3_vsa_state_m2
```

Tags:

- `fsq_vision_tag`: e.g. `dino_frozen`, `dino_tuned`, `siglip_frozen`, or
  `siglip_tuned`.
- `fsq_encoder_input_suffix`: empty for zero-grounded; `_rawstate` for raw state.
- `weighted_suffix`: empty when `weighted_loss: false`; `_weighted` when true.

The Slurm script records architecture, vision, freeze state, and encoder input
mode in `fsq_meta.json`.

Server paths in YAML must be updated for `/scratch2/mdorazi`; do not assume the
old absolute DINO/PI paths exist unchanged.

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

SkillVLA data construction:

- `lerobot/examples/libero/configs/train_skillVLA/build_data/`
- `lerobot/examples/libero/configs/train_skillVLA/build_data_eval/`

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

