# Stage-0 VLM Residual Experiment Handoff

Updated: 2026-07-25 14:40 KST  
Branch at snapshot: `skillVLA_rollback`

This document transfers the current Stage-0 design, experiment matrix, W&B
interpretation, and the latest quantitative analysis. The code and current YAML
remain authoritative; job status and W&B values below are a time-stamped
snapshot.

## 1. Current objective and architecture

The intended decomposition is:

```text
base VSA: vision(cond) + state + skill -> raw expert action hidden h_base
VLM:      image + language -> final VLM hidden tokens h_vlm

delta = CrossAttention(Q=Norm(h_base), K=h_vlm, V=h_vlm)
h_cond = h_base + alpha * delta

unconditional action = ActionHead(Norm(h_base))
conditional action   = ActionHead(Norm(h_cond))
```

Important implementation details:

- The VLM residual cross-attention is applied once, after all expert blocks and
  before the existing expert final norm/action head.
- `suffix_raw` is the expert hidden after every expert layer and before final
  norm. It is returned without `detach()` by the joint backbone.
- Conditional and unconditional forwards use the same sample, noisy action,
  time, cond path, expert, state, and GT skill. Their intended functional
  difference is only the final VLM residual.
- The current residual gate is a scalar bounded to `[0.1, 0.2]`, initialized at
  `0.15`. Cross-attention output projection `Wo` is zero-initialized.
- `token_access` controls which valid VLM tokens become residual K/V. Exp3 and
  both current Exp4 jobs use language only. Padding tokens are masked.
- Current VLM LLM and VLM vision tower are frozen. Cond, cond vision, and the
  action expert are trainable. There are no active LoRAs in these runs.
- Skill conditioning uses expert hidden broadcast. `cond.state_adarms` is false
  in the current experiments.

Primary implementation locations:

```text
lerobot/src/lerobot/policies/skillVLA/modeling_skillVLA.py
  Stage0VLMResidual
  SkillVLAModel._stage0_dual_flow_view
  _stage0_endpoint_xyz_loss
  SkillVLAPolicy.forward loss assembly and W&B metrics

lerobot/src/lerobot/policies/skillVLA/configuration_skillVLA.py
lerobot/examples/libero/configs/train_skillVLA/stage0/src/stage0_train_config.py
lerobot/examples/libero/configs/train_skillVLA/stage0/stage0_train_config.yaml
```

## 2. Gradient routing

`shared`:

- Unconditional loss updates the base VSA.
- Conditional loss updates both the shared base VSA and the VLM residual.
- Thus an alternative conditional objective can move the reusable base motion.

`split`:

- Unconditional loss updates the base VSA.
- Conditional loss reads a detached base hidden and updates only the residual
  branch. Final norm/action-head parameters are functionally detached for the
  conditional pass.
- The residual parameter group has its own gradient clipping path.

The auxiliary skill predictor is intentionally isolated: VLM features are
detached for predictor training, and its loss updates only reader/head. The
Exp4 split/shared predictor curves are exactly identical, confirming that this
isolation currently works. Terminator training is disabled in the jobs analyzed
here; when enabled, it also has a separate optimizer/gradient path.

## 3. Experiment definitions

| Experiment | Definition |
|---|---|
| Exp0 | Initial final-layer residual, shared routing, cond:uncond `1:0.5`, alpha near `0.01`; residual effectively stayed inactive. |
| Exp1 | `Wo` zero-init, alpha `0.15` bounded to `[0.1,0.2]`, gate excluded from weight decay, residual separately clipped; shared `1:0.5`. |
| Exp2 | Exp1 residual with split routing. Conditional trains residual, unconditional trains base. Weights `1:1`. |
| Exp3 | Exp2 plus residual K/V restricted to language tokens. Predictor still reads image+language. Same-skill/different-task grouping enabled. |
| Exp4-1 | Conditional objective becomes chunk endpoint XYZ loss; unconditional remains flow MSE. Both split and shared variants are running. |
| Exp4-2 | Optional wrong-task-language relative hinge ranking. Implemented behind a flag but currently disabled. |
| Exp5 | Exp3 with VLM LLM and vision unfrozen. Predictor remains read-only with respect to the VLM. Not part of the six-run snapshot below. |

## 4. Loss and logging semantics

Do not compare metrics with different units.

### Flow objective

The per-step flow target is `u = noise - action`. Conditional and unconditional
flow metrics are both per-step/per-action-dimension MSE:

```text
train_regime/conditional_flow_loss
train_regime/unconditional_flow_loss
```

These two are directly comparable.

### Exp4-1 endpoint objective

For valid chunk steps and XYZ dimensions only:

```text
endpoint_error = sum_t(pred_action_xyz[t] - gt_action_xyz[t])
endpoint_loss  = mean_xyz(endpoint_error ** 2)
```

This is chunk-end displacement error, not a per-step trajectory loss. Its scale
can grow with chunk length and correlated errors, so it is not directly
comparable to flow MSE.

```text
train_regime/conditional_objective_loss  # endpoint_xyz in Exp4-1
train/stage0/cond_endpoint_xyz_loss       # same value
train_regime/unconditional_flow_loss      # different unit
```

Therefore, conditional objective loss being numerically greater than
unconditional flow loss is not a loss inversion and is not itself dangerous.

In Stage-0, `train/loss_flow` and `train_regime/dual_objective_loss` are the
weighted dual objective. In Exp4-1 this means endpoint plus unconditional flow,
so they are not comparable to older all-flow experiments. `train/action_loss`
remains raw conditional flow MSE for cross-experiment diagnostics.

### Residual diagnostics

- `stage0/alpha`: bounded scalar gate. It remains close to initialization in all
  current runs; this alone does not imply the residual weights are frozen.
- `stage0/scaled_delta_rms`: hidden correction after alpha.
- `stage0/scaled_to_base_ratio`: scaled hidden correction RMS divided by raw
  base hidden RMS. This is the cleanest hidden-space influence measure.
- `stage0/velocity_delta_rms`: RMS of conditional minus unconditional predicted
  vector fields. This directly measures output change.
- `param_drift/vlm_residual`: residual-module parameter movement.
- `param_drift/lang_bridge`: legacy language-K/V bridge group. It is disabled
  and irrelevant to the renewed Stage-0 residual; expect zero. The relevant
  group is `vlm_residual`.

## 5. Running jobs and W&B mapping

All six jobs were still advancing in their Slurm logs at this snapshot.

| Job | W&B ID | Experiment | Train log | Latest W&B |
|---:|---|---|---:|---:|
| 32000 | `h5bnwpwb` | Exp2 split, grouped on, older broadcast-150 data | 44K | 28.4K |
| 32005 | `28zmvr0l` | Exp2 split, grouped off, broadcast-175 new data | 32K | 32.2K |
| 32006 | `92ox9i8t` | Exp2 split, grouped on, broadcast-175 new data | 32K | 16K |
| 32014 | `qhqbhrff` | Exp3 split, language-only, grouped on | 30K | 14.4K |
| 32017 | `9tvbvfg2` | Exp4-1 split, language-only, grouped on | 27K | 26.4K |
| 32018 | `fbblrh1z` | Exp4-1 shared, language-only, grouped on | 27K | 26.4K |

Jobs 32006 and 32014 did not actually crash. Their training logs continue at
normal speed, but W&B file streaming stopped after repeated API 500/504 errors
and then `409 Conflict: filestream at capacity`. W&B marks those runs `crashed`
and has no metrics after 16K/14.4K. Do not interpret the missing W&B tail as a
training failure.

Job 32000 uses the older broadcast-150 FSQ/data and should not be used for a
strict numerical comparison with the broadcast-175 new-data jobs.

## 6. Quantitative findings

Values below are means over the latest ten W&B logging windows unless stated
otherwise. Logging frequency is 200 optimizer steps.

### 6.1 Exp2 grouped batching: on versus off

Fair comparison at common step 14.4K:

| Metric | Exp2 off (32005) | Exp2 on (32006) |
|---|---:|---:|
| Conditional flow | 0.110293 | 0.109804 |
| Unconditional flow | 0.110298 | 0.109806 |
| Scaled/base ratio | 0.0478% | 0.0497% |
| Velocity delta RMS | 0.000624 | 0.000630 |
| Residual drift | 2.899 | 2.916 |
| Predictor skill accuracy | 40.9% | 31.4% |

Grouping makes the residual only marginally larger and changes flow by less
than 0.5%. It does not create meaningful conditional/unconditional separation.
The lower auxiliary predictor accuracy likely reflects the grouped sampler's
changed skill/task distribution; it is not evidence that the action residual
benefited, and should be checked again in simulation rather than interpreted
as a standalone quality ranking.

The current sampler/jitter implementation now preserves the requested pairs:

```text
requested grouped fraction:       0.50
effective after jitter fraction:  0.50
effective of requested:           1.00
original progress gap:            about 0.009-0.010
jittered progress gap:            about 0.008-0.009
```

This supersedes an earlier broken state where jitter destroyed many pairs and
made the progress gap about 0.26. The current runs do not show that problem.

### 6.2 Exp3 language-only versus Exp2 image+language

Fair comparison at common step 14.4K, both grouped-on:

| Metric | Exp2 image+language | Exp3 language-only |
|---|---:|---:|
| Conditional flow | 0.109804 | 0.109859 |
| Scaled/base ratio | 0.0497% | 0.0403% |
| Velocity delta RMS | 0.000630 | 0.000390 |
| Residual drift | 2.916 | 2.655 |
| Alpha | 0.149954 | 0.149688 |

Language-only K/V does not improve action flow and reduces the residual's
output influence. Restricting K/V to language made the correction path weaker;
it did not force useful language dependence.

### 6.3 Exp4-1 split versus shared at 26.4K

The configurations are not identical in weight: split uses cond:uncond `1:1`,
while shared uses `1:0.5`. The objective and data are otherwise matched.

| Metric | Exp4 split (32017) | Exp4 shared (32018) |
|---|---:|---:|
| Conditional endpoint objective | 1.2162 | **1.0116** |
| Conditional raw flow | **0.09263** | 0.13479 |
| Unconditional flow | **0.09263** | 0.13475 |
| Main pre-clip grad norm | **0.929** | 25.82 |
| Residual grad norm | 0.162 | 0.148 |
| Scaled delta RMS | 0.01353 | 0.04631 |
| Scaled/base ratio | 0.0353% | 0.0894% |
| Velocity delta RMS | 0.000557 | 0.001882 |
| Action-expert drift | 27.58 | 27.71 |
| Residual drift | 3.235 | 3.480 |

At 14.4K, shared endpoint loss was only about 5% below split. At 26.4K it is
about 16.8% below split, so the endpoint advantage is widening. However, shared
raw flow is now about 45.5% worse than split, and this disadvantage also grew.

The correct interpretation is:

- The numerical endpoint-vs-flow loss ordering is meaningless because the
  objectives have different units.
- Shared routing really does optimize the endpoint objective more strongly.
- It does so while moving the shared base away from the original per-step flow
  solution. Its main gradient remains roughly 26, far above the global clip
  threshold of 1.0, whereas split is now near/below that threshold.
- Similar action-expert drift magnitudes do not make the two solutions
  equivalent. Clipping constrains update magnitude, not direction; the shared
  update direction is strongly endpoint-biased.

Despite the larger residual in shared, conditional and unconditional flow are
still effectively identical:

```text
split:  0.0926297 vs 0.0926301  (difference below 0.001%)
shared: 0.1347916 vs 0.1347525  (conditional is about 0.029% worse)
```

The scaled residual is still less than 0.1% of base-hidden RMS. Therefore the
lower shared endpoint loss is not evidence that language is being used. Much
of the endpoint improvement can come from the shared base itself adapting to
the endpoint objective.

The skill predictor loss/accuracy curves are numerically identical between the
two Exp4 jobs, which is a useful confirmation that predictor gradients are not
leaking into the body.

## 7. Current conclusions

1. The residual branch is trainable and its parameters move, but it still has
   very little functional influence. Across Exp2/3, conditional and
   unconditional flow losses remain effectively identical.
2. Same-skill/different-task batching alone did not force VLM use. Its pairing
   implementation is now healthy, so this is an experimental result rather
   than the old jitter bug.
3. Language-only residual K/V weakened the branch relative to image+language.
4. Exp4 shared demonstrates a real endpoint-versus-flow trade-off. Its danger is
   not a scalar loss "inversion"; it is that a large endpoint gradient updates
   and clips the shared base while degrading reusable flow quality.
5. For the stated goal, preserving base VSA motion and using VLM only for a
   correction, split routing remains the cleaner default.
6. Training loss cannot decide whether Exp4 shared's endpoint trade-off is
   behaviorally useful. Conditional/unconditional simulation eval at matched
   checkpoints is still decisive.

## 8. Missing diagnostic and recommended next checks

The most important missing metric is the unconditional endpoint loss. Add or
compute:

```text
unconditional_endpoint_xyz_loss
endpoint_gain = unconditional_endpoint_xyz_loss - conditional_endpoint_xyz_loss
relative_endpoint_gain = endpoint_gain / unconditional_endpoint_xyz_loss
```

Without it, the conditional endpoint curve cannot distinguish:

- the shared base becoming better at endpoints, from
- the VLM residual supplying a language-dependent endpoint correction.

Recommended order:

1. Evaluate Exp4 split/shared at the same 25K checkpoint in both conditional
   and unconditional modes.
2. Add unconditional endpoint diagnostics before launching another objective
   ablation.
3. Keep split as the baseline if base-motion preservation matters.
4. Treat Exp4 shared as a deliberate endpoint-finetuning baseline, not as proof
   that VLM conditioning works.
5. Do not use W&B tails after 16K/14.4K for jobs 32006/32014 unless their local
   W&B data are recovered/synced; their Slurm text logs only contain aggregate
   loss and grad norm after upload failure.
