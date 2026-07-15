# Skill Boundary Detection: ABC Porting Design Handoff

This document summarizes the investigation and design discussion for porting the
existing LIBERO Skill Boundary Detection (SBD) pipeline to the ABC dataset.
It is intended to let work continue on another server without relying on chat
history.

## Implementation status (2026-07-15)

The first generic `pca_action` implementation is now present alongside the
unchanged `spherical_xyz` path.

```text
lerobot/examples/libero/action_manifold.py
  - checkpoint action-normalizer adapter
  - streaming covariance/PCA and shared artifact cache
  - fixed PCA-subspace direction sampling
  - full-action probes and discrete gripper projection
  - temporal-mean PCA descriptors and GMM divergence

lerobot/examples/libero/build_skill_dataset.py
  - new probe/action/gripper CLI options
  - one shared action_probe_pca.npz per output skillset
  - dataset and anchor_relative action modes

lerobot/examples/libero/skill_divider.py
  - shared one-step denoised-chunk query
  - preserved legacy spherical path
  - generic PCA-action branch
```

Both `train_skills/build_data/build_data_config.yaml` and
`train_skillVLA/build_data/train_skillVLA_config.yaml` select the probe mode and
forward the same settings through their resolvers/Slurm scripts. A readable
probe tag is included in new PCA skillset paths, while `spherical_xyz` keeps the
old path exactly.

The implemented first-version descriptor is:

```text
normalized denoised chunk (H,D)
  -> temporal mean (D)
  -> fitted PCA coordinates (r)
  -> GMM cluster-mean cosine divergence
```

LIBERO smoke validation used the existing 100k state-DP checkpoint. PCA fit on
191,059 anchors from 3,921 episodes retained 5/7 components at the 0.95 target
(0.9765 cumulative explained variance). Both `pca_action` and legacy
`spherical_xyz` completed a real CPU one-step denoising/GMM query.

## 1. Current SBD Implementation

The current batch segmentation entry point is:

```text
lerobot/examples/libero/build_skill_dataset.py
```

It calls `run_vf_analysis()` from:

```text
lerobot/examples/libero/skill_divider.py
```

The Slurm entry point is:

```text
lerobot/examples/libero/configs/train_skillVLA/build_data/src/build_skillset.sbatch
```

### Current VF cosine-divergence pipeline

At every replanning time `t`:

1. Build the observation history from the demonstration and compute DP global
   conditioning (`policy.diffusion._prepare_global_conditioning`).
2. Take the demonstration action chunk of length `horizon`.
3. Treat action dimensions `0:3` as per-step delta EEF XYZ.
4. Cumulatively sum those XYZ deltas into an EEF path, then create spherical
   probes by rotating the complete path around its endpoint direction.
5. Keep every action dimension from index `3` onward unchanged for every
   probe.
6. Feed all probe chunks through the UNet at one fixed DDIM denoising step.
   This is not full diffusion sampling; it is one deterministic scheduler step.
7. Sum the denoised action chunk along time to get one action-sized vector per
   probe.
8. Fit a GMM on the first three components only, then compute RMS pairwise
   cosine distance among GMM cluster means. This scalar is `div_cos`.
9. Savitzky-Golay smooth `div_cos`; local maxima above the mean become skill
   boundaries. NMS and a boundary margin remove nearby/edge peaks.

Relevant functions and locations:

```text
skill_divider.py
  _generate_spherical_samples()  # EEF-XYZ path rotation
  _make_probe_chunks()           # first 3 dimensions are XYZ; remaining dims fixed
  _query_vf_error()              # UNet + one DDIM step
  compute_vf_divergence()        # GMM and cosine score on [:3]
  run_vf_analysis()              # per-episode loop

build_skill_dataset.py
  _detect_boundaries()           # SG filtering + peak detection
```

The old/simple alternative is in `lerobot/examples/libero/replay_demo.py`.
It compares a fully denoised DP action chunk with the demo chunk using action
MSE, smooths that curve, and peaks it. The current `skill_divider.py` also has
an optional `compute_pred_mse` path, but it computes MSE on `:3` only, so it is
not an ABC-ready full-action error baseline.

### Important limitation: gripper is not directly scored now

In the current implementation:

- Gripper and all non-XYZ action dimensions are passed into the UNet unchanged
  for every probe.
- The GMM/divergence sees only `vf_batch[:, :3]`.
- Optional prediction MSE also sees only `:3`.

Therefore gripper state may affect the UNet response indirectly, but a gripper
open/close transition does not directly create a boundary score. A transition
with little EEF XYZ change can be missed.

## 2. ABC Dataset Facts Confirmed in This Repository

ABC actions are stored as absolute joint commands, not delta EEF commands.
The 14-D YAM layout is:

```text
[left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]
```

Evidence in the repository:

- `abcdl_RLLAB/abcdl/mcap/reader.py` reads `*-arm-action` `position` fields,
  aligns streams, and concatenates left/right actions to `(T, 14)`.
- `lerobot/examples/libero/configs/generate_training_dataset/ABC_dataset/src/convert_abc_dataset.py`
  copies that action vector unchanged into the LeRobot v3 `action` feature.
- `lerobot/examples/libero/configs/generate_training_dataset/ABC_dataset/README.md`
  explicitly calls ABC action "absolute joint".

The ABC converter also preserves measured EE 4x4 poses from MCAP
`RobotState.pose` in optional columns:

```text
observation.states.ee_pose_left
observation.states.ee_pose_right
```

These are useful for offline analysis, but are not the policy action labels.

### Relative action used by downstream VLA-style training

The repository supports a training-time conversion for relative action models:

```text
relative_action[t + k] = absolute_action[t + k] - state[t]
```

for action dimensions selected by a mask. The default configuration can exclude
gripper dimensions, leaving gripper absolute. This conversion is anchored at the
current state once per action chunk; it is not a per-step action delta.

Relevant implementation:

```text
lerobot/src/lerobot/processor/relative_action_processor.py
```

For any SBD DP that is meant to match VLA action semantics, its inputs/probes
must use the exact representation and normalization that the DP was trained on.

## 3. Why Current VF Code Cannot Be Applied Directly to ABC

Direct reuse would incorrectly interpret:

```text
left_joint_0, left_joint_1, left_joint_2
```

as delta EEF X/Y/Z. In particular, the current code would:

- Cumulatively sum absolute joint positions as if they were translational
  deltas.
- Apply a 3D rotation to three joint coordinates. Joint coordinates are not
  Cartesian axes, so this has no physical EEF-direction interpretation.
- Score only the left arm's first three joints; right-arm behavior and both
  grippers are excluded from the divergence score.
- Create probes that may violate joint limits, self-collision constraints, or
  coordinated bimanual motion constraints.
- Let the magnitude/direction of the absolute pose relative to zero influence
  cosine distance, rather than measuring intended movement.

Therefore, do not run the existing spherical-XYZ probe code on raw ABC action
vectors.

## 4. Action Error vs VF Divergence

Action error and VF divergence answer different questions.

```text
Action error:
  "How poorly does this policy sample/predict the demonstrated action here?"

VF divergence:
  "Around this action/state, does the policy's denoising flow converge to one
   action mode or to several different modes?"
```

Action error can peak at contacts, gripper changes, occlusion, teleoperation
jitter, poorly trained portions of the dataset, or genuine skill boundaries.
It is a surprise/difficulty signal, not necessarily a semantic boundary signal.

VF divergence is intended to detect a decision or mode-splitting point more
directly. However, that advantage only holds when its action probes and metric
are meaningful for the policy's action representation.

### Practical baseline recommendation

For a quick ABC baseline, a full-action error signal is more defensible than
the unmodified XYZ VF signal:

- Compare all relative-joint action dimensions in the DP/VLA's normalized
  action space.
- Include gripper with an intentional weight or a separate gripper-change term.
- Avoid a single random full-diffusion sample if possible. Use fixed noise or
  multiple seeds and average/median the error curve to reduce sampling noise.
- Smooth and peak-detect it as in the existing pipeline.

This baseline is action-space consistent with VLA but should not be treated as
the final semantic SBD method without visual/manual evaluation.

## 5. Main Proposed Direction: Action-Manifold VF

The preferred eventual design is not a separate EEF-action DP, because SBD
segments are ultimately used to train a VLA using relative joint actions.

Instead:

```text
SBD DP action representation:
  same normalized relative-joint action representation as the downstream VLA

SBD concept:
  measure multimodality of the policy's local action manifold
```

At a high level:

```text
demo relative-joint action chunk
  -> construct nearby, plausible action-space probes
  -> one-step VF/denoising query for all probes
  -> determine whether outputs gather into one action mode or several modes
  -> high multimodality peak is a boundary candidate
```

This generalizes the current EEF approach. The 3D spherical shell is only a
way to create local probes; it is not the core idea. The core idea is to probe
the neighborhood of a demo action and inspect the denoising basins.

Using the same relative-joint representation addresses the concern that SBD
boundaries might be based on a separate delta-EEF policy while VLA learns
relative joint chunks.

## 6. Simplest Action-Space Probe Construction

For a first VF prototype, do not sample independent random noise at every
time/action coordinate. That creates jittery, unrealistic action chunks.

Let the continuous arm-joint part of a demo relative-action chunk be:

```text
A_demo: (H, 12)
```

For each probe:

1. Sample one 12-D direction `u_i`.
2. Normalize it to unit length in a scale-aware action coordinate system.
3. Use one small radius `rho`.
4. Add the same offset to every future step:

```text
A_probe_i[h] = A_demo[h] + rho * u_i
```

Why the same `u_i` is used across the action chunk:

```text
A_probe[h + 1] - A_probe[h] = A_demo[h + 1] - A_demo[h]
```

Thus the demo trajectory's temporal shape is retained; only the joint-target
trajectory is shifted. This avoids the artificial high-frequency shaking caused
by an independent perturbation at every timestep.

The probe perturbation is deliberately added local exploration noise. It is
not the diffusion scheduler's forward-process noise. It is the action-space
analogue of rotating the demo EEF path in the current implementation.

### Gripper probes

Do not put gripper directly into the continuous random direction. Treat it as
a separate near-discrete mode. For each continuous-joint probe, enumerate a
small set such as:

```text
- keep demonstrated left/right gripper behavior
- left open/close alternative
- right open/close alternative
- both alternative (optional)
```

The exact choices should respect how gripper values are encoded and whether the
relative-action transform excludes gripper.

### Validity and coordinate requirements

- Build perturbations in a scale-aware coordinate system. The safest workflow
  is: create a physical relative-joint candidate, validate it, then apply the
  same action normalizer that the DP uses.
- Recover absolute target joints from the anchor state before checking limits:

```text
q_target[h] = q_state[t] + A_probe[h]
```

  Only apply this to dimensions that are actually relative; excluded gripper
  dimensions remain absolute.
- Reject/resample invalid probes rather than clamping them. Clamping piles
  probes onto limits and can artificially inflate a divergence score.
- `rho` should be chosen in whitened/joint-range-normalized units, not as a
  fraction of the raw action norm. A small parameter sweep is required.

## 7. PCA Improvement for Probe Directions

The simple first version uses a random 12-D unit vector for `u_i`.
It can still generate uncommon correlated joint motions.

PCA is an optional improvement that replaces arbitrary directions with
directions common in the dataset:

```text
u_i = P[:, :r] @ z_i
```

where:

- `P` is the PCA basis of standardized relative-joint actions.
- `r` is a small number of principal components (for example 3--6).
- `z_i` is a random vector in this low-dimensional PCA subspace, then
  normalized.

This does not change the probe structure. It only changes how the shared
12-D offset direction `u_i` is selected.

### PCA fitting scope

Recommended default: fit PCA once on a large, task-balanced portion of the
DP/VLA training dataset, not independently per episode.

Reasoning:

- The PCA basis should represent common robot joint coordinations/motion
  synergies, not only the one or two motions present in a particular episode.
- Per-episode PCA is biased by that episode and makes scores less comparable
  across episodes.
- If global ABC behavior is very heterogeneous, task/group-specific PCA is a
  reasonable later refinement. Per-episode PCA is usually not the first choice.

For the simple constant-offset probe, PCA can be fitted to a 12-D endpoint
relative target such as:

```text
x_t = action[t + H - 1, :12] - state[t, :12]
```

with exactly the same anchor semantics as the downstream relative-action
processor.

### PCA cost

This is cheap. For 12-D data, a streaming mean and `12 x 12` covariance are
enough; no action samples need to be retained in memory. The final
eigendecomposition is negligible. Reading parquet state/action data is the
main cost, and even that is small compared with DP training or SBD inference.

If a full pass is inconvenient, use a task-balanced random sample of anchors.
Hundreds of thousands of samples are already ample for a 12-D PCA basis.

## 8. Scoring the Generalized VF Output

The current score uses denoised chunk sums and only XYZ. The generalized
version should include both arms and gripper, but should not blindly flatten an
`H x 14` chunk into a very high-dimensional GMM input: probe count will be
small and fixed-component GMMs can become unstable/artificial.

Possible compact continuous descriptors for a denoised probe output:

```text
- final relative joint target for all continuous joints (12-D)
- a few low-frequency temporal coefficients in addition to that endpoint
- optionally FK-derived left/right EEF displacement as an auxiliary metric
```

For gripper, compute a separate mode/disagreement score after converting the
denoised value to the dataset's open/close representation. Do not rely solely
on continuous cosine similarity to represent gripper semantics.

The final score can combine:

```text
continuous joint-action multimodality
+ gripper-mode disagreement/entropy
+ optional EEF-space multimodality
```

### GMM caution

A fixed `K` GMM always partitions samples into `K` groups, even when they lie
on one smooth, unimodal continuum. This was already a heuristic in the current
3D implementation; it becomes more dangerous in higher dimensions.

Before calling a curve "multimodality," consider using at least one of:

```text
- separation between cluster means relative to within-cluster covariance
- comparison with a 1-component GMM (BIC/AIC improvement)
- cluster-size/weight checks
- a non-GMM pairwise spread score as a baseline
```

The term "VF multimodality" should therefore be read as local denoising-basin
dispersion unless this validation is added.

## 9. Alternative: Full Conditional Samples

Instead of handcrafted local probes, one can sample multiple full action
chunks from the DP at the same observation with different initial diffusion
noise, then cluster/measure dispersion of those outputs.

```text
same observation conditioning
  -> N different initial diffusion noises
  -> full denoising for N action chunks
  -> action-manifold multimodality score
```

This is a more direct measure of the policy's conditional action distribution
and needs no probe-shell design. It is more expensive than the current
single-step batched VF query, although batching may make it usable. It is a
valuable validation baseline for any fast local-probe approximation.

## 10. Suggested Execution Plan

1. Confirm the exact DP action preprocessing used for the ABC DP checkpoint:
   relative-action mask, gripper treatment, and normalization statistics.
2. Build an ABC action-error baseline using the full policy action space,
   including deliberate gripper handling. Inspect curves/videos on a small,
   diverse set of episodes.
3. Implement the simplest action-manifold VF prototype:
   - continuous relative-joint probes with a shared offset across the horizon;
   - random unit directions first;
   - gripper candidate enumeration;
   - absolute joint-limit rejection/resampling;
   - compact all-joint output descriptor.
4. Add global PCA directions only if random probes are visibly implausible or
   produce unstable curves.
5. Compare action-error, action-manifold VF, and optionally full-sample curves
   against manually inspected boundaries. Do not select the method solely from
   aggregate peak counts.
6. Once a metric is selected, use it to build the skill dataset consumed by
   SkillVLA. Preserve raw absolute action storage; relative conversion remains
   the training-time action representation.

## 11. Open Decisions

- Exact ABC DP model/checkpoint and its preprocessing representation.
- Relative-action mask: whether gripper stays absolute, and its open/close
  numeric values.
- Probe radius `rho`, number of continuous probes, and gripper combinations.
- Whether to reject or otherwise handle candidates invalid under joint limits.
- Output descriptor: final joint target only, endpoint plus low-frequency
  trajectory information, or an FK-assisted descriptor.
- Multimodality score: current fixed-GMM cosine heuristic versus a
  covariance-aware/BIC-validated measure.
- Boundary evaluation protocol and a manually reviewed ABC episode subset.

## 12. Key Takeaway

Do not port the literal LIBERO XYZ spherical code to ABC absolute joints.
The coherent long-term design is to make SBD operate in the same normalized
relative-joint action space used by VLA, while measuring whether nearby action
probes denoise into one or several action modes. This includes both arms and
gripper, avoids representation mismatch, and leaves EEF/FK as an optional
auxiliary geometric metric rather than an incompatible policy action space.
