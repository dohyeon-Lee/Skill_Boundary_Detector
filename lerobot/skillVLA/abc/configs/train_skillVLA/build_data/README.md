# ABC native SkillVLA data build

This is the ABC counterpart of LIBERO's `train_skillVLA/build_data` pipeline.
It deliberately reuses the exact skillset recorded in the selected FSQ run:

1. encode that skillset with the selected FSQ epoch;
2. build compact per-frame skill annotations and transition-jitter ISS;
3. copy the selected checkpoint as `FSQ.pt`;
4. build exact bimanual start/end XYZ and four-view UV targets from the
   official per-frame MuJoCo scene state.

Run after the configured FSQ checkpoint exists:

```bash
./submit_build_all.sh
```

Use `./submit_build_all.sh --73428` to run encoding followed by assembly in
the existing allocation's worker, without a new allocation or Slurm step.
Existing encoded latents are reused. Without a job selector, the two stages
retain the usual batch-job dependency chain.

Before enabling the bimanual UV contract, validate the exact recorded
left/right EEF projections over the original top-camera video:

```bash
./preview_dual_focus_uv.sh --task grab_clutter --episodes 3
```

This uses each official sim_224 episode's `scene_qpos.npy` and
`scene_assembled.xml`, not approximate policy-space FK. Output is written to
`<SkillVLA run>/dual_focus_uv_preview/index.html`. Cyan/magenta rings are the
current skill's left/right endpoint UVs. The synchronized layout is
`Left wrist | Top | Right wrist`; wrist targets use the current frame's moving
wrist-camera pose, while all target points remain the skill-end grasp sites.

The default output is:

```text
dataset_ABC/skillvla/abc_sim_all/FSQ333_contrastive_2000_pt/
  FSQ.pt
  fsq_source.json
  skill_latents.npz
  skill_initial_state.npz
  skill_spatial_targets.npz
  build_manifest.json
  skillvla/
    native_dataset.json
    skill_annotations.npz
    meta/
```

Unlike a full LeRobot conversion, `skillvla/` does not duplicate ABC's official
state/action binaries or camera video. Its native marker references the existing
`dataset_ABC/_sim_raw` files. Only episodes represented in the FSQ training
skillset are selected by default, preventing unrelated `abc_sim_all` tasks from
entering training with empty skill labels.

ABC-specific contracts are intentionally explicit in the YAML:

- 14D absolute YAM joint state and action, with grippers at axes 6 and 13;
- no LIBERO EEF-XYZ proprio grounding in the 14D model state;
- a spatial sidecar with `[skill, left/right, xyz]` world-frame start/end grasp
  sites, per-frame wrist projection matrices, and normalized top-L/top-R/
  wrist-L/wrist-R endpoint UVs. Projection matrices are retained so transition
  jitter can select a neighbouring skill and re-project its endpoint without
  rebuilding the dataset;
- transition jitter uses a fixed `pmax=30` (the 30 Hz time-equivalent of
  LIBERO's 20-step window), half-normal, with percentage jitter disabled;
- transition randomization ignores gripper open/close signals;
- the 100k delta-joint DP determines boundaries, while FSQ and the final action
  target continue to use each skill's untouched 14D absolute joint actions.

This folder builds and validates the data artifact. Porting the current LIBERO
SkillVLA training/evaluation wrapper to consume ABC's three-camera native view is
the next stage; the generic factory still rejects ABC native SkillVLA policies.
