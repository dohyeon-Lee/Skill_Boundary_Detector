# ABC SkillVLA simulator evaluation

Edit `pt_eval_config.yaml`, then run `./submit_eval.sh` or
`./submit_eval.sh --JOB_ID` to use an existing allocation. The latter uses the
shared allocation dispatcher and does not request another GPU allocation.
`ABC_EVAL_DRY_RUN=1 ./submit_eval.sh` validates and prints the frozen configuration
without submitting. Set `ABC_SKILLVLA_EVAL_CONFIG=/absolute/path.yaml` to select
another config.

The model root follows ABC's global `outputs_root`. Supply a run name and a
numeric checkpoint step or `last` for each component. Submission resolves `last`
to a numeric checkpoint in the YAML snapshot; a queued evaluation cannot
silently move to a newer checkpoint. `output.run_name: auto` incorporates the
three checkpoint steps and a settings hash. The same output resumes completed
episodes, checks its configuration contract, and rejects simultaneous writers.
Submission retains inference checkpoint files under `src/models` using hard
links on the same filesystem. This survives training's `keep_checkpoints: 1`
deleting the original checkpoint while evaluation is queued. It does not copy
optimizer state. The weights occupy storage until both the training checkpoint
and evaluation's links are deleted; across filesystems a file copy is required.
Dry-run does not retain model files.

## Closed-loop behavior

`condition_source: predictor` selects the following deployment behavior.

1. At reset, Predictor mode2 consumes task language, current proprio and all
   three cameras, with its learned BOS previous-skill condition. It predicts a
   real FSQ code and **two** Top-view UV points, restored from model crop to the
   original camera frame.
2. VSA uses that code and held UV goals plus current proprio/three camera images
   to predict 30 absolute joint actions. Execute at most `action_chunk_size`.
3. After every simulator action, Term_Token_1 consumes the live images, raw
   14D proprio, held skill-start proprio, FSQ coordinates and both held UV goals.
   A probability above the threshold (or skill timeout) discards remaining
   actions. Predictor then chooses the next skill, conditioned on the preceding
   predicted code. Consecutive identical codes are valid new skill occurrences.

No ground-truth skill or ground-truth endpoint is supplied to these models.
The official ABC task evaluator supplies episode success/failure only.
This runtime supports the trained ABC current-vision dual-bottleneck VSA,
Predictor mode2 UV and Term_Token_1 DINO-S checkpoints. Independent MuJoCo scenes
allow randomized object topology; VSA inference is batched across active worlds.
World identity survives early completion and asynchronous skill changes.

## GT skill and goal ablation

The default YAML now selects `condition_source: gt`, **VSA 30k** and Terminator
100k. Predictor is neither required nor loaded. `gt_dataset` points to the built
SkillVLA dataset whose FSQ code space matches VSA. `gt_episode_ids: null` chooses
`num_worlds` demonstration episodes without replacement using `rollout.seed`;
an explicit list must have exactly `num_worlds` episode IDs.

Each world restores its demonstration's assembled scene XML and complete
frame-zero qpos, including all objects and both fingers. Initial velocities are
zero because this dataset release contains qpos without recorded velocities.
No new scene randomization is applied. Existing GT FSQ codes are the recorded
demonstration's FSQ encodings of SBD segments, not human semantic labels.
The corresponding two grasp-site endpoints (world XYZ) are projected through
the live Top camera to supply both UV goals to VSA and Terminator.

**The learned Terminator controls skill changes.** GT boundary frame numbers
are logged for reference and never used to schedule transitions. On termination
or timeout, advance one occurrence in the GT sequence; once the final skill
ends, stop the episode and use actual simulator task success. Current and held
skill-start proprio come from the live rollout. Predicted full-skill plans
remain predictions, not copied demonstration trajectories.

This is a **training-demonstration oracle-condition diagnostic**, not a held-out
randomized deployment success rate comparable to the pi0.5 test. Reports label
the protocol, source episode and GT conditions; output names contain `GT`.
Switch `condition_source` to `predictor` to restore deployment evaluation.

## Full-skill overlay

The independent diagnostic planner reuses the VSA observation features and
predicts `bspline_tokens × 14` normalized joints (the point count comes from the checkpoint).
All 14 channels are denormalized with the **VSA checkpoint's** plan quantiles.
Arm joint angles remain radians. Grippers are mapped using the ABC simulator's
policy-to-qpos scale. Separate `MjData` performs FK to the two grasp sites;
live physics state is never overwritten or stepped. No angle wrapping is applied.

The resulting left/right XYZ trajectories appear **only on Top view**, using
the live Top camera's pose and intrinsics. Lines and endpoint dots are cyan
(left) / orange (right). Input UV crosses also appear only on Top view, using
the actual GT/Predictor goals and their validity masks. These are predicted
grasp-site paths, without collision or dynamic feasibility guarantees. The
plan slots represent resampled skill progress, not a fixed frame duration.
Visualization sampling preserves the control RNG.

Top and both wrist views also display the trained bottleneck auxiliary heads:
Upper is coral; Lower is purple. Top shows separate left/right predicted patch
rectangles, while each wrist shows its own arm's rectangle. Both predicted
endpoint XYZs are projected into each live camera as labelled circle/crosses
when visible. XYZ6 is inverse-normalized with the checkpoint's separate
`bimanual_goal_xyz_q01/q99` world-coordinate bounds; joint14 plan statistics
and GT endpoints are never substituted. Readouts come from the real action
forward, update at replans, and reset at each skill occurrence. Patch locations
are held until the next replan; XYZ projections follow the live cameras.
Capture adds only the small auxiliary heads, without another VLM/Expert pass.

## Results

`outputs/<run>/index.html` contains success rates and per-episode videos.
Everything else is under `outputs/<run>/src/`: raw/annotated videos, metrics,
frozen model settings, per-frame termination probabilities and skill transitions.
New evaluations retain clean camera recordings under `videos_raw`, diagnostic
camera overlays under `videos_camera_overlay`, and the final presentation under
`videos` inside each panel directory. HTML episode cards link to the clean
recording as **Original video**. Existing evaluations lack the auxiliary-head
captures and cannot acquire them from an HTML-only refresh. Auto output names
separate the new visualization format from older cached evaluations.
Playing an episode reveals its current skill's termination-probability graph,
threshold and synchronized playhead. The graph switches at each skill transition;
click the graph or the colored skill timeline to seek. Annotated videos include
a bottom skill-ID color bar using the same ID-to-color mapping as LIBERO.
Both views distinguish repeated occurrences of the same skill ID.

Refresh a completed report and its video banners using CPU only (no new rollout):

```bash
.venv/bin/python lerobot/skillVLA/abc/configs/train_pi05/PT/eval/src/report.py \
  lerobot/skillVLA/abc/configs/train_skillVLA/PT/eval/outputs/<run> --workers 4
```

Use `--html-only` to update just the HTML. Raw videos and numerical rollout
traces remain intact. Report data is embedded for offline viewing.

Per-episode `*_trace.npz` contains plan frame, skill order, absolute joint plans
and FK XYZ, so overlays can be inspected without running the models again.

For Predictor deployment comparisons, use the same tasks, seeds, randomization,
max steps and success-stop behavior as the pi0.5 evaluation. GT mode instead
selects 50 recorded `grab_clutter` episodes with the same overall time limit.
