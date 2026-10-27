# Stage-1 closed-loop evaluation

Edit `stage1_eval_config.yaml`, then run:

```bash
./submit_eval.sh
```

This is the shared evaluator previously located at `train_skillVLA/stage1_eval`.
Its main YAML and evaluation behavior are unchanged. Results are stored in
`outputs/`; existing result folders have moved here without being regenerated.
With `resume: true`, completed compatible panels are reused as before.

`src/` contains the config resolver, rollout engine, GT sequence loader, video
composition, and report/chart generation. Modules imported by that engine are
retained, including goal overlays and optional attention capture. Standalone
attention/goal-noise experiments, SBD comparison, wrist probes, and source-map
builders are no longer included. Episode-exact evaluation still reads existing
dataset-owned `eval_init_states.npz` maps.

NewTask FT, per-skill evaluation, and pi05 report tools reuse this directory.
NewTask FT keeps its own YAML, logs, and outputs under
`FT/eval`; it does not write its results here.

Annotated `eval_episode_*.mp4` videos show the actual Expert XYZ as a red
crosshair (projected into Top and Wrist cameras) and the input UV as a cyan
circle on the rollout/Top views. The rollout panel also prints XYZ in grounded
metres and UV in normalized `[-1, 1]` coordinates. Predicted XYZ comes from the
action call's cached output, independently of whether UV is GT or predicted;
GT Expert overrides are labelled `GT XYZ`. Chunk-level trace points keep goals
aligned with skill transitions and video stride. No extra inference is run.
Presentation/per-skill tint-only clips keep their existing clean rendering.

Set `video.fullskill_overlay: true` to draw B-spline fullskill trajectories on
the ordinary video's ROLLOUT panel: green is the first plan of the active
skill occurrence, orange is its latest action-replanning plan. The green plan
stays fixed until the next skill; orange updates at each actual replan (e.g.
every 5 steps), holding between replans. Both show the full 30-point XYZ path.
The final denoising forward's existing proprio output is inverse-normalized
with the checkpoint's state quantiles and translated back into world metres.
No extra forward, noise sampling, or action decoding is performed. The option
defaults to false; non-B-spline panels are skipped with a warning. It is
independent of the more expensive `motion_plan.enabled` image galleries.
Turning it on invalidates older panel caches so videos are regenerated.

Checkpoint lists retain the existing bar charts and generate
`checkpoint_success_rates.png` plus one task-wise line chart per model under
`checkpoint_task_success_rates/`.

`eval_max_workers_per_gpu` accepts 1–10 independent evaluation processes per GPU.
Each worker loads its own model and simulator; increasing concurrency requires
enough GPU/host memory and does not guarantee higher throughput.
Work is split over task x model/checkpoint panels. With 10 tasks and 9 panels,
`eval_num_gpus: 1` and `eval_max_workers_per_gpu: 10` run 10 workers on one GPU,
covering all 90 combinations. `eval_num_gpus: 10` and
`eval_max_workers_per_gpu: 5` instead request 10 GPUs with 5 workers each.
With only one panel, those same 10/5 settings need only 2 GPUs;
`eval_num_gpus` is an upper bound.
PT and FT evaluation share this limit and packing code.

UV VSA checkpoints can use the existing XYZ-conditioned Newterm7. Goal loading
accounts for both components independently: with `focus_source: gt` and
`pose_source: gt`, VSA receives GT endpoint UV while Newterm7 receives GT endpoint
XYZ. End XYZ is not injected into the UV VSA action inputs. The resolver records
`terminator_needs_goal_xyz`, requires exact-demo alignment for GT XYZ, and rejects
`pose_source: predictor` if the selected predictor has no compatible XYZ head.
This also applies to FT evaluation through the shared engine.

## Bspline proprio visualization (one episode per task)

```bash
./submit_bspline_visualization.sh
```

This reads the **same** `stage1_eval_config.yaml` and snapshots it normally.
Without editing that YAML, the visualization resolver uses one episode per
selected task/model/checkpoint and batch size 1. GPU allocation and worker
packing follow the shared `eval_num_gpus` and `eval_max_workers_per_gpu`
settings and task x panel allocation. For 10 tasks and one panel, values of
10 and 2 request 5 GPUs; values of 2 and 2 request 2 GPUs and process the
remaining tasks sequentially within each worker. Each worker runs only its
assigned panels, sequentially when there is more than one. All model, Terminator,
GT/predictor, seed, and scene-selection settings remain shared.
Extra per-skill videos and sampled FSQ training-example clips are skipped.
Only Bspline checkpoints are accepted. W&B logging is disabled: the incidental
single-trial success numbers are not a success-rate comparison.

Results are separate from ordinary eval:
`outputs/<output_name>_bspline_visualization/`. In each panel, open
`videos/<task>/motion_plans/episode_000/index.html` (also linked from skill HTML).
Four uniformly spaced real replans per skill are shown by default; an optional
`motion_plan.samples_per_skill` in the shared YAML changes that count. PNGs and
NPZs preserve the readouts and live camera matrices for later inspection.

- Blue: the actual 30-point proprio auxiliary readout at the last action flow
  evaluation. Inverse state quantile normalization and the runtime episode
  grounding origin recover world XYZ. These are **absolute positions**, not
  integrated delta actions, and are not forced to start at the current EEF.
- Red: the actual predicted action chunk, approximately integrated with the
  controller scale. The solid prefix is its execution budget.
- Yellow/green: the skill-start/end XYZ conditions, when present. The lower
  plots show grounded XYZ, axis-angle rotation, and both gripper positions.

The 30 points span normalized **full-skill phase**, not 30 future steps. Action
layers read the corresponding hidden plan tokens, not these decoded 8-D states
directly. No additional model forward or random sampling is performed, and
the captured head cache and executed actions are unchanged. The ordinary
`./submit_eval.sh` is unchanged: motion-plan capture is off unless explicitly
enabled in the YAML.
