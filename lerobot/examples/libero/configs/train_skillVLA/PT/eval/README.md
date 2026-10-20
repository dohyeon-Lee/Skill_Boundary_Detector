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

Checkpoint lists retain the existing bar charts and generate
`checkpoint_success_rates.png` plus one task-wise line chart per model under
`checkpoint_task_success_rates/`.

`eval_max_workers_per_gpu` accepts 1–5 independent evaluation processes per GPU.
The default stays 4. Each worker loads its own model and simulator, so a value
of 5 requires enough GPU/host memory and does not guarantee higher throughput.
With 10 tasks, `eval_num_gpus: 10` and `eval_max_workers_per_gpu: 5` request only
2 physical GPUs, with 5 workers each; `eval_num_gpus` is an upper bound.
PT and FT evaluation share this limit and packing code.

## Bspline proprio visualization (one episode per task)

```bash
./submit_bspline_visualization.sh
```

This reads the **same** `stage1_eval_config.yaml` and snapshots it normally.
Without editing that YAML, the visualization resolver uses one episode per
selected task/model/checkpoint and batch size 1. GPU allocation and worker
packing follow the shared `eval_num_gpus` and `eval_max_workers_per_gpu`
settings. For 10 tasks, values of 10 and 2 request 5 GPUs; values of 2 and 2
request 2 GPUs and process the remaining tasks sequentially within each worker.
Each worker runs model/checkpoint panels sequentially. All model, Terminator,
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
