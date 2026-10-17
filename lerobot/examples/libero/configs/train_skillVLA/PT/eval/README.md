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
