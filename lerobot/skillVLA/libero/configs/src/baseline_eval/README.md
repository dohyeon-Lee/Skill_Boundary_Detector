# Isolated baseline evaluation

Shared implementation for `train_molmoact2/PT/eval` and `train_groot/PT/eval`.
No upstream policy or simulator code is copied or patched.

- The existing main `.venv` runs `lerobot.scripts.lerobot_eval.eval_policy_all`:
  identical LIBERO scene selection, task time limits, success checks, state
  conversion, exact-demo mapping, and SUCCESS/FAIL video annotations as pi05.
- A private child process uses the corresponding baseline venv and pinned
  third-party LeRobot checkout. It loads the requested **trained LeRobot
  checkpoint**, with strict weight loading and its saved pre/postprocessors.
  Their normalization statistics are not replaced by evaluation dataset stats.
- For GR00T, saved gripper ranges also determine whether NVIDIA's zero-close
  conversion is needed. Our signed [-1, 1] targets already match env commands:
  legacy checkpoints are repaired in memory, without rewriting their files or
  statistics. Native [0, 1] zero-close checkpoints retain conversion. Unsupported
  ranges fail instead of guessing. Corrected evals cannot resume pre-fix metrics.
- Grounding comes from `sbd_molmoact2.json` / `sbd_groot.json`, saved at training
  checkpoints. For `episode_start_xyz`, subtract the first observed EEF XYZ once
  per rollout, before the native preprocessor. Only proprio XYZ is grounded;
  delta-EEF actions are not differenced again.
- Native inference and full-chunk postprocessing happen in the child; the
  simulator replays `n_action_steps` raw decoded actions before requesting a new
  chunk. The action queue, grounding origin and processor state reset at each
  rollout. No predictor/terminator/skill labels are used by these baselines.
- The model process receives images/state/task through private binary pipes,
  not a network service. Only that venv imports the baseline's LeRobot. The main
  process holds no baseline weights. A timeout or model exception fails the job
  instead of returning dummy actions or silently using the CPU.

## Running

Edit the model-specific `*_eval_config.yaml` under `PT/eval`, then run there:

```bash
./submit_eval.sh --dry-run
./submit_eval.sh
```

`MOLMOACT2_EVAL_CONFIG` / `GROOT_EVAL_CONFIG` select an alternative YAML.
`BASELINE_EVAL_DRY_RUN=1` is equivalent to `--dry-run`. Dry-run freezes a config
snapshot and checks local checkpoint/runtime files, but submits no jobs.
The scripts select both existing venvs automatically; no activation or new
simulation dependencies are required. Both venvs must exist on the target server.

Use `model_defaults.checkpoint: [30000, 50000, 100000]` for a sweep, or `last` for
the largest numeric saved step. All listed models use the same ordered step
list, matching the existing side-by-side layout. Only complete trained
checkpoints are accepted, including processor state files and grounding metadata.
Base Hugging Face weights in `models/` are **not** a task-adapted PT checkpoint.
Changing the evaluation recipe requires another `output_name` so old results
are never silently mixed with new conditions.

`eval_num_gpus` is an upper bound, using the existing GPU packing planner.
`eval_max_workers_per_gpu` supports 1–5; the default is 1 because these are large
models. Each worker evaluates its tasks across model/checkpoint panels
sequentially. `max_parallel_tasks` stays 1: sharing one stateful policy/processor
between task threads would contaminate action queues and episode grounding.
Local/RunPod scheduling uses the existing `submit_job.sh`; Slurm uses a GPU job
array. Global server profiles control QoS, partitions and excluded nodes.
Batch workers locate their Python sources from the exported `PROJECT_ROOT`,
not the Slurm-spooled batch script's directory. Output/metrics directories are
created before startup; startup failures skip an empty merge and retain the
original worker exit status and traceback.

Results are under each model's `PT/eval/outputs/<output_name>`: per-panel videos,
per-task/overall JSON metrics, merged task bar plots, checkpoint sweep line
plots, side-by-side comparison videos and optional W&B logging. Merging/stitching
reuse the pi05/SkillVLA scripts. Finished worker panels are skipped on resume.

## GR00T authentication

The shared job runtime sets `HF_HOME=<project>/.cache/huggingface`. Cosmos
processor access must be authenticated there, not only in the shell's default
HF cache. GR00T tokenizer initialization is deliberately checked while loading
the model, before starting the simulation. No access-token values are logged.

## Verification limits

CPU regression tests cover checkpoint/shape validation, step-list expansion,
runtime separation, grounding/reset, decoded-action playback, IPC framing,
timeouts and resume/output contracts. A real GPU closed-loop rollout is still
needed to verify the hardware/driver/model-memory setup on each target server.
