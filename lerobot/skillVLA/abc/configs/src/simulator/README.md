# ABC Simulator environment

The official `amazon-far/abc` source is pinned in `third_party/abc` and installed
editable into `.venv_abc_sim`. This environment is isolated from the main SBD,
GR00T, and MolmoAct2 environments because the official simulator pins its own
PyTorch 2.11 / CUDA 12.8 / MuJoCo-Warp stack.

From the project root:

```bash
bash setup_env.sh --abc-sim-only
bash setup_env.sh --abc-sim-only --check

# Include ABC Sim while rebuilding the main environment on a new server.
SBD_SETUP_ABC_SIM=1 bash setup_env.sh
```

The installer prepares code, Python dependencies, the minimal LeRobot PI0.5
inference runtime, all standard simulator asset packages, and both RoboCasa
object packs. A successful `--check` verifies the official package markers,
PI0.5 imports, extracted paths, RoboCasa directories, and an actual asset-free
MuJoCo step.

Episode datasets, policy checkpoints, and standalone DINO weights remain
explicit operations. For example, a task dataset can be prepared with:

```bash
cd third_party/abc
../../.venv_abc_sim/bin/python prepare.py --sim-data sim_put_the_plastic_bottles_in_the_bin
```

The source revision and every non-editable Python package are pinned, while an
existing dirty checkout at another revision is rejected instead of overwritten.

`resolver.constraints.txt` keeps `mjviser` at 0.0.16. Its next release requires
the MuJoCo 4 `mujoco-core` distribution, which overwrites files from ABC's
explicitly pinned MuJoCo 3.8 package and breaks its Python ABI. The environment
checker performs a real asset-free MuJoCo step so this conflict cannot pass as a
successful installation.

## Closed-loop evaluation

`../../train_pi05/PT/eval/` contains the ABC Sim rollout runner for PI0.5. It supports the
PI0.5 adapter and a `hold` diagnostic policy. The PI0.5 adapter applies the
checkpoint's saved 14D state/action quantile statistics (dataset statistics are
the fallback for a base checkpoint without saved processors), reproduces PI0.5 state-prompt
tokenization, maps the top/left/right cameras to the pretrained camera names,
and executes absolute 14D joint-action chunks.

```bash
cd lerobot/skillVLA/abc/configs/train_pi05/PT/eval
./submit_eval.sh

# Resolve config, including the selected server profile, without submitting.
ABC_EVAL_DRY_RUN=1 ./submit_eval.sh

# Local config inspection only (does not construct a simulator).
../../../../../../../.venv_abc_sim/bin/python src/run_eval.py --dry-run
```

Edit `abc_sim_eval_config.yaml` to select explicit task aliases or `tasks: all`.
`all` is resolved from the selected dataset's `native_dataset.json`.
`backend: mujoco` keeps independent randomized CPU scenes and batches their
observations through one GPU policy. `parallel_worlds` controls inference batch
size, not the number of model copies; the final batch can be smaller.
This supports tasks such as `grab_clutter`, whose object counts and meshes vary
between worlds. `backend: auto` falls back to independent MuJoCo scenes for
unsupported runtimes or incompatible MJWarp scene topologies. Slurm partition, QoS, node list, and
excluded nodes always come from the active server profile (for Yonsei,
`lerobot/skillVLA/servers/yonsei.yaml`).

The current config evaluates the ABC `grab_clutter` PI0.5 50k checkpoint:
50 randomized episodes, four parallel scenes, at most 3540 actions per episode.
Results and videos default to `train_pi05/PT/eval/outputs/<run_name>/`, alongside
the evaluation config and launcher, independently of the training output root.
`model_chunk_size: 30` and `action_chunk_size: 15` retain the training convention.
The policy uses synchronous replanning; these numbers should not be labeled as
the official RTC benchmark.

Each completed MuJoCo episode is committed to `metrics/tasks/<task>/world_NNN.json`.
Re-running the same output skips completed episodes; changing checkpoint,
seed or horizon requires another `output.run_name`. A smaller `parallel_worlds`
is permitted when resuming after OOM. `metrics/summary.json` reports final success
rates; `metrics/partial_summary.json` reports completed episodes while running.

The run folder contains:

```text
index.html                      # Success/failure filter, language search, videos
task_success_rates.png           # Per-task success rate and counts
episode_steps.png                # Action count per episode, colored by outcome
metrics/                        # Config, overall/task summaries, episode records
panels/00_pi05_50k/videos/       # Videos with an outcome bar and task instruction
panels/00_pi05_50k/videos_raw/   # Original camera frames, retained unchanged
```

The HTML and figures update after each completed episode. Video banners follow
the LIBERO convention: green `SUCCESS` or red `FAIL` above the images and the
native language prompt below. They show the final episode outcome throughout the
video, not a per-frame success signal. Annotation is CPU-only and preserves frame
count and FPS. The outcome JSON is saved first, so a video-processing failure can
resume without rerunning the completed episode.

To organize an existing **idle** run and generate the same report without a GPU:

```bash
.venv_abc_sim/bin/python lerobot/skillVLA/abc/configs/train_pi05/PT/eval/src/report.py \
  lerobot/skillVLA/abc/configs/train_pi05/PT/eval/outputs/<run_name> --workers 4
```

Inside an **already allocated GPU shell**, use the existing entry point directly
instead of requesting another allocation:

```bash
bash lerobot/skillVLA/abc/configs/train_pi05/PT/eval/src/eval.sbatch
```

This entry point enables the shared newest-run VRAM guard with 0.3 GiB reserve
and 0.15 GiB stop threshold. It stops only the newest execution on pressure and
does not automatically retry an OOM. There is no per-world allocator cap.
For a separate short integration test, append
`--num-worlds 2 --max-steps 2 --run-name pi05_grab_clutter_50k_integration_smoke`.
