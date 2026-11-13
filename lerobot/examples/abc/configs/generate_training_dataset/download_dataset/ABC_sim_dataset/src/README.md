# ABC Sim native virtual datasets

This folder consumes the official `sim_224` episode release. It does not need
the MuJoCo simulator or its assets: the release already contains the recorded
14D state/action stream and a vertically stacked RGB video for each episode.

## Quick start

```bash
cd lerobot/examples/abc/configs/generate_training_dataset/download_dataset/ABC_sim_dataset

./download_ABC_sim.sh --list-tasks
./download_ABC_sim.sh --dry-run
./download_ABC_sim.sh
./build_ABC_sim_dataset.sh
```

Or run `./submit_build_ABC_sim.sh` to download all 24 task packages on the
current node and then submit native indexing through the repository's normal
Slurm/local submission helper. On Yonsei, indexing uses the CPU-only
`dell_cpu/cpu_qos` queue and requests no GPU; servers without a configured CPU
profile fall back to their normal Slurm queue with one GPU reservation.

Configuration lives in `ABC_sim_dataset_config.yaml`. A local subset can merge
several official tasks and can select `train`, `val`, or both splits. Episode
limits are deterministic and apply to each selected split, but the official
server distributes one tar per task, so selecting 20 episodes still downloads
that task's entire tar. The current manifest is roughly 145 GiB compressed for
all 24 tasks. Use `ABC_SIM_ONLY=abc_sim_all` for the complete training view.
`tasks: all` expands from the manifest, so the YAML does not duplicate the
official task list. The official `val_sim` demonstrations are not included in
that view: policy quality is evaluated with simulator rollouts/SR, while FSQ
already makes its own episode-grouped validation split from training skills.

## Layout

```text
dataset_ABC/
  _sim_manifest.json
  _sim_downloads/                 # resumable .part files; completed tars optional
  _sim_raw/<official-task>/       # SHA256-verified official episode directories
  <subset>/
    native_dataset.json           # portable source references + provenance
    meta/info.json                # feature/fps/count contract
    meta/tasks.parquet            # prompt -> task_index
    meta/episodes/...parquet      # episode offsets + source paths
    meta/stats.json               # state/action normalization statistics
```

The native view contains no `data/` or `videos/` directory.
`lerobot.datasets.ABCNativeDataset` is model-independent. It reads `states_actions.bin` and
`combined_camera-images-rgb.mp4` directly from `_sim_raw`, maps
`top,left,right` to `top,left_wrist,right_wrist`, and preserves mid-episode
prompt changes recorded by `prompt_timeline`. It returns each official camera
as its raw 224x168 crop. PI0.5 then performs one aspect-preserving 224x224
letterbox inside the model, identically during training and simulator evaluation.
ABC-VLA, SkillVLA, or another policy can consume the same native view and apply
its own image processor without rebuilding or duplicating the dataset.

The adapter accepts the same `delta_timestamps` contract as `LeRobotDataset`.
By default, a requested action horizon must fit completely inside its episode:
for ABC's official 30-step training chunk, the last 29 possible start frames
are excluded rather than padded. Pass
`require_complete_action_chunks=False` only when LeRobot-style boundary
clamping is explicitly desired.

When training selects source tasks through `dataset.tasks` (the `pt_tasks`
setting in the PI0.5 YAML), the adapter also replaces the full-dataset
state/action statistics with statistics computed from exactly those selected
episodes. Exact min/max/mean/std and deterministic sampled quantiles are cached
under `<dataset>/meta/selection_stats/`. The cache fingerprint includes the
resolved task list, selected episodes, source-task hashes, dimensions, and stats
algorithm settings. A filesystem lock and atomic rename make the one-time scan
safe when distributed training processes initialize together; later runs reuse
the cache.

Downloads are checked against the official byte count and SHA256. Extraction is
staged and validated against the manifest's episode counts before it atomically
replaces a task directory. A final native view records its exact source episode
list and task hashes in `native_dataset.json`; changed selection requires
`FORCE=1` instead of silently reusing stale data.

Two distinct task archives download concurrently by default (`download_workers`
in the YAML, or `--download-workers N`). Extraction and episode-count validation
remain serialized to avoid competing disk-heavy installs. Installed tasks and
valid `.part` files survive interruption and are skipped or resumed on rerun.

If a standalone LeRobot dataset is needed for export, run
`export_ABC_sim_lerobot.sh`. That optional path retains the old video conversion
and therefore needs the additional time and storage. It writes a separate
`<subset>_lerobot/` folder by default, so it never overwrites the native view.

## PI0.5 training

The native view can be trained directly without exporting a duplicated LeRobot
dataset:

```bash
cd lerobot/examples/abc/configs/train_pi05/PT
./submit_train.sh
```

`pi05_PT_config.yaml` selects `dataset_ABC/abc_sim_all`, maps the three ABC
cameras to the PI0.5 camera names, trains 30-step chunks, and records a 15-step
execution horizon for later simulator evaluation.
