# ABC Sim native virtual datasets

This folder consumes the official `sim_224` episode release. It does not need
the MuJoCo simulator or its assets: the release already contains the recorded
14D state/action stream and a vertically stacked RGB video for each episode.

## Quick start

```bash
cd lerobot/examples/libero/configs/generate_training_dataset/download_dataset/ABC_sim_dataset

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

The native view contains no `data/` or `videos/` directory. A training-time
adapter reads `states_actions.bin` and `combined_camera-images-rgb.mp4` from
`_sim_raw`, maps `top,left,right` to `top,left_wrist,right_wrist`, and preserves
mid-episode prompt changes recorded by `prompt_timeline`. The configured square
resize is also applied by that adapter, not by this build.

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
