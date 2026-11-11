# ABC Sim → LeRobot v3

This folder consumes the official `sim_224` episode release. It does not need
the MuJoCo simulator or its assets: the release already contains the recorded
14D state/action stream and a vertically stacked RGB video for each episode.

## Quick start

```bash
cd lerobot/examples/libero/configs/generate_training_dataset/download_dataset/ABC_sim_dataset

./download_ABC_sim.sh --list-tasks
./download_ABC_sim.sh --dry-run
ABC_SIM_ONLY=abc_sim_toy ./download_ABC_sim.sh
ABC_SIM_ONLY=abc_sim_toy ./build_ABC_sim_dataset.sh
```

Or run `ABC_SIM_ONLY=abc_sim_toy ./submit_build_ABC_sim.sh` to download on the
current node and submit the CPU-heavy conversion through the repository's
normal Slurm/local submission helper.

Configuration lives in `ABC_sim_dataset_config.yaml`. A local subset can merge
several official tasks and can select `train`, `val`, or both splits. Episode
limits are deterministic and apply to each selected split, but the official
server distributes one tar per task, so selecting 20 episodes still downloads
that task's entire tar. The current manifest is roughly 145 GiB compressed for
all 24 tasks; raw episodes and the final LeRobot videos coexist during a build.

## Layout

```text
dataset_ABC/
  _sim_manifest.json
  _sim_downloads/                 # resumable .part files; completed tars optional
  _sim_raw/<official-task>/       # SHA256-verified official episode directories
  _sim_v3_shards/<subset>/        # resumable conversion shards (removed on success)
  <subset>/                       # final LeRobot v3 dataset
```

The converter maps the official cameras `top,left,right` to
`top,left_wrist,right_wrist`, resizes with aspect ratio preserved and black
padding, keeps the 30 Hz 14D bimanual state/action contract, and preserves
per-frame prompt changes from `prompt_timeline`.

Downloads are checked against the official byte count and SHA256. Extraction is
staged and validated against the manifest's episode counts before it atomically
replaces a task directory. A final dataset records its exact source episode list
and task hashes in `meta/abc_sim_source.json`; changed selection requires
`FORCE=1` instead of silently reusing stale data.
