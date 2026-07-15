# ABC-130k selective downloader

Pull **only the tasks / episodes you want** from the gated
[`XDOF/ABC-130k`](https://huggingface.co/datasets/XDOF/ABC-130k) release, instead
of the full 3,553 h. Optionally convert the pulled `episode.mcap` files into the
abcdl training format in one shot.

## Layout

```
download/
├── run.sh          # ← run this
├── config.yaml     # ← edit this (what to download)
└── src/
    ├── download_abc.py   # the actual downloader
    └── README.md         # this file
```

`run.sh` picks a Python (`$PYTHON` → project venv → `python3`), puts the repo
root on `PYTHONPATH` (so the optional convert step can `import abcdl`), and calls
`src/download_abc.py --config config.yaml` forwarding any extra flags.

## Setup (gated dataset — one time)

```bash
# 1) accept the license at the dataset page while logged in
# 2) authenticate + deps
huggingface-cli login          # or: export HF_TOKEN=hf_xxx
pip install pyyaml huggingface_hub
```

## Use

```bash
cd download
./run.sh --list-tasks          # discover the 197 task names
./run.sh --list-tasks --counts # + episode counts (slow)
./run.sh --dry-run             # show the plan, download nothing
./run.sh                       # download (+ convert if convert_to_abcdl: true)
```

Override the interpreter: `PYTHON=/path/to/python ./run.sh`.

## config.yaml in one glance

```yaml
groups:                       # 7 official primitive categories -> task lists
  folding: [fold_and_stack_the_towels, ...]
downloads:
  - group: folding
    episodes: 5               # 5 episodes PER TASK in the group
  - task: arrange_the_flowers_into_the_vase
    episodes: 3               # N | all | [0,1,2] | [<uuid substring>, ...]
    split: train              # optional per-entry override
group_subdirs: true           # nest output under out_dir/<group>/ (grouped by category)
convert_to_abcdl: false       # true -> mcap_to_abcdl(size=…) after download
```

Output layout depends on `group_subdirs`:

- `false` (default) — mirrors the repo: `out_dir/data/<split>/<task>/episode_<uuid>/`
- `true` — grouped by category: `out_dir/<group>/data/<split>/<task>/episode_<uuid>/`
  (each group is its own `snapshot_download` root, so HF incremental caching still
  works; bare `task:`/`tasks:` entries with no group land under `<out_dir>/misc/`).
  Converted abcdl dirs mirror it: `abcdl_out_dir/<group>/<split>/<task>/…`.

> `episodes: N` = the **first N** episodes sorted by uuid (deterministic, not
> random). For a random-ish subset, list uuids explicitly.
