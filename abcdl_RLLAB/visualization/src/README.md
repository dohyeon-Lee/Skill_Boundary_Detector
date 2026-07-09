# ABC-130k visualizer

Eyeball what you downloaded: a readable **HTML gallery** organized as
**category tabs → tasks → episodes**, each episode shown as a short **looping
animation of only the 3rd-person (top) camera** next to its **language prompt**.
Click a group tab to see just that category's tasks. Minimal footprint — one
view, sampled frames, downscaled.

The group comes from the download layout: with the downloader's
`group_subdirs: true` (`<in_dir>/<group>/data/...`) each category becomes a tab;
a flat/ungrouped download just renders as a single group (no tab bar).

## Layout

```
visualization/
├── run.sh          # ← run this
├── config.yaml     # ← edit this
└── src/
    ├── visualize_abc.py
    └── README.md
```

## How it works

Reads the raw `episode.mcap` (the downloaded format) directly:
`/instruction` → prompt, `/top-camera` (or `/top-left-camera` for ZED-X) →
frames. Decode (h264/hevc) and encode (WebP/GIF) both go through **PyAV**, so
**no system ffmpeg is required**. Output is `out_dir/index.html` +
`out_dir/assets/<task>__<uuid>.webp`.

## Setup

```bash
pip install av mcap mcap-protobuf-support pyyaml numpy   # into the project venv
```

(`uv pip install ...` also works if you have [uv](https://github.com/astral-sh/uv) — faster, optional.)

## Use

```bash
cd visualization
./run.sh --dry-run           # list episodes that would render
./run.sh                     # render  (cap count via max_episodes in config.yaml)
./run.sh --force             # re-render everything, ignoring the cache
# then open  ../abc_viz/index.html
```

**Incremental / skip-if-exists.** A tiny `.viz_manifest.json` in `out_dir` records
what was rendered. Re-running skips any episode whose asset already exists (no mcap
read, no decode/encode) — so after downloading more episodes, only the new ones
render. Changing `anim_format` / `width` / `target_frames` / `playback_fps` /
`view_priority` invalidates the cache; `--force` re-renders all.

## config.yaml knobs

| key | meaning |
|---|---|
| `in_dir` | where downloaded `episode.mcap` live (download's `out_dir`) |
| `split` | search `data/<split>/` first; `all` = everything under `in_dir` |
| `out_dir` | writes `index.html` + `assets/` here |
| `view_priority` | 3rd-person camera preference; first present wins |
| `anim_format` | `webp` (smaller) or `gif` |
| `target_frames` | frames sampled per episode (bounds size on long clips) |
| `playback_fps` | loop speed |
| `width` | downscale width (px); height keeps aspect |
| `max_episodes` | `0` = all, else cap |

> Only the top camera is decoded (wrist views skipped) to keep it tiny. Prompt
> comes from the episode's `/instruction`; falls back to episode metadata, then
> the task folder name.
