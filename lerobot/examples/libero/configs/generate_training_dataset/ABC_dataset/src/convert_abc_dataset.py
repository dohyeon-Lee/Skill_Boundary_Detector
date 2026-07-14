#!/usr/bin/env python3
"""Convert staged ABC mcap episodes → abcdl cache → LeRobot v3 dataset (+ quantile stats).

Pipeline per subset name (see ABC_dataset_config.yaml):
  ② mcap→abcdl : abcdl.convert.mcap_abcdl.mcap_to_abcdl — 30 Hz fixed-clock resample +
                 square downscale + stacked-mp4 cache. Episode-parallel (ffmpeg-heavy).
                   {abc_root}/_mcap/{name}/**/episode.mcap → {abc_root}/_abcdl/{name}/<ep>/
  ③ abcdl→v3  : OUR pyav-based reader (torchcodec NOT required — the cluster lacks system
                 libav; pyav ships its own) feeding lerobot's create/add_frame/save_episode/
                 finalize. Mirrors abcdl.convert.lerobot.abcdl_to_lerobot faithfully
                 (same features dict / HWC uint8 frames / task string).
                   → {abc_root}/{name}  (genuine v3: per-camera mp4 + parquet + stats)
  ④ stats     : filtered_dataset/ensure_quantile_stats.py --root {abc_root} (fast mode:
                 non-video quantiles from parquet).

Idempotent: abcdl episodes with episode_metadata.json are skipped (written via tmp-dir +
rename, so a crashed run never leaves a "done-looking" partial); a finished v3 dataset
(meta/info.json) is skipped unless FORCE=1/--force (which deletes ONLY {abc_root}/{name}).

Requires: `uv pip install mcap "mcap-protobuf-support>=0.5,<0.6" foxglove-schemas-protobuf`
(pure-python; checked with an exact hint at startup). ffmpeg CLI is auto-shimmed from the
imageio_ffmpeg bundled binary when absent from PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from ABC_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    abc_root,
    abcdl_repo as _abcdl_repo,
    load_config,
    project_root,
    subsets,
)


# ── environment plumbing ──────────────────────────────────────────────────────


def _ensure_ffmpeg(tools_dir: Path) -> None:
    """abcdl shells out to `ffmpeg`; if absent from PATH, shim the imageio_ffmpeg binary."""
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "ffmpeg not in PATH and imageio_ffmpeg unavailable — install ffmpeg or "
            "`uv pip install imageio-ffmpeg`") from e
    tools_dir.mkdir(parents=True, exist_ok=True)
    shim = tools_dir / "ffmpeg"
    if not shim.exists():
        shim.symlink_to(exe)
    os.environ["PATH"] = f"{tools_dir}:{os.environ.get('PATH', '')}"
    print(f"[env] ffmpeg shim: {shim} → {exe}")


def _setup_paths(cfg: dict) -> tuple[Path, Path]:
    """sys.path + PYTHONPATH for abcdl repo and lerobot/src (also for child procs).

    Import/dep enforcement lives in stage_mcap_to_abcdl — the abcdl-entry path (pre-built
    abcdl episodes, ② skipped) needs neither abcdl nor the mcap deps nor ffmpeg."""
    abcdl_repo = _abcdl_repo(cfg)  # relative → resolved against project_root
    lerobot_src = project_root(cfg) / "lerobot" / "src"
    for p in (abcdl_repo, lerobot_src):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ["PYTHONPATH"] = ":".join(
        [str(lerobot_src), str(abcdl_repo)]
        + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    return abcdl_repo, lerobot_src


def _require_stage_a_deps(abcdl_repo: Path) -> None:
    """Fail fast with exact prescriptions — only the mcap→abcdl stage needs these."""
    try:
        import abcdl  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"abcdl import failed from {abcdl_repo} — check abcdl_repo in the yaml") from e
    try:
        import mcap  # noqa: F401
        import mcap_protobuf  # noqa: F401
        import foxglove_schemas_protobuf  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"missing dep ({e.name}) — run:\n"
            '  uv pip install mcap "mcap-protobuf-support>=0.5,<0.6" foxglove-schemas-protobuf') from e


# ── ② mcap → abcdl (episode-parallel) ─────────────────────────────────────────


def _worker_init(abcdl_repo: str, path_env: str) -> None:
    sys.path.insert(0, abcdl_repo)
    os.environ["PATH"] = path_env  # ffmpeg shim must reach the subprocess calls


def _convert_one_mcap(mcap_path: str, out_dir: str, size: int) -> str:
    out = Path(out_dir)
    if (out / "episode_metadata.json").exists():
        return f"skip {out.name}"
    from abcdl.convert.mcap_abcdl import mcap_to_abcdl
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)  # leftover from a crashed run
    mcap_to_abcdl(mcap_path, str(tmp), size=size)
    tmp.rename(out)  # atomic "done" marker: final dir only appears complete
    return f"done {out.name}"


def _ep_dir_name(rel: Path) -> str:
    """data/train/<task>/<ep>/episode.mcap → '<task>__<ep>' (flat file → its stem)."""
    parts = list(rel.parts[:-1])
    if parts[:1] == ["data"]:
        parts = parts[2:]  # drop data/<split>
    return "__".join(parts) if parts else rel.stem


def stage_mcap_to_abcdl(name: str, mcap_root: Path, abcdl_out: Path, size: int, workers: int,
                        abcdl_repo: Path, tools_dir: Path) -> None:
    _require_stage_a_deps(abcdl_repo)
    _ensure_ffmpeg(tools_dir)  # abcdl decode/encode shells out to `ffmpeg`
    mcaps = sorted(mcap_root.rglob("episode.mcap")) or sorted(
        p for p in mcap_root.rglob("*.mcap") if p.name != "annotation.mcap")
    if not mcaps:
        raise SystemExit(f"[{name}] no mcap files under {mcap_root} — run download_ABC.sh first")
    jobs = [(str(m), str(abcdl_out / _ep_dir_name(m.relative_to(mcap_root))), size) for m in mcaps]
    abcdl_out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] ② mcap→abcdl: {len(jobs)} episodes, {workers} workers → {abcdl_out}")
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(sys.path[0], os.environ["PATH"])) as pool:
        futs = {pool.submit(_convert_one_mcap, *j): j[0] for j in jobs}
        for fut in as_completed(futs):
            msg = fut.result()  # worker exceptions propagate here (fail loud, keep cache clean)
            n_done += 1
            print(f"  [{n_done}/{len(jobs)}] {msg}", flush=True)


# ── ③ abcdl → LeRobot v3 (pyav, no torchcodec) ────────────────────────────────


def _abcdl_episode_dirs(abcdl_dir: Path) -> list[Path]:
    """COMPLETE abcdl episode dirs, sorted: states_actions.bin + episode_metadata.json both
    present (partial copies / crashed writes excluded), .tmp staging dirs excluded."""
    if not abcdl_dir.is_dir():
        return []
    return sorted(d for d in abcdl_dir.iterdir()
                  if d.is_dir() and not d.name.endswith(".tmp")
                  and (d / "states_actions.bin").exists()
                  and (d / "episode_metadata.json").exists())


def _load_ep_meta(ep_dir: Path) -> dict:
    return json.loads((ep_dir / "episode_metadata.json").read_text())


def _camera_row_slices(meta: dict) -> list[tuple[str, int, int]]:
    """[(cam, row_lo, row_hi)] of the vertically stacked combined mp4, in stack order."""
    slices, off = [], 0
    for cam in meta["cameras"]:
        w, h = meta["camera_resolutions"][cam]
        slices.append((cam, off, off + int(h)))
        off += int(h)
    return slices


def stage_abcdl_to_v3(name: str, abcdl_dir: Path, final_dir: Path, fps: int, force: bool) -> None:
    if (final_dir / "meta" / "info.json").exists():
        if not force:
            print(f"[{name}] ③ v3 already built → {final_dir}  (FORCE=1 to rebuild)")
            return
        print(f"[{name}] ③ FORCE: removing {final_dir}")
        shutil.rmtree(final_dir)
    elif final_dir.exists():
        raise SystemExit(f"[{name}] {final_dir} exists but has no meta/info.json (crashed build?) — "
                         "remove it or rerun with FORCE=1")

    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ep_dirs = _abcdl_episode_dirs(abcdl_dir)
    if not ep_dirs:
        raise SystemExit(f"[{name}] no abcdl episodes under {abcdl_dir}")

    first = _load_ep_meta(ep_dirs[0])
    state_dim, action_dim = int(first["state_dim"]), int(first["action_dim"])
    cams = list(first["cameras"])
    if int(round(float(first.get("fps", fps)))) != fps:
        raise SystemExit(f"[{name}] abcdl fps {first.get('fps')} != configured abc_fps {fps}")

    # Mirrors abcdl.convert.lerobot.abcdl_to_lerobot features exactly (video=HWC).
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    for cam in cams:
        w, h = first["camera_resolutions"][cam]
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (int(h), int(w), 3), "names": ["height", "width", "channel"],
        }
    print(f"[{name}] ③ abcdl→v3: {len(ep_dirs)} episodes, state/action {state_dim}/{action_dim}D, "
          f"cams {cams} → {final_dir}")

    ds = LeRobotDataset.create(f"dohyeon/{name}", fps=fps, features=features,
                               root=str(final_dir), vcodec="h264")
    for n_ep, ep_dir in enumerate(ep_dirs, 1):
        meta = _load_ep_meta(ep_dir)
        if list(meta["cameras"]) != cams or int(meta["state_dim"]) != state_dim:
            raise SystemExit(f"[{name}] heterogeneous episode {ep_dir.name}: cams/dims differ from "
                             f"{ep_dirs[0].name} — split into separate subsets")
        T = int(meta["num_steps"])
        sa = np.fromfile(ep_dir / "states_actions.bin", dtype="<f8").reshape(T, state_dim + action_dim)
        task = meta.get("task_name") or ""
        rows = _camera_row_slices(meta)
        n_frames = 0
        with av.open(str(ep_dir / "combined_camera-images-rgb.mp4")) as container:
            for frame in container.decode(video=0):  # stream: never holds the full episode in RAM
                arr = frame.to_ndarray(format="rgb24")
                item = {
                    "observation.state": sa[n_frames, :state_dim].astype(np.float32),
                    "action": sa[n_frames, state_dim:].astype(np.float32),
                    "task": task,
                }
                for cam, lo, hi in rows:
                    item[f"observation.images.{cam}"] = np.ascontiguousarray(arr[lo:hi])
                ds.add_frame(item)
                n_frames += 1
        if n_frames != T:
            raise SystemExit(f"[{name}] {ep_dir.name}: decoded {n_frames} frames != num_steps {T} "
                             "(corrupt abcdl episode — delete its dir and rerun ②)")
        ds.save_episode()
        print(f"  [{n_ep}/{len(ep_dirs)}] {ep_dir.name}: {T} frames  task={task!r}", flush=True)
    ds.finalize()
    # Sidecar so abcdl.convert.lerobot.lerobot_to_abcdl can round-trip this dataset if ever needed.
    (final_dir / ".abcdl_meta.json").write_text(json.dumps({"repo_id": f"dohyeon/{name}"}))


# ── ④ quantile stats (reuse filtered_dataset tool) ────────────────────────────


def stage_stats(name: str, root: Path, config_path: Path) -> None:
    script = SRC_DIR.parent.parent / "filtered_dataset" / "ensure_quantile_stats.py"
    cmd = [sys.executable, str(script), "--config", str(config_path),
           "--dataset", name, "--root", str(root)]
    print(f"[{name}] ④ stats: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", default=os.environ.get("ABC_ONLY", ""),
                        help="Space/comma-separated subset names (default: all in yaml)")
    parser.add_argument("--workers", type=int, default=None, help="mcap→abcdl process count")
    parser.add_argument("--force", action="store_true",
                        default=os.environ.get("FORCE", "") not in ("", "0"),
                        help="Rebuild the final v3 dataset (deletes {abc_root}/{name})")
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = abc_root(cfg)
    size = int(cfg.get("abc_image_size", 256))
    fps = int(cfg.get("abc_fps", 30))
    workers = args.workers or int(cfg.get("convert_workers", 4))

    specs = subsets(cfg)
    only = {s for s in args.only.replace(",", " ").split() if s}
    if only:
        unknown = only - set(specs)
        if unknown:
            raise SystemExit(f"unknown subset(s) {sorted(unknown)}; yaml has {sorted(specs)}")
        specs = {k: v for k, v in specs.items() if k in only}
    if not specs:
        raise SystemExit("no abc_subsets configured")

    abcdl_repo, _ = _setup_paths(cfg)
    sys.path.insert(0, str(abcdl_repo))  # sys.path[0] handed to workers

    for name in specs:
        print(f"\n════ subset {name} ════")
        mcap_root, abcdl_dir = root / "_mcap" / name, root / "_abcdl" / name
        # 진입점 2개: (a) mcap 스테이징 있음 → ②부터. (b) mcap 없이 abcdl 에피소드가 이미
        # 준비됨(예: 별도 다운로드 파이프라인 산출물을 _abcdl/{name}/에 배치) → ② 스킵, ③부터.
        if mcap_root.is_dir() and any(mcap_root.rglob("*.mcap")):
            stage_mcap_to_abcdl(name, mcap_root, abcdl_dir, size, workers,
                                abcdl_repo, root / "_tools")
        else:
            n_ready = len(_abcdl_episode_dirs(abcdl_dir))
            if n_ready == 0:
                raise SystemExit(
                    f"[{name}] neither mcap staging ({mcap_root}) nor abcdl episodes ({abcdl_dir}) "
                    "found — run download_ABC.sh, or place pre-built abcdl episode dirs under "
                    f"{abcdl_dir}/")
            print(f"[{name}] ② skip: no mcap staging — using {n_ready} existing abcdl episodes")
        stage_abcdl_to_v3(name, abcdl_dir, root / name, fps, args.force)
        if not args.skip_stats:
            stage_stats(name, root, args.config)

        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        m = LeRobotDatasetMetadata(f"dohyeon/{name}", root=str(root / name))
        print(f"[{name}] ✅ v3 ready: {m.total_episodes} episodes / {m.total_frames} frames, "
              f"fps {m.fps}, features: {sorted(m.features)}")


if __name__ == "__main__":
    main()
