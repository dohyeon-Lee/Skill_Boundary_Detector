"""Convert between LeRobot datasets and the abcdl format (via the Episode IR).

LeRobot 0.4.4 API notes (verified against installed package):
- LeRobotDataset.create(repo_id, fps, features, root=None, vcodec='h264', ...)
- features dict: {'key': {'dtype': 'float32'/'video', 'shape': (...), 'names': ...}}
  - video features: shape=(H, W, C) HWC, dtype='video'; frames added as HWC uint8 ndarray
- add_frame(frame_dict) — frame_dict must include 'task' key (string)
- save_episode() — call after all frames for one episode are added
- finalize() — call after all episodes
- Reading back: ds[i] returns dict with torch tensors; images are CHW float32 in [0,1]
- ds.meta.camera_keys — list of camera feature keys
- ds.meta.fps — fps as int/float
- To reload a local dataset: LeRobotDataset(repo_id, root=<root>) where root is the
  directory that was passed to create() (NOT root/<repo_id>).
- abcdl_to_lerobot writes a .abcdl_meta.json sidecar in the lerobot root so that
  lerobot_to_abcdl can determine the repo_id for local datasets.
"""

from __future__ import annotations

import json
import os

import numpy as np

from abcdl.episode import CameraStream, Episode, EpisodeMeta
from abcdl.format.reader import read_abcdl
from abcdl.format.writer import write_abcdl


def _require_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError as e:
        raise ImportError("lerobot conversion needs `pip install abcdl[lerobot]`") from e


def abcdl_to_lerobot(
    abcdl_root: str,
    repo_id: str,
    lerobot_root: str | None = None,
    fps: int = 30,
) -> str:
    """Convert an abcdl dataset directory to a LeRobotDataset.

    Parameters
    ----------
    abcdl_root:
        Directory containing ``episode_*/`` subdirectories written by
        :func:`abcdl.format.writer.write_abcdl`.
    repo_id:
        HuggingFace-style ``"owner/name"`` repo ID for the LeRobot dataset.
    lerobot_root:
        Local root directory for the LeRobot dataset.  Uses the library
        default when *None*.
    fps:
        Frames-per-second for the dataset.

    Returns
    -------
    str
        Path to the created LeRobot dataset root.
    """
    LeRobotDataset = _require_lerobot()

    # Collect episode directories (must contain states_actions.bin)
    dirs = sorted(
        os.path.join(abcdl_root, d)
        for d in os.listdir(abcdl_root)
        if os.path.exists(os.path.join(abcdl_root, d, "states_actions.bin"))
    )
    if not dirs:
        raise ValueError(f"No abcdl episodes found in {abcdl_root!r}")

    first = read_abcdl(dirs[0])
    if not first.meta.cameras:
        raise ValueError("abcdl_to_lerobot: dataset has no cameras; state/action-only export not supported")
    h = first.cameras[first.meta.cameras[0]].frames.shape[1]
    w = first.cameras[first.meta.cameras[0]].frames.shape[2]

    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (first.states.shape[1],),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (first.actions.shape[1],),
            "names": None,
        },
    }
    for cam in first.meta.cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        }

    # Use h264 vcodec — works with small frames (e.g. 16x16) unlike libsvtav1
    ds = LeRobotDataset.create(
        repo_id,
        fps=fps,
        features=features,
        root=lerobot_root,
        vcodec="h264",
    )

    for d in dirs:
        ep = read_abcdl(d)
        task = ep.meta.task or ""
        for i in range(ep.num_steps):
            frame: dict = {
                "observation.state": ep.states[i].astype(np.float32),
                "action": ep.actions[i].astype(np.float32),
                "task": task,
            }
            for cam in ep.meta.cameras:
                frame[f"observation.images.{cam}"] = np.asarray(
                    ep.cameras[cam].frames[i], dtype=np.uint8
                )
            ds.add_frame(frame)
        ds.save_episode()

    ds.finalize()

    # Write a sidecar so lerobot_to_abcdl can find the repo_id for local paths
    sidecar = os.path.join(str(ds.root), ".abcdl_meta.json")
    with open(sidecar, "w") as f:
        json.dump({"repo_id": repo_id}, f)

    return str(ds.root)


def lerobot_to_abcdl(repo_id_or_root: str, out_root: str) -> list[str]:
    """Convert a LeRobotDataset to abcdl episode directories.

    Parameters
    ----------
    repo_id_or_root:
        Either a local path to a LeRobot dataset root, or a HuggingFace
        ``"owner/name"`` repo ID.  A local path is detected by
        :func:`os.path.exists`.
    out_root:
        Output directory; episodes are written to ``<out_root>/episode_<i>/``.

    Returns
    -------
    list[str]
        Sorted list of written episode directories.
    """
    LeRobotDataset = _require_lerobot()

    if os.path.exists(repo_id_or_root):
        # Local path — look for the .abcdl_meta.json sidecar written by abcdl_to_lerobot
        sidecar = os.path.join(repo_id_or_root, ".abcdl_meta.json")
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                meta_sidecar = json.load(f)
            repo_id = meta_sidecar["repo_id"]
        else:
            # Fallback: derive repo_id from last two path components
            parts = repo_id_or_root.rstrip("/").split("/")
            repo_id = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else f"local/{parts[-1]}"
        ds = LeRobotDataset(repo_id, root=repo_id_or_root)
    else:
        # Remote or already-cached repo_id
        ds = LeRobotDataset(repo_id_or_root)

    cam_keys = list(ds.meta.camera_keys)
    # Strip "observation.images." prefix to get bare camera names
    cam_names = [k.replace("observation.images.", "") for k in cam_keys]

    fps = float(ds.meta.fps)
    tick_ns = int(1e9 / fps)

    # Group frames by episode_index
    by_ep: dict[int, list] = {}
    for i in range(len(ds)):
        item = ds[i]
        ep_idx = int(item["episode_index"])
        by_ep.setdefault(ep_idx, []).append(item)

    os.makedirs(out_root, exist_ok=True)
    written: list[str] = []

    for ep_idx, items in sorted(by_ep.items()):
        T = len(items)

        states = np.stack(
            [it["observation.state"].numpy() for it in items]
        ).astype(np.float64)
        actions = np.stack(
            [it["action"].numpy() for it in items]
        ).astype(np.float64)

        ts = np.arange(T, dtype=np.int64) * tick_ns

        cams: dict[str, CameraStream] = {}
        cam_resolutions: dict[str, tuple] = {}
        cam_codecs: dict[str, str] = {}

        for cam_key, cam_name in zip(cam_keys, cam_names):
            # Images decoded by LeRobot as CHW float32 in [0, 1]
            frames_list = [
                (it[cam_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                for it in items
            ]
            frames = np.stack(frames_list)  # (T, H, W, 3) uint8
            h, w = frames.shape[1], frames.shape[2]
            cams[cam_name] = CameraStream(
                frames=frames,
                timestamps=ts,
                width=w,
                height=h,
                codec="raw",
            )
            cam_resolutions[cam_name] = (w, h)
            cam_codecs[cam_name] = "h264"

        task = str(items[0].get("task", "")) if items else ""

        meta = EpisodeMeta(
            task=task,
            fps=fps,
            cameras=cam_names,
            camera_resolutions=cam_resolutions,
            camera_codecs=cam_codecs,
            alignment=f"fixed_clock_{int(round(fps))}hz_causal",
            t0_ns=0,
            tick_ns=tick_ns,
        )

        ep = Episode(
            states=states,
            actions=actions,
            timestamps=ts,
            cameras=cams,
            meta=meta,
        )

        out_dir = os.path.join(out_root, f"episode_{ep_idx:04d}")
        write_abcdl(ep, out_dir)
        written.append(out_dir)

    return written
