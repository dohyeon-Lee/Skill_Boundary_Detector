"""Training-only top-camera targets for the end of each *current* action chunk.

New focus artifacts carry the recorded camera matrices in the same grounded XYZ
frame as observation.state. Older artifacts can read camera XML (not videos or
simulator states) from the existing episode-exact HDF5 source map.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .focus_projection import camera_transform_from_recorded_xml


def project_top_points(
    xyz: np.ndarray, transform: np.ndarray, *, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized project_eef, returning normalized UV and an explicit validity mask.

    Unlike the preview helper, behind-camera/non-finite points are masked instead
    of raising. Out-of-view UV is clipped only to keep the batch finite; its loss
    remains masked. The image's stored horizontal mirror matches project_eef.
    """
    points = np.asarray(xyz, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ np.asarray(transform, dtype=np.float64).T
    depth = projected[:, 2]
    safe_depth = np.where(np.isfinite(depth) & (depth > 1e-8), depth, 1.0)
    pixels = np.column_stack((width - 1 - projected[:, 0] / safe_depth,
                              projected[:, 1] / safe_depth))
    valid = (np.isfinite(points).all(axis=1) & np.isfinite(pixels).all(axis=1)
             & (depth > 1e-8) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width - 1)
             & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height - 1))
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
    uv = 2 * np.clip(pixels, [0, 0], [width - 1, height - 1]) / [max(width - 1, 1), max(height - 1, 1)] - 1
    return uv.astype(np.float32), valid


def load_top_camera_transforms(
    focus_path: Path, offsets: dict[int, np.ndarray], *, first_actions: dict[int, np.ndarray] | None = None
):
    """Get exact per-episode matrices; never guess a common LIBERO camera pose."""
    with np.load(focus_path, allow_pickle=False) as artifact:
        height, width = int(artifact["image_height"]), int(artifact["image_width"])
        if {"camera_episode_id", "camera_grounded_transform"}.issubset(artifact.files):
            matrices = {
                int(ep): matrix for ep, matrix in zip(
                    artifact["camera_episode_id"], artifact["camera_grounded_transform"], strict=True
                )
            }
            missing = set(offsets) - matrices.keys()
            if missing:
                raise ValueError(f"Chunk-end camera calibration missing episodes: {sorted(missing)[:8]}")
            return {ep: matrices[ep] for ep in offsets}, height, width
        camera_name = str(artifact["camera"].item())

    source_dir = focus_path.parent.parent
    exact_path = source_dir / "eval_init_states.npz"
    if not exact_path.is_file() and os.environ.get("SKILLVLA_SHARED_DATASET_DIR"):
        shared_focus = Path(os.environ["SKILLVLA_SHARED_DATASET_DIR"]).parent / focus_path.name
        if shared_focus.resolve() != focus_path.resolve() and shared_focus.is_file():
            return load_top_camera_transforms(shared_focus, offsets, first_actions=first_actions)
    if not exact_path.is_file():
        raise FileNotFoundError(
            f"Chunk-end Top patch needs recorded camera calibration. No matrices in {focus_path} "
            f"and no episode-exact map at {exact_path}. Re-run build_skill_focus_uv.py to embed calibration."
        )
    with np.load(exact_path, allow_pickle=False) as exact:
        required = {"episode_index", "scene_file", "demo"}
        if not required.issubset(exact.files):
            raise ValueError("Chunk-end patch requires fixed-camera HDF5 calibration or embedded focus matrices.")
        by_scene: dict[str, list[tuple[int, str]]] = {}
        for ep, scene, demo in zip(exact["episode_index"], exact["scene_file"], exact["demo"], strict=True):
            if int(ep) in offsets:
                by_scene.setdefault(str(scene), []).append((int(ep), str(demo)))
    roots = [parent / "libero_original_dataset" for parent in focus_path.parents
             if (parent / "libero_original_dataset").is_dir()]
    import h5py

    matrices = {}
    for scene, episodes in by_scene.items():
        candidates = {path.resolve() for root in roots for path in root.glob(f"*/{scene}") if path.is_file()}
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Chunk-end Top patch cannot resolve recorded camera XML for {scene!r}. "
                "Keep libero_original_dataset beside the project or rebuild skill_focus_uv.npz "
                "with embedded camera calibration before transferring this dataset."
            )
        with h5py.File(candidates.pop(), "r") as source:
            for ep, demo in episodes:
                recorded = source[f"data/{demo}"]
                xml = recorded.attrs["model_file"]
                if isinstance(xml, bytes):
                    xml = xml.decode("utf-8")
                matrix = camera_transform_from_recorded_xml(
                    str(xml), camera_name=camera_name, height=height, width=width
                )
                offset = offsets[ep]
                if first_actions is not None:
                    # Same first ordered action match as SkillEvaluationDataset.align_actions.
                    # Filtering can remove leading frames; original frame zero is NOT the origin.
                    actions = np.asarray(recorded["actions"], dtype=np.float32)
                    matches = np.flatnonzero(np.abs(actions - first_actions[ep]).mean(axis=-1) <= 1e-3)
                    if not len(matches):
                        raise ValueError(f"No exact first-action match for chunk patch episode {ep}.")
                    offset = offset + np.asarray(recorded["obs/ee_states"][int(matches[0]), :3])
                matrix[:, 3] += matrix[:, :3] @ offset
                matrices[ep] = matrix
    missing = set(offsets) - matrices.keys()
    if missing:
        raise ValueError(f"Episode-exact map lacks camera sources for {sorted(missing)[:8]}")
    return matrices, height, width


def load_dataset_top_camera_transforms(dataset, focus_path: str | Path, grounded_states=None):
    """Recorded calibration in the dataset's XYZ frame; no video/simulator needed."""
    with np.load(focus_path, allow_pickle=False) as artifact:
        embedded = "camera_grounded_transform" in artifact.files
    if embedded:
        return load_top_camera_transforms(Path(focus_path), {int(ep): np.zeros(3) for ep in dataset.episodes})
    if grounded_states is None:
        grounded_states = torch.as_tensor(np.asarray(
            dataset.hf_dataset.select_columns(["observation.state"]).with_format("numpy")[:]["observation.state"],
            dtype=np.float32,
        ))
    return _legacy_dataset_top_camera_transforms(dataset, focus_path, grounded_states)


def _legacy_dataset_top_camera_transforms(dataset, focus_path, grounded_states):
    """Cache per-frame projections, then __getitem__ selects the moving chunk endpoint.

    Both camera labels refer to the same selected endpoint state. Legacy
    grounding origins come from the first action-aligned recorded EE state;
    both observation.state and observation.states.ee_state are grounded in
    SkillVLA and must NOT be mistaken for absolute world coordinates.
    """
    indices_by_episode = {}
    offsets = {}
    state = grounded_states.numpy()
    info = dataset.meta.info
    grounding = str(info.get("proprio_grounding", "none"))
    if grounding not in {"none", "episode_start_xyz"}:
        raise ValueError(f"Unsupported chunk-end patch grounding: {grounding}")
    # Only legacy focus artifacts need action matching to recover the absolute origin.
    with np.load(focus_path, allow_pickle=False) as artifact:
        embedded = "camera_grounded_transform" in artifact.files
    actions = None
    if grounding == "episode_start_xyz" and not embedded:
        cached_actions = getattr(dataset, "_canonical_action_cache", None)
        actions = (cached_actions.numpy() if cached_actions is not None else np.asarray(
            dataset.hf_dataset.select_columns(["action"]).with_format("numpy")[:]["action"], dtype=np.float32
        ))
    first_actions = {} if actions is not None else None
    relative = dataset.reader._absolute_to_relative_idx
    for ep in dataset.episodes:
        metadata = dataset.meta.episodes[int(ep)]
        start = int(torch.as_tensor(metadata["dataset_from_index"]).item())
        length = int(torch.as_tensor(metadata["length"]).item())
        indices = np.arange(start, start + length)
        if relative is not None:
            indices = np.asarray([relative[int(index)] for index in indices])
        indices_by_episode[int(ep)] = indices
        offsets[int(ep)] = -state[indices[0], :3] if actions is not None else np.zeros(3)
        if first_actions is not None:
            first_actions[int(ep)] = actions[indices[0]]
    return load_top_camera_transforms(Path(focus_path), offsets, first_actions=first_actions)


def cache_top_focus(dataset, focus_path: str | Path, grounded_states: torch.Tensor):
    """Cache normalized moving chunk endpoints with the same recorded calibration."""
    matrices, height, width = load_dataset_top_camera_transforms(dataset, focus_path, grounded_states)
    state = grounded_states.numpy()
    uv = np.zeros((len(state), 2), dtype=np.float32)
    valid = np.zeros(len(state), dtype=np.bool_)
    relative = dataset.reader._absolute_to_relative_idx
    for ep in dataset.episodes:
        metadata = dataset.meta.episodes[int(ep)]
        start = int(torch.as_tensor(metadata["dataset_from_index"]).item())
        length = int(torch.as_tensor(metadata["length"]).item())
        indices = np.arange(start, start + length)
        if relative is not None:
            indices = np.asarray([relative[int(index)] for index in indices])
        uv[indices], valid[indices] = project_top_points(
            state[indices, :3], matrices[ep], height=height, width=width
        )
    return torch.from_numpy(uv), torch.from_numpy(valid)
