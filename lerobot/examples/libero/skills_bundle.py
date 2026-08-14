"""Pack per-skill .npz files into one bundle for fast startup on Lustre.

Opening 11k+ small .npz files costs one metadata RPC each on a Lustre mount,
which can take many minutes on a cold compute node. This module packs the
exact same content into a single uncompressed ``skills_bundle.npz`` next to
the ``skills/`` directory so later runs stream it in seconds.

Losslessness/consistency contract:
  - Skills are stored in ``sorted(skills_dir.rglob("*.npz"))`` order — the
    same order the per-file loader uses, so train/val split fingerprints are
    unchanged.
  - The bundle records a fingerprint of the sorted relative file list. On
    load it is revalidated against the live directory listing (a readdir is
    cheap; per-file opens are not) and a stale bundle is ignored.
  - Building is best-effort and atomic (tmp file + rename), so concurrent
    jobs may race but never corrupt or half-write the bundle.

CLI (pre-pack manually, e.g. from a login node with a warm cache):
    python examples/libero/skills_bundle.py /path/to/skillset/skills
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_NAME = "skills_bundle.npz"
BUNDLE_VERSION = 1

_METADATA_INT_KEYS = ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length")


def bundle_path_for(skills_dir: Path) -> Path:
    """The bundle lives NEXT TO skills/, never inside it (rglob must not see it)."""
    return Path(skills_dir).parent / BUNDLE_NAME


def _sorted_relative_names(skills_dir: Path) -> list[str]:
    return [str(f.relative_to(skills_dir)) for f in sorted(Path(skills_dir).rglob("*.npz"))]


def _fingerprint(names: list[str]) -> str:
    return hashlib.sha1("\n".join(names).encode()).hexdigest()


def build_bundle(skills_dir: Path, bundle_path: Path | None = None) -> Path:
    """Read every per-skill npz and write the single-file bundle atomically."""
    skills_dir = Path(skills_dir)
    bundle_path = bundle_path or bundle_path_for(skills_dir)
    names = _sorted_relative_names(skills_dir)
    if not names:
        raise FileNotFoundError(f"No .npz files in {skills_dir}")

    actions_list: list[np.ndarray] = []
    states_list: list[np.ndarray] = []
    metadata_cols: dict[str, list[int]] = {key: [] for key in _METADATA_INT_KEYS}
    for name in names:
        d = np.load(str(skills_dir / name))
        actions = d["actions"].astype(np.float32)
        states = d["states"].astype(np.float32)
        actions_list.append(actions)
        states_list.append(states)
        metadata_cols["episode_id"].append(int(d["episode_id"]))
        metadata_cols["task_id"].append(int(d["task_id"]) if "task_id" in d else -1)
        metadata_cols["skill_index"].append(int(d["skill_index"]))
        metadata_cols["frame_start"].append(int(d["frame_start"]))
        metadata_cols["frame_end"].append(int(d["frame_end"]))
        metadata_cols["length"].append(len(actions))

    payload: dict[str, np.ndarray] = {
        "bundle_version": np.int64(BUNDLE_VERSION),
        "fingerprint": np.asarray(_fingerprint(names)),
        "files": np.asarray(names),
        "actions_cat": np.concatenate(actions_list),
        "states_cat": np.concatenate(states_list),
        "actions_len": np.asarray([len(a) for a in actions_list], dtype=np.int64),
        "states_len": np.asarray([len(s) for s in states_list], dtype=np.int64),
    }
    for key, values in metadata_cols.items():
        payload[f"meta_{key}"] = np.asarray(values, dtype=np.int64)

    # Uncompressed savez: load speed matters far more than ~250MB of Lustre space.
    tmp = bundle_path.with_name(f"{bundle_path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez(str(tmp), **payload)
        os.replace(str(tmp), str(bundle_path))
    finally:
        tmp.unlink(missing_ok=True)
    return bundle_path


def _load_bundle(
    skills_dir: Path, bundle_path: Path
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]] | None:
    """Return (states, actions, metadata) from a valid bundle, else None."""
    data = np.load(str(bundle_path))
    if int(data["bundle_version"]) != BUNDLE_VERSION:
        print(f"[skills] ignoring bundle with version {int(data['bundle_version'])}")
        return None
    live_names = _sorted_relative_names(skills_dir)
    if str(data["fingerprint"]) != _fingerprint(live_names):
        print("[skills] bundle is stale (skills/ contents changed); falling back to per-file load")
        return None

    files = [str(name) for name in data["files"]]
    actions_cat, states_cat = data["actions_cat"], data["states_cat"]
    actions_len, states_len = data["actions_len"], data["states_len"]
    action_offsets = np.concatenate([[0], np.cumsum(actions_len)])
    state_offsets = np.concatenate([[0], np.cumsum(states_len)])
    meta_cols = {key: data[f"meta_{key}"] for key in _METADATA_INT_KEYS}

    states, actions, metadata = [], [], []
    for i, name in enumerate(files):
        actions.append(np.array(actions_cat[action_offsets[i] : action_offsets[i + 1]]))
        states.append(np.array(states_cat[state_offsets[i] : state_offsets[i + 1]]))
        metadata.append({
            "file": Path(name).name,
            **{key: int(meta_cols[key][i]) for key in _METADATA_INT_KEYS},
        })
    return states, actions, metadata


def load_skills(
    skills_dir: Path | str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Load all skills as (states, actions, metadata), bundle-first.

    Falls back to the per-file layout and then (best-effort) writes the bundle
    so the NEXT run starts fast. Content and ordering are identical either way.
    """
    skills_dir = Path(skills_dir)
    bundle_path = bundle_path_for(skills_dir)
    if bundle_path.is_file():
        start = time.perf_counter()
        loaded = _load_bundle(skills_dir, bundle_path)
        if loaded is not None:
            states, actions, metadata = loaded
            print(
                f"[skills] loaded {len(states)} skills from bundle "
                f"{bundle_path.name} in {time.perf_counter() - start:.1f}s"
            )
            return states, actions, metadata

    start = time.perf_counter()
    names = _sorted_relative_names(skills_dir)
    if not names:
        raise FileNotFoundError(f"No .npz files in {skills_dir}")
    states, actions, metadata = [], [], []
    for name in names:
        d = np.load(str(skills_dir / name))
        a = d["actions"].astype(np.float32)
        s = d["states"].astype(np.float32)
        actions.append(a)
        states.append(s)
        metadata.append({
            "file": Path(name).name,
            "episode_id": int(d["episode_id"]),
            "task_id": int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end": int(d["frame_end"]),
            "length": len(a),
        })
    print(f"[skills] loaded {len(states)} skills per-file in {time.perf_counter() - start:.1f}s")
    try:
        built = build_bundle(skills_dir, bundle_path)
        print(f"[skills] wrote bundle for fast future startups: {built}")
    except OSError as error:  # read-only dir, quota, concurrent cleanup — never fatal
        print(f"[skills] could not write bundle (continuing without): {error}")
    return states, actions, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pack a skills/ directory into one bundle.")
    parser.add_argument("skills_dir", type=Path)
    args = parser.parse_args()
    path = build_bundle(args.skills_dir)
    size_mb = path.stat().st_size / 1e6
    print(f"[skills] bundle written: {path} ({size_mb:.0f} MB)")
