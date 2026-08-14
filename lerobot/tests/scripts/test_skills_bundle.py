from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/libero"))

from skills_bundle import (  # noqa: E402
    BUNDLE_NAME,
    build_bundle,
    bundle_path_for,
    load_skills,
)


def _write_skill(path: Path, *, episode: int, index: int, length: int) -> None:
    rng = np.random.default_rng(episode * 100 + index)
    np.savez(
        str(path),
        actions=rng.normal(size=(length, 7)).astype(np.float32),
        states=rng.normal(size=(length, 8)).astype(np.float32),
        episode_id=np.int64(episode),
        task_id=np.int64(episode % 3),
        skill_index=np.int64(index),
        frame_start=np.int64(index * length),
        frame_end=np.int64((index + 1) * length),
    )


def _make_skills_dir(root: Path, count: int = 5) -> Path:
    skills = root / "skillset" / "skills"
    skills.mkdir(parents=True)
    for i in range(count):
        _write_skill(skills / f"ep{i:03d}_skill0.npz", episode=i, index=0, length=10 + i)
    return skills


def test_bundle_round_trip_is_lossless(tmp_path: Path) -> None:
    skills = _make_skills_dir(tmp_path)
    # First load: per-file path, and the bundle gets written as a side effect.
    states_a, actions_a, metadata_a = load_skills(skills)
    assert bundle_path_for(skills).is_file()
    # Second load: bundle path.
    states_b, actions_b, metadata_b = load_skills(skills)
    assert metadata_a == metadata_b
    for a, b in zip(states_a, states_b, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(actions_a, actions_b, strict=True):
        np.testing.assert_array_equal(a, b)


def test_bundle_preserves_sorted_order_and_metadata(tmp_path: Path) -> None:
    skills = _make_skills_dir(tmp_path, count=4)
    build_bundle(skills)
    states, actions, metadata = load_skills(skills)
    assert [m["file"] for m in metadata] == sorted(m["file"] for m in metadata)
    assert [m["length"] for m in metadata] == [len(a) for a in actions]
    assert all(len(s) == len(a) for s, a in zip(states, actions, strict=True))
    assert metadata[2]["episode_id"] == 2 and metadata[2]["task_id"] == 2


def test_stale_bundle_falls_back_to_per_file(tmp_path: Path) -> None:
    skills = _make_skills_dir(tmp_path, count=3)
    build_bundle(skills)
    # A new skill file invalidates the recorded fingerprint.
    _write_skill(skills / "ep999_skill0.npz", episode=999, index=0, length=12)
    states, _, metadata = load_skills(skills)
    assert len(states) == 4
    assert any(m["episode_id"] == 999 for m in metadata)
    # The fallback rebuilt a fresh bundle covering the new file.
    fresh = np.load(str(bundle_path_for(skills)))
    assert len(fresh["files"]) == 4


def test_bundle_lives_outside_skills_dir(tmp_path: Path) -> None:
    skills = _make_skills_dir(tmp_path, count=2)
    bundle = build_bundle(skills)
    assert bundle.name == BUNDLE_NAME
    assert skills not in bundle.parents
    # rglob-based loaders must never pick the bundle up as a skill.
    assert bundle not in set(skills.rglob("*.npz"))
