from __future__ import annotations

import os
from pathlib import Path
import subprocess


HELPER = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/FSQ/src/fsq_node_local_frame_cache.sh"
)


def _run_stage(
    shared: Path,
    local_root: Path,
    *,
    slurm_mem_mb: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = (
        f"source {HELPER!s}; "
        f"fsq_stage_frame_cache_on_node {shared!s} {local_root!s} 0"
    )
    env = os.environ.copy()
    if slurm_mem_mb is not None:
        env["SLURM_MEM_PER_NODE"] = str(slurm_mem_mb)
    return subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_node_local_frame_cache_stages_sparse_tree_and_reuses_it(
    tmp_path: Path,
) -> None:
    fingerprint = "abc123"
    shared = tmp_path / "shared" / fingerprint
    frames = shared / "frames/camera/file.npy"
    frames.parent.mkdir(parents=True)
    frames.write_bytes(b"header")
    with frames.open("r+b") as stream:
        stream.seek(8 * 1024 * 1024)
        stream.write(b"tail")
    (shared / "manifest.json").write_text('{"source_fingerprint":"abc123"}\n')
    (shared / "_SUCCESS").write_text(fingerprint + "\n")

    local_root = tmp_path / "local"
    first = _run_stage(shared, local_root)
    assert first.returncode == 0, first.stderr
    staged = Path(first.stdout.strip())
    assert staged == local_root / fingerprint
    assert (staged / "manifest.json").read_bytes() == (
        shared / "manifest.json"
    ).read_bytes()
    assert (staged / "frames/camera/file.npy").read_bytes() == frames.read_bytes()
    # rsync -S must preserve the intentionally sparse test payload.
    assert os.stat(staged / "frames/camera/file.npy").st_blocks * 512 < frames.stat().st_size

    second = _run_stage(shared, local_root)
    assert second.returncode == 0, second.stderr
    assert Path(second.stdout.strip()) == staged
    assert "reusing" in second.stderr


def test_node_local_frame_cache_rejects_incomplete_shared_cache(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared/incomplete"
    shared.mkdir(parents=True)
    (shared / "manifest.json").write_text("{}\n")

    result = _run_stage(shared, tmp_path / "local")

    assert result.returncode != 0
    assert "shared cache is incomplete" in result.stderr


def test_node_local_frame_cache_rejects_insufficient_slurm_memory(
    tmp_path: Path,
) -> None:
    fingerprint = "large123"
    shared = tmp_path / "shared" / fingerprint
    payload = shared / "frames/payload.npy"
    payload.parent.mkdir(parents=True)
    with payload.open("wb") as stream:
        stream.seek(8 * 1024 * 1024)
        stream.write(b"tail")
    (shared / "manifest.json").write_text("{}\n")
    (shared / "_SUCCESS").write_text(fingerprint + "\n")

    result = _run_stage(
        shared,
        tmp_path / "local",
        slurm_mem_mb=1,
    )

    assert result.returncode != 0
    assert "Slurm memory allocation is too small" in result.stderr
