from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess


HELPER = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/FSQ/src/fsq_node_local_venv.sh"
)


def _archive_fake_venv(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "shared_venv"
    python = source / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env bash\nexit 0\n")
    python.chmod(0o755)
    (source / "marker.txt").write_text("exact environment\n")
    archive = tmp_path / "venv-test123.tar.zst"
    subprocess.run(
        [
            "bash",
            "-c",
            f"tar -C {shlex.quote(str(source))} -cf - . "
            f"| zstd -1 -q -o {shlex.quote(str(archive))}",
        ],
        check=True,
    )
    (Path(str(archive) + ".size")).write_text("4096\n")
    return archive, source


def _stage(archive: Path, fallback: Path, local_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("SLURM_TMPDIR", None)
    env["FSQ_VENV_LOCAL_ROOT"] = str(local_root)
    command = (
        f"source {shlex.quote(str(HELPER))}; "
        f"fsq_stage_venv_on_node {shlex.quote(str(archive))} "
        f"{shlex.quote(str(fallback))}"
    )
    return subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_node_local_venv_uses_explicit_root_without_slurm_tmpdir_and_reuses(
    tmp_path: Path,
) -> None:
    archive, fallback = _archive_fake_venv(tmp_path)
    local_root = tmp_path / "local"

    first = _stage(archive, fallback, local_root)
    assert first.returncode == 0, first.stderr
    staged = Path(first.stdout.strip())
    assert staged == local_root / "venv-test123"
    assert os.access(staged / "bin/python", os.X_OK)
    assert (staged / "marker.txt").read_text() == "exact environment\n"
    assert (staged / ".fsq_venv_archive").read_text().strip() == archive.name

    second = _stage(archive, fallback, local_root)
    assert second.returncode == 0, second.stderr
    assert Path(second.stdout.strip()) == staged
    assert "reusing" in second.stderr


def test_node_local_venv_falls_back_when_size_metadata_exceeds_capacity(
    tmp_path: Path,
) -> None:
    archive, fallback = _archive_fake_venv(tmp_path)
    Path(str(archive) + ".size").write_text(str(2**63 - 1) + "\n")

    result = _stage(archive, fallback, tmp_path / "local")

    assert result.returncode != 0
    assert "insufficient local space" in result.stderr
