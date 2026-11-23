"""Check dispatch and preservation contracts without network or simulator installs."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE = Path("lerobot/skillVLA/mimicgen/configs/src/simulator")
SPEC = importlib.util.spec_from_file_location("mimicgen_setup_check", ROOT / MODULE / "check_environment.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    module = root / MODULE
    module.mkdir(parents=True)
    shutil.copy(ROOT / "setup_env.sh", root / "setup_env.sh")
    for name in ("setup_env.sh", "source_revision.txt"):
        shutil.copy(ROOT / MODULE / name, module / name)
    main = root / ".venv"
    main.mkdir()
    (main / "preserve").write_text("existing training environment")
    return root


def run(project, *args):
    return subprocess.run(["bash", str(project / "setup_env.sh"), "--mimicgen-only", *args],
                          capture_output=True, text=True, env=os.environ.copy(), timeout=10)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True,
                                   stderr=subprocess.PIPE).strip()


def source_checkout(project):
    source = project / "third_party/mimicgen"
    source.mkdir(parents=True)
    git(source, "init", "-q")
    (source / "tracked").write_text("official source")
    git(source, "add", "tracked")
    git(source, "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
        "commit", "-qm", "fixture")
    return source, git(source, "rev-parse", "HEAD")


def test_help_and_check_never_create_an_installation(project):
    assert run(project, "--help").returncode == 0
    result = run(project, "--check")
    assert result.returncode == 1
    assert "source missing" in result.stderr
    assert not (project / "third_party").exists()
    assert not (project / ".venv_mimicgen").exists()
    assert (project / ".venv/preserve").read_text() == "existing training environment"


def test_check_rejects_other_revision_without_changing_checkout(project):
    source, before = source_checkout(project)
    result = run(project, "--check")
    assert result.returncode == 1
    assert "revision differs" in result.stderr
    assert git(source, "rev-parse", "HEAD") == before
    assert not (project / ".venv_mimicgen").exists()


def test_check_preserves_local_source_edits(project):
    source, revision = source_checkout(project)
    (project / MODULE / "source_revision.txt").write_text(revision)
    (source / "tracked").write_text("my local change")
    result = run(project, "--check")
    assert result.returncode == 1
    assert "local source changes" in result.stderr
    assert (source / "tracked").read_text() == "my local change"


def test_render_flag_reaches_existing_venv_checker_without_install(project):
    source, revision = source_checkout(project)
    (project / MODULE / "source_revision.txt").write_text(revision)
    venv = project / ".venv_mimicgen"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("fixture")
    python = venv / "bin/python"
    python.write_text('#!/bin/bash\nif [[ "$1" != "-c" ]]; then printf "%s\\n" "$@"; fi\n')
    python.chmod(0o755)
    result = run(project, "--check", "--render")
    assert result.returncode == 0, result.stderr
    assert "check_environment.py\n--render\n" in result.stdout
    assert (project / ".venv/preserve").is_file()


def test_venv_symlinks_are_rejected(project):
    source, revision = source_checkout(project)
    (project / MODULE / "source_revision.txt").write_text(revision)
    (project / ".venv_mimicgen").symlink_to(project / ".venv", target_is_directory=True)
    result = run(project, "--check")
    assert result.returncode == 1
    assert "venv symlink" in result.stderr


def test_lock_requires_exact_unique_versions(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("numpy==2.2.6\nrequests[socks]==2.34.2\n")
    assert CHECK.read_pins(path) == {"numpy": "2.2.6", "requests": "2.34.2"}
    for invalid in ("numpy>=2\n", "six==1.17.0\nsix==1.17.0\n", "# empty\n"):
        path.write_text(invalid)
        with pytest.raises(ValueError):
            CHECK.read_pins(path)


def test_existing_setup_options_are_still_advertised():
    result = subprocess.check_output(["bash", str(ROOT / "setup_env.sh"), "--help"], text=True)
    for option in ["--mimicgen-only", "--abc-sim-only", "--groot-only", "--molmoact2-only"]:
        assert option in result
