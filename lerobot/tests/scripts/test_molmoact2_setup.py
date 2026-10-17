"""Exercise the setup flow using local Git and a fake package installer (no network)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE = Path("lerobot/examples/libero/configs/train_molmoact2")
SETUP = ROOT / MODULE / "setup_env.sh"
SPEC = importlib.util.spec_from_file_location("molmo_setup_check", ROOT / MODULE / "PT/src/check_environment.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture
def setup_tree(tmp_path: Path):
    # A real local upstream, with the same training tree but no packages/models.
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git("init", cwd=upstream)
    for name in ("experiments/pyproject.toml", "experiments/lerobot/pyproject.toml",
                 "experiments/launch_scripts/train_lerobot.py"):
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original fixture\n")
    git("add", ".", cwd=upstream)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-m", "fixture", cwd=upstream)
    target = git("rev-parse", "HEAD", cwd=upstream)

    project = tmp_path / "project"
    module = project / MODULE
    module.mkdir(parents=True)
    script = SETUP.read_text()
    script = re.sub(r'SOURCE_REV="[0-9a-f]{40}"', f'SOURCE_REV="{target}"', script)
    (module / "setup_env.sh").write_text(script)
    shutil.copy(ROOT / "setup_env.sh", project / "setup_env.sh")
    shutil.copy(ROOT / MODULE / "requirements.lock.txt", module / "requirements.lock.txt")

    # Only uv and the target Python are fake. Source checkout is real Git, whose
    # official URL is rewritten to our local fixture via environment-only config.
    home = tmp_path / "home"
    uv = home / ".local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(f"#!{sys.executable}\n" + '''
import json, os
from pathlib import Path
import sys
with open(os.environ["SETUP_LOG"], "a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "path": os.environ["PATH"],
                             "pythonpath": os.environ.get("PYTHONPATH"),
                             "virtual_env": os.environ.get("VIRTUAL_ENV")}) + "\\n")
if sys.argv[1] == "venv":
    python = Path(sys.argv[2]) / "bin/python"
    python.parent.mkdir(parents=True)
    (python.parent.parent / "pyvenv.cfg").write_text("fixture")
    python.write_text("#!/bin/sh\\nexit 0\\n")
    python.chmod(0o755)
''')
    uv.chmod(0o755)
    log = tmp_path / "uv.jsonl"
    env = dict(os.environ, HOME=str(home), SETUP_LOG=str(log),
               GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0=f"url.{upstream.as_uri()}.insteadOf",
               GIT_CONFIG_VALUE_0="https://github.com/allenai/molmoact2.git",
               PYTHONPATH="/wrong/main/lerobot/src", VIRTUAL_ENV="/wrong/main/.venv")
    return project, upstream, target, log, env


def run_setup(tree, *args: str):
    project, _, _, _, env = tree
    return subprocess.run(["bash", str(project / "setup_env.sh"), "--molmoact2-only", *args],
                          env=env, text=True, capture_output=True, timeout=30)


def calls(tree):
    log = tree[3]
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_fresh_setup_checks_out_pinned_source_and_isolates_builds(setup_tree):
    project, _, target, _, _ = setup_tree
    main = project / ".venv/keep"
    main.parent.mkdir()
    main.write_text("do not touch")
    result = run_setup(setup_tree)
    assert result.returncode == 0, result.stdout + result.stderr
    source = project / "third_party/molmoact2"
    assert git("rev-parse", "HEAD", cwd=source) == target
    assert (source / "experiments/pyproject.toml").exists()  # --no-checkout clone needs checkout even at HEAD.
    assert main.read_text() == "do not touch"
    install = calls(setup_tree)
    assert install[0]["args"][0] == "venv"
    assert install[1]["args"][-1] == "cmake==3.31.10"  # BEFORE egl-probe builds.
    for call in install[1:]:
        assert call["path"].split(os.pathsep)[0] == str(project / ".venv_molmoact2/bin")
        assert call["pythonpath"] is None and call["virtual_env"] is None
        assert call["args"][call["args"].index("--python") + 1] == str(project / ".venv_molmoact2/bin/python")
    assert "cu128" in install[2]["args"]
    assert str(source / "experiments/lerobot[async,libero]") in install[2]["args"]
    assert not (project / "models").exists()


def test_existing_venv_and_same_revision_source_edits_are_preserved(setup_tree):
    assert run_setup(setup_tree).returncode == 0
    project = setup_tree[0]
    sentinel = project / ".venv_molmoact2/keep"
    sentinel.write_text("keep")
    edited = project / "third_party/molmoact2/experiments/pyproject.toml"
    edited.write_text("# user edit\n")
    result = run_setup(setup_tree)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "preserving local" in result.stdout
    assert sentinel.read_text() == "keep" and edited.read_text() == "# user edit\n"
    assert sum(call["args"][0] == "venv" for call in calls(setup_tree)) == 1


def test_check_only_never_installs_and_reports_missing_source(setup_tree):
    result = run_setup(setup_tree, "--check")
    assert result.returncode != 0 and "source missing" in result.stderr
    assert not calls(setup_tree)
    assert run_setup(setup_tree).returncode == 0
    count = len(calls(setup_tree))
    assert run_setup(setup_tree, "--check").returncode == 0
    assert len(calls(setup_tree)) == count


def test_dirty_wrong_revision_is_not_overwritten(setup_tree):
    assert run_setup(setup_tree).returncode == 0
    project, _, _, _, _ = setup_tree
    source = project / "third_party/molmoact2"
    edited = source / "experiments/pyproject.toml"
    edited.write_text("# another revision\n")
    git("add", ".", cwd=source)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-m", "other", cwd=source)
    other = git("rev-parse", "HEAD", cwd=source)
    edited.write_text("# unsaved user edit\n")
    count = len(calls(setup_tree))
    result = run_setup(setup_tree)
    assert result.returncode != 0 and "local changes" in result.stderr
    assert git("rev-parse", "HEAD", cwd=source) == other
    assert edited.read_text() == "# unsaved user edit\n"
    assert len(calls(setup_tree)) == count


def test_clean_wrong_revision_is_pinned_and_check_only_does_not_switch_it(setup_tree):
    assert run_setup(setup_tree).returncode == 0
    project, _, target, _, _ = setup_tree
    source = project / "third_party/molmoact2"
    (source / "experiments/pyproject.toml").write_text("# future revision\n")
    git("add", ".", cwd=source)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-m", "future", cwd=source)
    future = git("rev-parse", "HEAD", cwd=source)
    result = run_setup(setup_tree, "--check")
    assert result.returncode != 0 and "revision" in result.stderr
    assert git("rev-parse", "HEAD", cwd=source) == future
    assert run_setup(setup_tree).returncode == 0
    assert git("rev-parse", "HEAD", cwd=source) == target


def test_invalid_existing_venv_is_preserved_without_package_installs(setup_tree):
    project = setup_tree[0]
    marker = project / ".venv_molmoact2/keep"
    marker.parent.mkdir()
    marker.write_text("user data")
    result = run_setup(setup_tree)
    assert result.returncode != 0 and "not a usable venv" in result.stderr
    assert marker.read_text() == "user data"
    assert not calls(setup_tree)


def test_lock_reader_and_legacy_version_normalization(tmp_path, monkeypatch):
    pins = CHECK.read_pins(ROOT / MODULE / "requirements.lock.txt")
    assert len(pins) > 200 and pins["torch"] == "2.10.0+cu128"
    monkeypatch.setattr(CHECK.metadata, "version", lambda _: "0.1.1-2209072238")
    assert CHECK.package_problems({"thop": "0.1.1.post2209072238"}) == []
    invalid = tmp_path / "requirements.txt"
    invalid.write_text("torch>=2.10\n")
    with pytest.raises(ValueError, match="exact package pin"):
        CHECK.read_pins(invalid)
