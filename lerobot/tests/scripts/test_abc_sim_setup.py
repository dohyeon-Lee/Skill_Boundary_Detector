"""Exercise the isolated ABC Simulator setup without network or real installs."""
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
MODULE = Path("lerobot/examples/abc/configs/src/simulator")
SETUP = ROOT / MODULE / "setup_env.sh"
SPEC = importlib.util.spec_from_file_location("abc_sim_setup_check", ROOT / MODULE / "check_environment.py")
CHECK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECK)


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture
def setup_tree(tmp_path: Path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git("init", cwd=upstream)
    for name in ("pyproject.toml", "abc_sim/env.py", "abc_minimal/config.py", "prepare.py"):
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# official fixture\n")
    git("add", ".", cwd=upstream)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-m", "fixture", cwd=upstream)
    target = git("rev-parse", "HEAD", cwd=upstream)

    project = tmp_path / "project"
    module = project / MODULE
    module.mkdir(parents=True)
    script = re.sub(r'SOURCE_REV="[0-9a-f]{40}"', f'SOURCE_REV="{target}"', SETUP.read_text())
    (module / "setup_env.sh").write_text(script)
    shutil.copy(ROOT / "setup_env.sh", project / "setup_env.sh")
    shutil.copytree(
        ROOT / "tools/setup",
        project / "tools/setup",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    shutil.copy(ROOT / MODULE / "requirements.lock.txt", module / "requirements.lock.txt")
    shutil.copy(ROOT / MODULE / "check_environment.py", module / "check_environment.py")

    home = tmp_path / "home"
    bin_dir = home / ".local/bin"
    uv = bin_dir / "uv"
    bin_dir.mkdir(parents=True)
    uv.write_text(
        f"#!{sys.executable}\n"
        + '''
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
    python.write_text(
        "#!/bin/sh\\n"
        "printf '%s\\n' \\"$*\\" >> \\"$PYTHON_LOG\\"\\n"
        "case \\"$*\\" in *mujoco-core*) exit 1;; esac\\n"
        "exit 0\\n"
    )
    python.chmod(0o755)
'''
    )
    uv.chmod(0o755)
    for name in ("gcc", "g++", "make", "curl", "rsync", "tmux", "tar", "ffmpeg"):
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)

    log = tmp_path / "uv.jsonl"
    python_log = tmp_path / "python.log"
    env = dict(
        os.environ,
        HOME=str(home),
        SETUP_LOG=str(log),
        PYTHON_LOG=str(python_log),
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0=f"url.{upstream.as_uri()}.insteadOf",
        GIT_CONFIG_VALUE_0="https://github.com/amazon-far/abc.git",
        PYTHONPATH="/wrong/main/lerobot/src",
        VIRTUAL_ENV="/wrong/main/.venv",
    )
    return project, target, log, env


def run_setup(tree, *args: str):
    project, _, _, env = tree
    return subprocess.run(
        ["bash", str(project / "setup_env.sh"), "--abc-sim-only", *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def calls(tree):
    log = tree[2]
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_fresh_setup_is_pinned_isolated_and_installs_all_assets(setup_tree):
    project, target, _, _ = setup_tree
    main_marker = project / ".venv/keep"
    main_marker.parent.mkdir()
    main_marker.write_text("do not touch")

    result = run_setup(setup_tree)
    assert result.returncode == 0, result.stdout + result.stderr
    source = project / "third_party/abc"
    assert git("rev-parse", "HEAD", cwd=source) == target
    assert (source / "abc_sim/env.py").exists()
    assert main_marker.read_text() == "do not touch"
    install = calls(setup_tree)
    assert install[0]["args"][:2] == ["venv", str(project / ".venv_abc_sim")]
    assert install[1]["args"][0:2] == ["pip", "install"]
    assert install[1]["args"][-2:] == ["-r", str(project / MODULE / "requirements.lock.txt")]
    assert "https://download.pytorch.org/whl/cu128" in install[1]["args"]
    assert "https://download.pytorch.org/whl/cpu" in install[1]["args"]
    assert install[1]["args"][install[1]["args"].index("--index-strategy") + 1] == "unsafe-best-match"
    assert install[2]["args"][-3:] == ["--no-deps", "-e", str(source)]
    assert install[3]["args"][0:2] == ["pip", "check"]
    for call in install[1:]:
        assert call["path"].split(os.pathsep)[0] == str(project / ".venv_abc_sim/bin")
        assert call["pythonpath"] is None and call["virtual_env"] is None
    python_calls = (project.parent / "python.log").read_text().splitlines()
    assert f"{source / 'prepare.py'} --sim --sim-robocasa" in python_calls


def test_check_only_is_read_only_and_reports_missing_source(setup_tree):
    result = run_setup(setup_tree, "--check")
    assert result.returncode != 0 and "source missing" in result.stderr
    assert not calls(setup_tree)
    assert run_setup(setup_tree).returncode == 0
    count = len(calls(setup_tree))
    assert run_setup(setup_tree, "--check").returncode == 0
    assert len(calls(setup_tree)) == count


def test_existing_venv_and_same_revision_edits_are_preserved(setup_tree):
    assert run_setup(setup_tree).returncode == 0
    project = setup_tree[0]
    marker = project / ".venv_abc_sim/keep"
    marker.write_text("keep")
    edited = project / "third_party/abc/pyproject.toml"
    edited.write_text("# user edit\n")

    result = run_setup(setup_tree)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "preserving local" in result.stdout
    assert marker.read_text() == "keep" and edited.read_text() == "# user edit\n"
    assert sum(call["args"][0] == "venv" for call in calls(setup_tree)) == 1


def test_dirty_wrong_revision_is_never_overwritten(setup_tree):
    assert run_setup(setup_tree).returncode == 0
    project = setup_tree[0]
    source = project / "third_party/abc"
    edited = source / "pyproject.toml"
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


def test_exact_lock_matches_official_cuda_stack(tmp_path):
    pins = CHECK.read_pins(ROOT / MODULE / "requirements.lock.txt")
    assert len(pins) == 117
    assert pins["torch"] == "2.11.0+cu128"
    assert pins["mujoco-warp"] == "3.8.0.3"
    assert pins["mjviser"] == "0.0.16"
    assert "mujoco-core" not in pins
    assert pins["warp-lang"] == "1.13.0"
    assert pins["transformers"] == "5.3.0"
    assert pins["draccus"] == "0.10.0"
    assert pins["huggingface-hub"] == "1.8.0"
    invalid = tmp_path / "requirements.txt"
    invalid.write_text("torch>=2.11\n")
    with pytest.raises(ValueError, match="exact package pin"):
        CHECK.read_pins(invalid)


def test_asset_check_requires_manifest_markers_contents_and_robocasa(tmp_path):
    source = tmp_path / "abc"
    manifest_path = source / "abc_sim/models/assets_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        '{"format":"abc_sim_assets_manifest/v1","packages":['
        '{"name":"i2rt_yam","sha256":"abc","extract_dir":"abc_sim/models/assets",'
        '"contains":["i2rt_yam"]}]}'
    )
    assert len(CHECK.asset_problems(source)) == 3

    assets = source / "abc_sim/models/assets"
    (assets / ".abc_sim_asset_packages").mkdir(parents=True)
    (assets / ".abc_sim_asset_packages/i2rt_yam.sha256").write_text("abc\n")
    (assets / "i2rt_yam").mkdir()
    robocasa = source / "abc_sim/models/assets_robocasa"
    for pack in CHECK.ROBOCASA_PACKS:
        directory = robocasa / pack
        directory.mkdir(parents=True)
        (directory / "object.xml").write_text("fixture")
    assert CHECK.asset_problems(source) == []


def test_root_help_documents_abc_sim_entry_points():
    help_text = subprocess.check_output(["bash", str(ROOT / "setup_env.sh"), "--help"], text=True)
    assert "--abc-sim-only [--check]" in help_text
    assert "SBD_SETUP_ABC_SIM=1" in help_text
