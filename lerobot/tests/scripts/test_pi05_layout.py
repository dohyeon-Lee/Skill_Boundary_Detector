"""PT/FT launchers keep their own configs and outputs while sharing implementation."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
PI05 = CONFIGS / "train_pi05"


def test_top_level_has_only_pt_and_ft():
    assert {p.name for p in PI05.iterdir() if p.is_dir() and not p.name.startswith(".")} == {"PT", "FT"}


def test_default_train_config_exists_in_pt():
    spec = importlib.util.spec_from_file_location("pi05_layout_train_config", PI05 / "PT/src/train_pi05_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_CONFIG_PATH == PI05 / "PT/pi05_PT_config.yaml"
    assert module.DEFAULT_CONFIG_PATH.is_file()


@pytest.mark.parametrize("script", [
    "PT/submit_train.sh", "FT/submit_train.sh", "PT/src/train.sbatch", "FT/src/train.sbatch",
    "PT/eval/submit_eval.sh", "FT/eval/submit_eval.sh", "PT/eval/src/eval.sbatch",
])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(PI05 / script)], check=True)


def test_eval_bootstrap_and_shared_import_paths():
    eval_dir = PI05 / "PT/eval"
    launcher = (eval_dir / "submit_eval.sh").read_text()
    batch = (eval_dir / "src/eval.sbatch").read_text()
    assert (eval_dir / "../../../../../../..").resolve() == CONFIGS.parents[3]
    assert '${SCRIPT_DIR}/../../../../../../../.venv/bin/python' in launcher
    assert '${SUBMIT_DIR}/../../../../../../../.venv/bin/python' in batch
    assert '${CONFIG_LIB}/src/eval_gpu_packing.py' in launcher
    assert '${PI05_EVAL_SRC_DIR:-${SUBMIT_DIR}/src}' in batch
    assert '${SCRIPT_DIR}/../../PT/eval/submit_eval.sh' in (PI05 / "FT/eval/submit_eval.sh").read_text()
    for name in ("merge_eval_chunks.py", "stitch_panels.py", "convert_legacy_output.py"):
        spec = importlib.util.spec_from_file_location(f"pi05_layout_{name.removesuffix('.py')}", eval_dir / "src" / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if name == "merge_eval_chunks.py":
            assert module._STAGE1_MERGE == CONFIGS / "train_skillVLA/PT/eval/src/merge_eval_chunks.py"
            assert callable(module._stage1_merge().run_merge)


def _checkout(tmp_path):
    repo = tmp_path / "repo"
    configs = repo / "lerobot/examples/libero/configs"
    root = configs / "train_pi05"
    for stage in ("PT", "FT"):
        for relative in ("submit_train.sh", "src/train.sbatch", "eval/submit_eval.sh"):
            dest = root / stage / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PI05 / stage / relative, dest)
        (root / stage / f"pi05_{stage}_config.yaml").write_text(yaml.safe_dump({
            "stage": stage.lower(),
            "pi05_tokenizer": "models/tokenizer",
            "ft_pretrained_run_name": "pt_run",
            "ft_pretrained_checkpoint": "030000",
            "training": {"dataloader": {"batch_size": 2}},
        }))
        (root / stage / "eval/pi05_eval_config.yaml").write_text(yaml.safe_dump({
            "pi05_tokenizer": "models/tokenizer",
            "model_defaults": {"stage": stage, "checkpoint": "030000"},
            "models": [{"model_dir": "run"}],
            "task_ids": [0], "video": {"enable": False}, "output_name": "layout_smoke",
        }))
        checkpoint = repo / f"outputs/pi05_{stage}/run/checkpoints/030000/pretrained_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(json.dumps({"type": "pi05", "chunk_size": 10}))
    shutil.copy2(PI05 / "PT/src/train_pi05_config.py", root / "PT/src/train_pi05_config.py")
    eval_src = root / "PT/eval/src"
    eval_src.mkdir()
    for filename in ("pi05_eval_config.py", "eval.sbatch"):
        shutil.copy2(PI05 / "PT/eval/src" / filename, eval_src / filename)
    shared = configs / "src"
    shared.mkdir()
    for filename in ("global_config_loader.py", "eval_gpu_packing.py"):
        shutil.copy2(CONFIGS / "src" / filename, shared / filename)
    (configs / "global_config.yaml").write_text(yaml.safe_dump({
        "project_root": str(repo), "outputs_root": "outputs", "dataset_root": "dataset",
    }))
    (shared / "snapshot_config.sh").write_text('snapshot_config() { printf "%s\\n" "$1"; }\n')
    # Never submit real jobs; exercise resolution up to the submission boundary.
    (shared / "submit_job.sh").write_text('submit_job() { printf "%s\\n" "$TRAIN_PI05_CONFIG" "$@"; }\n')
    (shared / "node_local_venv.sh").touch()
    tokenizer = repo / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / filename).write_text("{}")
    (repo / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    return root


@pytest.mark.parametrize("stage", ["PT", "FT"])
@pytest.mark.parametrize("action", ["train", "eval"])
def test_launchers_resolve_from_an_unrelated_cwd_without_submitting(tmp_path, monkeypatch, stage, action):
    root = _checkout(tmp_path)
    for key in list(os.environ):
        if key.startswith(("PI05_", "TRAIN_PI05_", "SLURM_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EVAL_NODE_LOCAL_VENV", "0")
    monkeypatch.setenv("PI05_EVAL_DRY_RUN", "1")
    script = root / stage / ("submit_train.sh" if action == "train" else "eval/submit_eval.sh")
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, capture_output=True, text=True, check=True)
    if action == "train":
        assert str(root / stage / f"pi05_{stage}_config.yaml") in result.stdout
        assert str(root / stage / "src/train.sbatch") in result.stdout
        assert f"outputs/pi05_{stage}/" in result.stdout
    else:
        assert str(root / stage / "eval/outputs/layout_smoke") in result.stdout
        assert f"[{stage}:030000" in result.stdout
        assert "Dry run: nothing submitted." in result.stdout


@pytest.mark.parametrize("stage", ["PT", "FT"])
def test_eval_hands_the_stage_workdir_and_shared_sources_to_a_mock_scheduler(tmp_path, monkeypatch, stage):
    root = _checkout(tmp_path)
    for key in list(os.environ):
        if key.startswith(("PI05_", "TRAIN_PI05_", "SLURM_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EVAL_NODE_LOCAL_VENV", "0")
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    sbatch = mock_bin / "sbatch"
    sbatch.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$PI05_EVAL_DIR" "$PI05_EVAL_SRC_DIR" "$PI05_EVAL_CONFIG" "$@"\n'
    )
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{mock_bin}:{os.environ['PATH']}")
    result = subprocess.run(
        ["bash", str(root / stage / "eval/submit_eval.sh")],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert str(root / stage / "eval") in result.stdout.splitlines()
    assert str(root / "PT/eval/src") in result.stdout.splitlines()
    assert str(root / stage / "eval/pi05_eval_config.yaml") in result.stdout.splitlines()
    assert str(root / "PT/eval/src/eval.sbatch") in result.stdout.splitlines()
