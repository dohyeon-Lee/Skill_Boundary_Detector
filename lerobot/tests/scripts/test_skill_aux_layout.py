from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest


CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
SKILLVLA = CONFIGS / "train_skillVLA"
AUX_SRC = SKILLVLA / "PT/Terminator/src"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auxiliary_resolver_defaults_to_the_component_yaml():
    module = _load(AUX_SRC / "terminator_train_config.py", "aux_layout_config")
    assert module.DEFAULT_CONFIG_PATH == AUX_SRC.parent / "terminator_train_config.yaml"
    assert module.DEFAULT_CONFIG_PATH.is_file()
    assert module.load_stage1_component_config(module.DEFAULT_CONFIG_PATH)["stage1_component"] == "Terminator"
    assert not (SKILLVLA / "terminator/submit_train.sh").exists()


@pytest.mark.parametrize("component", ["VSA", "Predictor", "Terminator"])
def test_pt_components_use_the_renamed_checkpoint_group(tmp_path, component):
    module = _load(AUX_SRC / "terminator_train_config.py", "aux_layout_output_config")
    assert module.stage1_run_dirs(tmp_path, "run", component)[0] == (
        tmp_path / "skillVLA_PT" / component / "run"
    )


@pytest.mark.parametrize(
    "relative",
    [
        "PT/Predictor/submit_train.sh",
        "PT/Terminator/submit_train.sh",
        "FT/Predictor/submit_train.sh",
        "FT/Terminator/submit_train.sh",
    ],
)
def test_pt_and_ft_launchers_share_the_relocated_backend(relative):
    launcher = SKILLVLA / relative
    match = re.search(r'^exec "\$\{SCRIPT_DIR\}([^"\n]+)"', launcher.read_text(), re.MULTILINE)
    assert match is not None
    target = (launcher.parent / match.group(1).lstrip("/")).resolve()
    assert target == AUX_SRC / "submit_auxiliary.sh"
    assert os.access(target, os.X_OK)
    subprocess.run(["bash", "-n", str(launcher)], check=True)


@pytest.mark.parametrize("script", ["submit_auxiliary.sh", "train.sbatch"])
def test_auxiliary_backend_shell_paths_and_syntax(script):
    path = AUX_SRC / script
    text = path.read_text()
    match = re.search(r'^BOOTSTRAP_PYTHON="\$\{([^}]+)\}([^"\n]+)"', text, re.MULTILINE)
    assert match is not None
    prefix = {"CONFIG_LIB": CONFIGS, "SRC_DIR": AUX_SRC}[match.group(1)]
    bootstrap = (prefix / match.group(2).lstrip("/")).resolve()
    assert bootstrap == (CONFIGS.parents[3] / ".venv/bin/python").resolve()
    assert "../terminator_train_config.yaml" in text
    assert os.access(path, os.X_OK)
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_ft_vsa_and_joint_import_the_relocated_auxiliary_resolver():
    vsa = _load(
        SKILLVLA / "FT/VSA/src/newtask_ft_vsa_config.py",
        "aux_layout_ft_vsa",
    )
    joint = _load(
        SKILLVLA / "FT/Joint/src/newtask_ft_joint_config.py",
        "aux_layout_ft_joint",
    )
    expected = AUX_SRC / "terminator_train_config.py"
    assert Path(vsa._AUX.__file__) == expected
    assert Path(joint._AUX.__file__) == expected


def test_ft_rename_does_not_leave_the_old_code_directory():
    assert (SKILLVLA / "FT/newtask_ft_common_config.yaml").is_file()
    assert not (SKILLVLA / "NewTask_FT").exists()


@pytest.mark.parametrize("component", ["VSA", "Joint"])
def test_resumed_ft_jobs_override_the_saved_old_output_path(component):
    path = SKILLVLA / "FT" / component / "src/train.sbatch"
    resumed = path.read_text().split('if [ -f "${CONFIG_JSON}" ]; then', 1)[1].split("elif", 1)[0]
    assert '--output_dir="${OUTPUT_DIR}"' in resumed
    subprocess.run(["bash", "-n", str(path)], check=True)
