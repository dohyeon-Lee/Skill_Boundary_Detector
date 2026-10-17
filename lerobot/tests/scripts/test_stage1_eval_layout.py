from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
EVAL = CONFIGS / "train_skillVLA/PT/eval"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_eval_config_is_resolved_from_its_new_directory():
    module = _load(EVAL / "src/stage1_eval_config.py", "stage1_eval_layout_config")
    assert module.DEFAULT_CONFIG_PATH == EVAL / "stage1_eval_config.yaml"
    assert module.DEFAULT_CONFIG_PATH.is_file()
    assert not (CONFIGS / "train_skillVLA/stage1_eval/submit_eval.sh").exists()
    assert not (EVAL / "attention_eval_config.yaml").exists()


def test_pi05_reports_load_the_relocated_shared_merge():
    module = _load(
        CONFIGS / "train_pi05/PT/eval/src/merge_eval_chunks.py",
        "pi05_stage1_eval_layout_merge",
    )
    assert module._STAGE1_MERGE == EVAL / "src/merge_eval_chunks.py"
    assert callable(module._stage1_merge().run_merge)


@pytest.mark.parametrize("script", ["submit_eval.sh", "src/eval.sbatch"])
def test_relocated_eval_shell_syntax(script):
    subprocess.run(["bash", "-n", str(EVAL / script)], check=True)


def test_ft_launcher_targets_relocated_engine():
    launcher = CONFIGS / "train_skillVLA/FT/eval/submit_eval.sh"
    assert '${SCRIPT_DIR}/../../PT/eval/submit_eval.sh' in launcher.read_text()
    assert (launcher.parent / "../../PT/eval/submit_eval.sh").resolve() == EVAL / "submit_eval.sh"


def test_relocated_launcher_uses_repository_python_and_shared_gpu_packing():
    launcher = (EVAL / "submit_eval.sh").read_text()
    batch = (EVAL / "src/eval.sbatch").read_text()
    assert '${SCRIPT_DIR}/../../../../../../..' in launcher
    assert '${SUBMIT_DIR}/../../../../../../..' in batch
    assert (EVAL / "../../../../../../..").resolve() == CONFIGS.parents[3]
    assert '${CONFIG_LIB}/src/eval_gpu_packing.py' in launcher
    assert '${SUBMIT_DIR}/../../../src/eval_gpu_packing.py' in batch
    assert (EVAL / "../../../src/eval_gpu_packing.py").resolve() == CONFIGS / "src/eval_gpu_packing.py"
