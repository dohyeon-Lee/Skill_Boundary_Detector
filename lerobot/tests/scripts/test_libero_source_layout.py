"""No loose Python scripts; FSQ checkpoint modules keep their historical names."""

from __future__ import annotations

import ast
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


LEROBOT = Path(__file__).resolve().parents[2]
LIBERO = LEROBOT / "examples/libero"
CONFIGS = LIBERO / "configs"
FSQ_SRC = CONFIGS / "train_skills/FSQ/src"
EVAL_SRC = CONFIGS / "train_skills/skill_eval/src"
CODEBOOK_SRC = CONFIGS / "train_skills/skill_eval/fsq_codebook/src"
DATA_TOOLS = CONFIGS / "generate_training_dataset/tools"

MOVED = {
    **{name: FSQ_SRC for name in (
        "FSQ.py", "train_FSQ.py", "fsq_frame_cache.py", "fsq_dino_feature_cache.py", "skills_bundle.py",
    )},
    **{name: EVAL_SRC for name in (
        "codebook_visualizer.py", "dp_skillset_eval.py", "skillset_boundary_viz.py",
    )},
    **{name: CODEBOOK_SRC for name in ("decoder_eval.py", "visualize_dino_patches.py")},
    **{name: DATA_TOOLS for name in (
        "convert_lerobot_to_stateaction.py", "convert_libero_data_to_lerobot_dp.py",
        "convert_libero_stateaction_to_lerobot_dp.py", "inspect_dataset.py",
    )},
    **{name: LEROBOT / "tests/examples" for name in ("test_fsq_quantizer.py", "test_spline_codec.py")},
}


@pytest.mark.parametrize("directory", [LIBERO, CONFIGS])
def test_no_root_python_files_or_compatibility_shims(directory):
    assert not list(directory.glob("*.py"))


def test_no_root_shell_compatibility_links():
    assert not list(CONFIGS.glob("*.sh"))


@pytest.mark.parametrize("filename", [
    "gpu_guard.sh", "stall_guard.sh", "runtime_env.sh", "node_local_venv.sh",
    "stage_skillvla_dataset.sh",
])
def test_shared_shell_implementations_stay_under_src(filename):
    path = CONFIGS / "src" / filename
    assert path.is_file() and not path.is_symlink()
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize("filename", MOVED)
def test_moved_sources_exist_and_parse(filename):
    path = MOVED[filename] / filename
    assert path.is_file()
    assert not (LIBERO / filename).exists()
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize(("script", "filename"), [
    ("train_skills/FSQ/src/train_fsq.sbatch", "train_FSQ.py"),
    ("train_skills/FSQ/src/prepare_fsq_frame_cache.sbatch", "fsq_frame_cache.py"),
    ("train_skills/FSQ/src/prepare_fsq_dino_feature_cache.sbatch", "fsq_dino_feature_cache.py"),
])
def test_fsq_batches_use_relocated_sources(script, filename):
    path = CONFIGS / script
    assert str((FSQ_SRC / filename).relative_to(LEROBOT)) in path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_cache_launcher_uses_its_local_src_directory():
    path = CONFIGS / "train_skills/FSQ/submit_train_fsq.sh"
    text = path.read_text()
    for filename in ("fsq_frame_cache.py", "fsq_dino_feature_cache.py"):
        assert f'${{SCRIPT_DIR}}/src/{filename}' in text
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_fsq_module_defaults_and_pickled_configs_survive_relocation(tmp_path):
    sys.path.insert(0, str(FSQ_SRC))
    import FSQ

    assert Path(FSQ.__file__).resolve() == FSQ_SRC / "FSQ.py"
    assert FSQ._REPO_ROOT == LEROBOT.parent
    assert Path(FSQ._DEFAULT_IMAGE_MODEL) == LEROBOT.parent / "models/dinov3-vits16"
    assert FSQ.SplineFSQAEConfig.__module__ == "FSQ"
    cfg = FSQ.SplineFSQAEConfig(fsq_levels=[3, 3, 3])
    checkpoint = tmp_path / "legacy_config.pkl"
    checkpoint.write_bytes(pickle.dumps(cfg))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(LEROBOT / "src"), str(FSQ_SRC))),
        "HF_HUB_OFFLINE": "1",
    }
    result = subprocess.run([
        sys.executable, "-c",
        "import pickle, sys; from pathlib import Path; "
        "cfg = pickle.loads(Path(sys.argv[1]).read_bytes()); "
        "assert type(cfg).__module__ == 'FSQ'; assert cfg.fsq_levels == [3, 3, 3]; "
        "import FSQ; print(FSQ.__file__)",
        str(checkpoint),
    ], cwd=tmp_path, env=env, capture_output=True, text=True, check=True, timeout=60)
    assert str(FSQ_SRC / "FSQ.py") in result.stdout


def test_encoding_and_runtime_terminator_consumers_select_the_canonical_fsq():
    encoder = (CONFIGS / "train_skillVLA/build_data/src/encode_FSQ_skills.py").read_text()
    runtime = (LEROBOT / "src/lerobot/policies/skill_expert/modeling_utils.py").read_text()
    assert '/ "train_skills" / "FSQ" / "src"' in encoder
    assert "examples/libero/configs/train_skills/FSQ/src" in runtime
    assert 'from FSQ import load_fsq_terminator' in runtime
    eval_batch = (CONFIGS / "train_skillVLA/PT/eval/src/eval.sbatch").read_text()
    assert "examples/libero/configs/train_skills/FSQ/src" in eval_batch
