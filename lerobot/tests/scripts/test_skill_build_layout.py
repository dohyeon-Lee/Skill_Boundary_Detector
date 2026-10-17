"""Canonical locations and launchers for skill dataset builders."""

from pathlib import Path
import subprocess

import pytest


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
LIBERO = LEROBOT_ROOT / "examples/libero"
CONFIGS = LIBERO / "configs"
SBD_SRC = CONFIGS / "train_skills/build_data/src"
SKILLVLA_SRC = CONFIGS / "train_skillVLA/build_data/src"


@pytest.mark.parametrize(
    ("directory", "filename"),
    [
        (SBD_SRC, "build_skill_dataset.py"),
        (SBD_SRC, "skill_divider.py"),
        (SBD_SRC, "action_manifold.py"),
        (SBD_SRC, "sbd_multimodality.py"),
        (SBD_SRC, "SBD_visualize.py"),
        (SKILLVLA_SRC, "encode_FSQ_skills.py"),
        (SKILLVLA_SRC, "add_skill_latents_to_dataset.py"),
    ],
)
def test_build_modules_live_under_their_own_configs(directory: Path, filename: str) -> None:
    assert (directory / filename).is_file()
    assert not (LIBERO / filename).exists()


@pytest.mark.parametrize(
    ("launcher", "target"),
    [
        (SBD_SRC / "build_skillset.sbatch", SBD_SRC / "build_skill_dataset.py"),
        (SKILLVLA_SRC / "build_skillset.sbatch", SBD_SRC / "build_skill_dataset.py"),
        (SKILLVLA_SRC / "verify_skillset.sbatch", SBD_SRC / "build_skill_dataset.py"),
        (SKILLVLA_SRC / "encode_skills.sbatch", SKILLVLA_SRC / "encode_FSQ_skills.py"),
        (
            SKILLVLA_SRC / "build_skillvla.sbatch",
            SKILLVLA_SRC / "add_skill_latents_to_dataset.py",
        ),
        (
            CONFIGS / "train_skills/skill_eval/fsq_codebook/src/encode_latents.sbatch",
            SKILLVLA_SRC / "encode_FSQ_skills.py",
        ),
    ],
)
def test_submission_scripts_use_canonical_builders(launcher: Path, target: Path) -> None:
    source = launcher.read_text()
    assert target.is_file()
    assert str(target.relative_to(LEROBOT_ROOT)) in source
    assert f"examples/libero/{target.name}" not in source
    subprocess.run(["bash", "-n", str(launcher)], check=True, capture_output=True)
