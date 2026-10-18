"""Keep the dedicated Both3 component experiment's interventions explicit."""

import subprocess
import sys
from pathlib import Path

import yaml


_LIBERO = Path(__file__).resolve().parents[2] / "examples/libero/configs"
_EVAL_DIR = _LIBERO / "train_skillVLA/FT/eval"
sys.path.insert(0, str(_LIBERO / "train_skillVLA/PT/eval/src"))
from stage1_eval_config import _model_entries  # noqa: E402


def _config():
    return yaml.safe_load(
        (_EVAL_DIR / "both3_50k_component_diagnosis.yaml").read_text()
    )


def test_both3_diagnosis_has_four_distinct_component_interventions():
    rows = _model_entries(_config())
    assert len(rows) == 4
    assert len({row["label"] for row in rows}) == 4
    assert [
        (row["skill_source"], row["pose_source"], row["advance_mode"])
        for row in rows
    ] == [
        ("gt", "gt", "gt"),
        ("gt", "gt", "external"),
        ("gt", "predictor", "external"),
        ("external", "predictor", "external"),
    ]
    assert all(row["predictor_diagnostics"] for row in rows)


def test_both3_diagnosis_holds_checkpoints_and_reference_scenes_fixed():
    config = _config()
    models = config["models"]
    assert len({model["vsa_dir"] for model in models}) == 1
    assert all("both_3_" in model["vsa_dir"] for model in models)
    assert all(model["vsa_checkpoint"] == "050000" for model in models)
    assert len({model["predictor_dir"] for model in models}) == 1
    assert all(model["predictor_checkpoint"] == "001000" for model in models)
    assert len({model["terminator_dir"] for model in models[1:]}) == 1
    assert all(model["terminator_checkpoint"] == "010000" for model in models[1:])
    assert config["oracle"] == {
        "episode_exact": True,
        "repeat_episodes": True,
        "broadcast_single_episode": False,
    }
    assert config["terminator"]["gt_termination_min_fraction"] == 0.0
    assert config["n_episodes"] == 10
    assert config["task_ids"] == list(range(10))
    assert config["eval_max_workers_per_gpu"] == 1


def test_both3_diagnosis_has_a_separate_valid_submission_entrypoint():
    script = _EVAL_DIR / "submit_both3_component_diagnosis.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text()
    assert "both3_50k_component_diagnosis.yaml" in source
    assert 'exec bash "${SCRIPT_DIR}/submit_eval.sh"' in source
