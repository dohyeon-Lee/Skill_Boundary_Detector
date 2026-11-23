"""PT thresholds and target means must not be silently interchanged."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "coffee_data_modes", ROOT / "lerobot/skillVLA/mimicgen/configs/src/data_modes.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


@pytest.fixture
def reference(tmp_path):
    dataset = tmp_path / "old/skillvla/libero"
    pt = dataset / "FSQ_pt"
    pt.mkdir(parents=True)
    (pt / "skill_latents.npz").touch()
    segmentation = {"dataset_name": "libero", "policy": {"run": "dp", "checkpoint": "010000"}}
    (pt / "segmentation_contract.json").write_text(json.dumps({"contract": {"segmentation": segmentation}}))
    threshold = dataset / "_work/matching/skillset/global_boundary_threshold.json"
    threshold.parent.mkdir(parents=True)
    threshold.write_text(json.dumps({"boundary_threshold_mode": "global_mean", "global_mean": .0033,
                                     "segmentation_contract": segmentation}))
    cfg = {"project_root": str(tmp_path), "dataset_root": "coffee",
           "skillvla_pt_reference": {"dataset_root": "old", "source": "libero", "run": "FSQ_pt"},
           "skillset_seg_name": "seg_{skillvla_threshold_tag}_{skillvla_data_mode}"}
    return cfg, threshold


@pytest.mark.parametrize("mode,threshold_mode,tag", [
    ("pt", "global_mean", "globalmean"), ("ft", "global_mean", "globalref"),
    ("ft_own", "global_mean", "globalmean"), ("ft_epi", "episode_mean", "episodemean"),
])
def test_modes_resolve_thresholds_and_distinct_names(reference, mode, threshold_mode, tag):
    cfg, threshold = reference
    resolved = MOD.configure_data_mode({**cfg, "skillvla_data_mode": mode})
    assert resolved["skillset_boundary_threshold_mode"] == threshold_mode
    assert resolved["skillset_global_threshold_source"] == (str(threshold) if mode == "ft" else "")
    assert resolved["skillset_seg_name"] == f"seg_{tag}_{mode}"


def test_ft_rejects_a_skillset_built_with_the_target_mean(reference, tmp_path):
    cfg, threshold = reference
    cfg = MOD.configure_data_mode({**cfg, "skillvla_data_mode": "ft"})
    manifest = {"detector": {"boundary_threshold_mode": "global_mean",
                             "global_threshold_path": str(tmp_path / "coffee/global_boundary_threshold.json")}}
    with pytest.raises(ValueError, match="FT SBD must reuse"):
        MOD.validate_skillset_mode(cfg, manifest, tmp_path / "coffee")
    manifest["detector"]["global_threshold_path"] = str(threshold)
    MOD.validate_skillset_mode(cfg, manifest, tmp_path / "coffee")


def test_ambiguous_pt_provenance_is_rejected(reference):
    cfg, threshold = reference
    duplicate = threshold.parents[2] / "duplicate/skillset/global_boundary_threshold.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(threshold.read_text())
    with pytest.raises(ValueError, match="Expected one matching PT threshold"):
        MOD.configure_data_mode({**cfg, "skillvla_data_mode": "ft"})
