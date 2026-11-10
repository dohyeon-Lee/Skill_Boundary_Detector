"""Boundary input selection and launcher routing, with no GPU/Slurm/HF operations."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import boundary_eval
import prepare_eval as workflow

ROOT = Path(__file__).resolve().parents[3]


def test_boundary_inputs_cover_full_calibration_but_only_selected_video_chunks(tmp_path):
    dataset = tmp_path / "dataset_filtered/demo"
    skillset = tmp_path / "dataset_filtered/FSQ_dataset/demo/FSQ_inputs/seg/skillset"
    checkpoint = tmp_path / "outputs_filtered/DP/run/checkpoints/010000/pretrained_model"
    for p in [dataset / "data", dataset / "meta", skillset / "skills", checkpoint]:
        p.mkdir(parents=True)
    for name in ("skillset_manifest.json", "action_probe_pca.npz"):
        (skillset / name).write_bytes(b"fixture")
    bank = dataset.parent / "skillvla_dataset/demo/eval_init_states.npz"
    bank.parent.mkdir(parents=True)
    bank.touch()
    video = dataset / "videos/camera/chunk-000/file-001.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    settings = {"skillset_dir": str(skillset), "task_ids": [0], "task_id_space": "suite",
                "target_task": "libero_90", "n_episodes": 1, "report_image_key": "camera"}
    resolver = SimpleNamespace(load_config=lambda _: {}, build_settings=lambda _: {"dp_ablation_settings_json": json.dumps(settings)})
    evaluator = SimpleNamespace(_load_manifest=lambda _: {"dataset_dir": str(dataset), "policy_path": str(checkpoint)},
        _portable_path=lambda value, marker: Path(value), _selected_episode_ids=lambda *a, **kw: [1],
        _threshold_calibration_episode_ids=lambda *a: [0, 1, 2])
    video_helpers = SimpleNamespace(_load_episodes_meta=lambda _: None, _resolve_image_key=lambda *a: "camera",
                                   _video_path=lambda directory, meta, ep, key: video if ep == 1 else pytest.fail("Unexpected video"))
    with patch.object(boundary_eval, "load", side_effect=[resolver, evaluator]), patch.dict(sys.modules, {"codebook_visualizer": video_helpers}):
        resolved, _ = boundary_eval.configuration(tmp_path, tmp_path / boundary_eval.RELATIVE / "config.yaml")
    paths = set(map(Path, resolved["input_paths"]))
    assert {dataset / "data", dataset / "meta", skillset / "skills", checkpoint, bank, video} <= paths
    assert not any(p.name == "curves" for p in paths)


def test_small_skill_files_are_grouped_without_expanding_video_uploads():
    base = "dataset_filtered/FSQ_dataset/demo/FSQ_inputs/seg/skillset/skills"
    paths = [base + "/ep1.npz", base + "/ep2.npz", "dataset_filtered/demo/data/file-000.parquet",
             "dataset_filtered/demo/videos/camera/file-001.mp4"]
    assert boundary_eval.upload_groups(paths) == sorted([base, "dataset_filtered/demo/data", paths[-1]])


@pytest.mark.parametrize("server,argument,ready,uses_hf", [
    ("rllab", [], False, False), ("rllab", ["--prepare-runpod"], False, True),
    ("runpod", [], False, True), ("runpod", ["--download-only"], False, True),
    ("runpod", [], True, False),
])
def test_boundary_submit_preserves_scheduler_and_hf_reentry(tmp_path, server, argument, ready, uses_hf):
    work = tmp_path / boundary_eval.RELATIVE
    (work / "src").mkdir(parents=True)
    source = ROOT / boundary_eval.RELATIVE
    shutil.copy2(source / "submit.sh", work / "submit.sh")
    (work / "config.yaml").write_text("{}")
    shared = tmp_path / workflow.CONFIGS / "src"
    shared.mkdir()
    (shared / "global_config_loader.py").write_text(f"print({server!r})")
    (shared / "snapshot_config.sh").write_text('snapshot_config() { printf "%s\\n" "$1"; }\n')
    (shared / "submit_job.sh").write_text('if [ "${1:-}" = batch ]; then echo submitted > "$RECORD"; fi\n_SUBMIT_JOB_SRC="$(dirname "${BASH_SOURCE[0]}")"\n')
    values = {"PROJECT_ROOT": str(tmp_path), "DP_ABLATION_PARTITION": "test", "DP_ABLATION_QOS": "",
        "DP_ABLATION_GRES": "gpu:1", "DP_ABLATION_CPUS": "2", "DP_ABLATION_MEM": "64G",
        "DP_ABLATION_TIME": "3:00:00", "DP_ABLATION_NODELIST": "", "DP_ABLATION_EXCLUDE_NODES": "",
        "DP_ABLATION_SELECTION": "final", "DP_ABLATION_SKILLSET_DIR": "/skillset",
        "DP_ABLATION_OUTPUT_DIR": str(work / "outputs/run")}
    exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in values.items())
    (work / "src/config.py").write_text(f"print({exports!r})")
    tools = tmp_path / "tools/runpod"
    tools.mkdir(parents=True)
    (tools / "prepare_eval.py").write_text("import os,json,sys\nfrom pathlib import Path\nPath(os.environ['RECORD']).write_text(json.dumps(sys.argv[1:]))")
    (tmp_path / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    record = tmp_path / "record"
    env = {k: v for k, v in os.environ.items() if k not in {"DP_BOUNDARY_ABLATION_CONFIG", "SBD_EVAL_HF_READY"}}
    env.update(RECORD=str(record), SBD_EVAL_HF_READY="1" if ready else "0")
    p = subprocess.run(["bash", str(work / "submit.sh"), *argument], env=env, text=True, capture_output=True, timeout=15)
    assert p.returncode == 0, p.stdout + p.stderr
    if uses_hf:
        assert json.loads(record.read_text()) == ["--config", str(work / "config.yaml"), *argument]
        assert not (work / ".config_snapshots").exists()
    else:
        assert record.read_text().strip() == "submitted"
