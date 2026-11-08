import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/FT/eval"
sys.path.insert(0, str(EVAL / "src"))
from build_checkpoint_analysis import build_analysis, collect, protocol_signature


def test_protocol_keeps_evaluation_contract_not_model_or_chunk_identity():
    base = {"policy_path": "/model/005000", "architecture_label": "one", "tasks": [0],
            "seed": 1000, "external_predictor_model": "/predictor/002000", "skill_end_threshold": .3}
    other = dict(base, policy_path="/model/030000", architecture_label="two", tasks=[3, 4])
    assert protocol_signature(base) == protocol_signature(other)
    assert protocol_signature(base) != protocol_signature(dict(other, skill_end_threshold=.5))
    assert protocol_signature(base) != protocol_signature(dict(other, seed=1))


def test_collect_cross_job_checkpoints_and_separate_protocols(tmp_path):
    records = []; locations = {}
    for step, threshold in [(5000, .3), (10000, .3), (30000, .3), (40000, .5)]:
        original = f"source_{step}"
        label = f"vanila_last_pi05vlm_9_9_{step//1000}k"
        target = f"pi05vlm/vanilla_9_9/expert_frozen_vision_trainable/pred_preskill/{step//1000}k"
        folder = tmp_path / target
        metrics = folder / "metrics"; metrics.mkdir(parents=True)
        data = {"overall": {"pc_success": 50, "n_tasks": 1}, "per_task": [
            {"task_id": 0, "metrics": {"successes": [True, False], "pc_success": 50}}
        ]}
        (metrics / "eval_info_merged.json").write_text(json.dumps({label: data}))
        cache = tmp_path / ".evaluation_runs" / original / "metrics/panel_cache" / f"00_{label}"
        cache.mkdir(parents=True)
        (cache / "info.json").write_text(json.dumps({"signature": {
            "policy_path": f"/model/{step}", "seed": 1000, "skill_end_threshold": threshold}}))
        records.append({"original": original, "label": label, "target": target, "panel": f"00_{label}"})
        locations[original] = f".evaluation_runs/{original}"
    (tmp_path / "model_results.json").write_text(json.dumps({"results": records}))
    (tmp_path / "output_locations.json").write_text(json.dumps({"locations": locations}))
    groups = collect(tmp_path)
    assert sorted(len(points) for points in groups.values()) == [1, 3]
    normal = next(points for points in groups.values() if len(points) == 3)
    assert sorted(p["step"] for p in normal) == [5000, 10000, 30000]
    assert len({p["series"] for p in normal}) == 1
    before = {r["target"]: (tmp_path / r["target"] / "metrics/eval_info_merged.json").read_bytes() for r in records}
    result = build_analysis(tmp_path, charts=False)
    assert result["protocol_groups"] == 2 and result["model_reports"] == 1
    assert (tmp_path / "analysis/index.html").exists()
    for r in records:
        folder = tmp_path / r["target"]
        assert (folder / "metrics/eval_info_merged.json").read_bytes() == before[r["target"]]
        assert (folder.parent / "analysis/index.html").exists()
