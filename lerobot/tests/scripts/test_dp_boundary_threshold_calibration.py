"""Full-skillset calibration stays independent of the smaller report selection."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/src/evaluate.py"
SPEC = importlib.util.spec_from_file_location("boundary_calibration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def skillset(tmp_path, count=3):
    root = tmp_path / "skillset"
    (root / "skills").mkdir(parents=True)
    for episode in range(count):
        (root / "skills" / f"ep{episode:07d}_task000_skill000.npz").touch()
    return root


def boundary():
    return {"threshold": "mean", "threshold_scope": "dataset", "method": "peak",
            "score_power": 2, "smooth_window": 1, "polyorder": 0,
            "terminal_mask_frames": 5, "nms_frames": 0, "start_guard_frames": 0}


def derived(score, ts, cfg):
    cuts, threshold, smooth = MODULE._boundaries(score, ts, cfg)
    return {"score": score, "replan_ts": ts, "smooth": smooth,
            "threshold": threshold, "boundaries": cuts, "n_frames": np.array(60),
            "delta_bic": np.zeros(len(score)), "selected_k": np.ones(len(score))}


def test_calibration_uses_complete_skillset_and_rejects_partial_copy(tmp_path):
    root = skillset(tmp_path)
    settings = {"threshold_calibration": {"source": "skillset", "expected_episodes": 3}}
    assert MODULE._threshold_calibration_episode_ids(settings, root, [1]) == [0, 1, 2]
    next((root / "skills").glob("ep0000000*" )).unlink()
    with pytest.raises(ValueError, match="requires 3 episodes, found 2"):
        MODULE._threshold_calibration_episode_ids(settings, root, [1])


def test_legacy_selected_scope_does_not_expand_work(tmp_path):
    root = skillset(tmp_path)
    assert MODULE._threshold_calibration_episode_ids({}, root, [2, 1]) == [2, 1]


def test_each_condition_pools_its_own_valid_points_with_length_weighting():
    cfg = boundary()
    a = np.array([0., 1., 0., 3., 2., 100.])
    b = np.array([0., 2., 0., 4., 0., 0., 3., 100.])
    rows = {ep: [(name, derived(values * scale, np.arange(len(values)) * 5, cfg))
                 for name, scale in [("gripper", 1), ("no_gripper", 10)]]
            for ep, values in enumerate([a, b])}
    experiments = [{"name": name, "boundary": "global"} for name in ("gripper", "no_gripper")]
    stats = MODULE._apply_shared_threshold_scopes(rows, experiments, {"global": cfg}, {}, min_skill_len=1)
    expected = (np.square(a[:-1]).sum() + np.square(b[:-1]).sum()) / 12
    assert stats["gripper"]["groups"]["dataset"]["threshold"] == pytest.approx(expected)
    assert stats["no_gripper"]["groups"]["dataset"]["threshold"] == pytest.approx(expected * 100)
    assert stats["gripper"]["groups"]["dataset"]["points"] == 12
    for pairs in rows.values():
        np.testing.assert_array_equal(pairs[0][1]["boundaries"], pairs[1][1]["boundaries"])


def test_report_only_recalibrates_all_episodes_but_renders_selected_subset(tmp_path, monkeypatch):
    root = skillset(tmp_path)
    dataset = tmp_path / "dataset_filtered/libero_90_full_full"
    dataset.mkdir(parents=True)
    (root / "skillset_manifest.json").write_text(json.dumps({
        "dataset_dir": str(dataset), "detector": {"min_skill_len": 1, "eval_at_step": 7, "replan_interval": 5}, "action": {}}))
    out = tmp_path / "output"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({str(i): {} for i in range(3)}))
    cfg, ts = boundary(), np.arange(6) * 5
    for ep in range(3):
        for name in ("gripper", "no_gripper"):
            raw = out / "raw" / name / f"ep{ep:07d}.npz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            np.savez(raw, replan_ts=ts, descriptors=np.ones((6, 4, 2)), n_frames=60)
            cluster = out / "clusters" / name / "k5" / raw.name
            cluster.parent.mkdir(parents=True, exist_ok=True)
            scale = (ep + 1) * (10 if name == "no_gripper" else 1)
            np.savez(cluster, covariance_gated_cosine=np.array([0., 1., 0., 3., 2., 100.]) * scale)
    settings = {"skillset_dir": str(root), "output_dir": str(out), "task_ids": [0],
        "task_id_space": "suite", "target_task": "libero_90", "n_episodes": 1,
        "threshold_calibration": {"source": "skillset", "expected_episodes": 3},
        "experiments": [{"name": n, "probe": n, "cluster": "k5", "boundary": "global", "metric": "covariance_gated_cosine"}
                        for n in ("gripper", "no_gripper")],
        "cluster_variants": [{"name": "k5"}], "boundary_variants": [{"name": "global", **cfg}],
        "split_report_by_task": True}
    monkeypatch.setattr(MODULE, "_selected_episode_ids", lambda *a, **kw: [1])
    monkeypatch.setattr(MODULE, "_selected_task_episode_groups", lambda *a: {0: [1]})
    monkeypatch.setattr(MODULE, "_derive", lambda raw, cluster, metric, boundary, **kw:
                        derived(kw["scores"][metric], raw["replan_ts"], boundary))
    import dp_skillset_eval
    monkeypatch.setattr(dp_skillset_eval, "load_episode_skills", lambda *a: ([], None, ()))
    rendered = []
    monkeypatch.setattr(MODULE, "_render_task_split_reports", lambda **kw:
                        (rendered.append(kw) or out / "index.html"))
    MODULE.render_cached_report(settings)
    assert set(rendered[0]["episode_rows"]) == {1}
    stats = json.loads((out / "threshold_statistics.json").read_text())
    a, b = stats["gripper"], stats["no_gripper"]
    assert a["groups"]["dataset"]["episodes"] == 3
    assert b["groups"]["dataset"]["threshold"] == pytest.approx(a["groups"]["dataset"]["threshold"] * 100)
    assert a["calibration"]["evaluation_episodes"] == 1
    assert a["calibration"]["episodes"] == 3

    # Exercise the inference entry point as well: only policy computation is
    # stubbed; it must load/calibrate all episodes before restricting reports.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest_path = root / "skillset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_path"] = str(checkpoint)
    manifest_path.write_text(json.dumps(manifest))
    settings.update(seed=42, resume=True, pca_variance=.95, pca_stride=3,
        probe_variants=[{"name": name, "probe_generation": "scheduler_gaussian", "gaussian_sampling": "iid", "count": 4}
                        for name in ("gripper", "no_gripper")])
    import pandas as pd
    import skill_divider
    from action_manifold import NumpyActionNormalizer
    policy = SimpleNamespace(config=SimpleNamespace(state_only=True, action_feature=SimpleNamespace(shape=(7,)), horizon=2))
    monkeypatch.setattr(skill_divider, "load_policy", lambda *a: (policy, None))
    loaded = []
    monkeypatch.setattr(skill_divider, "load_data", lambda directory, episode_ids:
                        (loaded.append(episode_ids) or pd.DataFrame({"episode_index": episode_ids})))
    monkeypatch.setattr(NumpyActionNormalizer, "from_preprocessor", lambda pre: SimpleNamespace(mode="test"))
    pca = tmp_path / "pca.npz"
    pca.write_bytes(b"test PCA provenance")
    monkeypatch.setattr(MODULE, "_resolve_pca", lambda **kw: (None, tuple(range(7)), pca))
    monkeypatch.setattr(MODULE, "_descriptor_whitening_scale", lambda *a: np.ones(2))
    def probe(**kw):
        scale = (int(kw["ep_df"].iloc[0].episode_index) + 1) * (10 if kw["variant"]["name"] == "no_gripper" else 1)
        return {"descriptors": np.ones((6, 4, 2)) * scale, "replan_ts": ts, "n_frames": np.array(60)}
    monkeypatch.setattr(MODULE, "_probe_episode", probe)
    monkeypatch.setattr(MODULE, "_cluster_scores", lambda descriptors, cfg:
                        {"covariance_gated_cosine": np.array([0., 1., 0., 3., 2., 100.]) * descriptors[0, 0, 0]})
    rendered.clear()
    MODULE.run(settings)
    assert loaded == [[0, 1, 2]]
    assert set(rendered[0]["episode_rows"]) == {1}
    assert set(json.loads((out / "summary.json").read_text())) == {"0", "1", "2"}
    stats = json.loads((out / "threshold_statistics.json").read_text())
    assert stats["gripper"]["groups"]["dataset"]["episodes"] == 3
    assert stats["no_gripper"]["groups"]["dataset"]["threshold"] == pytest.approx(stats["gripper"]["groups"]["dataset"]["threshold"] * 100)
