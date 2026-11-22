"""Full-skillset calibration stays independent of the smaller report selection."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "skillVLA/libero/configs/train_skills/DP/eval/src/evaluate.py"
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


@pytest.mark.parametrize("history", [20, 60])
def test_initial_history_mask_excludes_spikes_from_smoothing_and_both_means(history, tmp_path, monkeypatch):
    from build_skill_dataset import Args, _prepare_boundary_curve, _save_boundary_curve, _detect_boundaries
    from compute_global_boundary_threshold import main as reduce_threshold

    ts = np.arange(0, 301, 5)
    clean = 1 + np.sin(ts / 25) ** 2
    dirty = clean.copy()
    dirty[ts < history] = 1e8
    dirty[ts >= 260] = 1e9
    cfg = {**boundary(), "initial_mask_frames": history, "terminal_mask_frames": 0,
           "smooth_window": 7, "polyorder": 4, "nms_frames": 15}
    rows = []
    for score in (clean, dirty):
        cuts, threshold, smooth = MODULE._boundaries(score, ts, cfg, terminal_mask_start_frame=260)
        row = {"score": score, "smooth": smooth, "replan_ts": ts,
               "terminal_mask_start_ts": np.array(260.)}
        pool = MODULE._threshold_pool_values(row, cfg)
        assert threshold == pytest.approx(pool.mean())
        assert np.all(cuts >= history)
        rows.append((smooth, threshold, cuts))
    np.testing.assert_array_equal(rows[0][0], rows[1][0])
    np.testing.assert_array_equal(rows[0][2], rows[1][2])
    assert rows[0][1] == rows[1][1]

    args = Args(initial_mask_frames=history, smooth_window=7, savgol_polyorder=4,
                score_power=2, terminal_mask_frames=0, nms_dist=15, min_skill_len=1)
    _, full_smooth, end = _prepare_boundary_curve(ts, dirty, args, 260)
    np.testing.assert_allclose(full_smooth, rows[1][0], rtol=1e-7)
    full_cuts = _detect_boundaries(ts, dirty, 320, args, terminal_mask_start_frame=260)
    assert all(cut >= history for cut in full_cuts[1:-1])
    curves = tmp_path / 'curves'
    _save_boundary_curve(curves, 0, 0, ts, dirty, full_cuts, 320, args,
                         terminal_mask_start_frame=260)
    output = tmp_path / 'threshold.json'
    monkeypatch.setattr('sys.argv', ['reduce', '--curves_dir', str(curves),
                        '--output_path', str(output), '--expected_episodes', '1'])
    reduce_threshold()
    result = json.loads(output.read_text())
    assert result['global_mean'] == pytest.approx(rows[1][1], rel=1e-6)
    assert result['n_replan_points'] == int(((ts >= history) & (ts < 260)).sum())
    assert result['provenance']['initial_mask_frames'] == history


def derived(score, ts, cfg):
    cuts, threshold, smooth = MODULE._boundaries(score, ts, cfg)
    return {"score": score, "replan_ts": ts, "smooth": smooth,
            "threshold": threshold, "boundaries": cuts, "n_frames": np.array(60),
            "delta_bic": np.zeros(len(score)), "selected_k": np.ones(len(score))}


@pytest.mark.parametrize("include_valid_episode", [False, True])
def test_overlapping_masks_keep_episode_without_cuts_or_calibration_values(include_valid_episode):
    cfg = {**boundary(), "initial_mask_frames": 20, "terminal_mask_frames": 24,
           "peak_curvature_scale": 0.5}
    rows = {}
    for ep, last_ts in enumerate([40, 100] if include_valid_episode else [40]):
        ts = np.arange(0, last_ts + 1, 5)
        score = 1 + np.sin(ts / 7) ** 2
        raw = {"replan_ts": ts, "n_frames": np.array(last_ts + 28)}
        scores = {"cosine": score, "selected_k": np.ones(len(ts)),
                  "delta_bic": np.zeros(len(ts))}
        rows[ep] = [("test", MODULE._derive(raw, {}, "cosine", cfg,
                                          min_skill_len=1, scores=scores))]
    masked = rows[0][0][1]
    assert bool(masked["no_valid_boundary_anchors"])
    assert not len(masked["boundaries"])
    assert not len(masked["peak_curvatures"])
    assert not len(MODULE._threshold_pool_values(masked, cfg))
    assert np.isnan(masked["smooth"]).all()
    expected_pool = MODULE._threshold_pool_values(rows[1][0][1], cfg) if include_valid_episode else np.empty(0)
    expected_peaks = len(rows[1][0][1]["peak_curvatures"]) if include_valid_episode else 0
    stats = MODULE._apply_shared_threshold_scopes(
        rows, [{"name": "test", "boundary": "b"}], {"b": cfg}, {}, min_skill_len=1
    )["test"]
    group = stats["groups"]["dataset"]
    assert group["calibration_episodes"] == int(include_valid_episode)
    assert group["excluded_episode_ids"] == [0]
    assert group["points"] == len(expected_pool)
    assert stats["peak_curvature"]["candidate_peaks"] == expected_peaks
    assert not len(rows[0][0][1]["boundaries"])
    if include_valid_episode:
        assert group["threshold"] == pytest.approx(expected_pool.mean())
    else:
        assert np.isnan(group["threshold"])


def test_curvature_uses_original_frames_and_nonuniform_neighbor_spacing():
    ts = np.array([-3., -1., 0., 2., 4.])
    y = 10 - ts**2
    np.testing.assert_allclose(MODULE._peak_curvatures(y, ts, np.array([2])), [2.])
    np.testing.assert_allclose(MODULE._peak_curvatures(y, ts * 10, np.array([2])), [.02])


def test_global_curvature_mean_filter_runs_before_height_and_nms():
    cfg = {**boundary(), "score_power": 1, "terminal_mask_frames": 0,
           "nms_frames": 5, "start_guard_frames": 0, "peak_curvature_scale": 1}
    values = [np.array([0., 0., 8., 10., 8., 0., 0., 7., 0., 0.]),
              np.array([0., 9., 10., 9., 0.])]
    def rows_for(recipe):
        rows = {}
        for ep, score in enumerate(values):
            raw = {"replan_ts": np.arange(len(score)), "n_frames": np.array(20)}
            scores = {"cosine": score, "selected_k": np.ones(len(score)), "delta_bic": np.zeros(len(score))}
            rows[ep] = [('test', MODULE._derive(raw, {}, 'cosine', recipe, min_skill_len=1, scores=scores))]
        return rows
    off = rows_for({k: v for k, v in cfg.items() if k != 'peak_curvature_scale'})
    off_stats = MODULE._apply_shared_threshold_scopes(off, [{'name': 'test', 'boundary': 'b'}], {'b': {**cfg, 'peak_curvature_scale': 0}}, {}, min_skill_len=1)
    np.testing.assert_array_equal(off[0][0][1]['boundaries'], [3])
    assert len(off[0][0][1]['curvature_rejected_ts']) == 0
    rows = rows_for(cfg)
    stats = MODULE._apply_shared_threshold_scopes(rows, [{'name': 'test', 'boundary': 'b'}], {'b': cfg}, {}, min_skill_len=1)
    # One shared mean of curvatures 4, 14, 2. The taller rounded peak at t=3
    # must be removed before it can suppress the sharper t=7 peak under NMS.
    assert stats['test']['peak_curvature']['mean'] == pytest.approx(20 / 3)
    assert stats['test']['peak_curvature']['candidate_peaks'] == 3
    assert stats['test']['peak_curvature']['rejected_peaks'] == 2
    assert stats['test']['groups']['dataset']['threshold'] == off_stats['test']['groups']['dataset']['threshold']
    np.testing.assert_array_equal(rows[0][0][1]['boundaries'], [7])
    np.testing.assert_array_equal(rows[0][0][1]['curvature_rejected_ts'], [3])
    assert not len(rows[1][0][1]['boundaries'])
    np.testing.assert_array_equal(rows[1][0][1]['curvature_rejected_ts'], [2])


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


def test_semantic_terminal_cut_is_masked_and_reapplied_after_global_threshold():
    cfg = {
        **boundary(),
        "score_power": 1,
        "terminal_mask_frames": 0,
        "min_skill_len": 15,
    }
    ts = np.array([0, 16, 32, 48, 64], dtype=np.int64)
    row = {
        "score": np.array([1., 2., 3., 100., 100.]),
        "smooth": np.array([1., 2., 3., 3., 3.]),
        "replan_ts": ts,
        "n_frames": np.array(96),
        "terminal_mask_start_ts": np.array(48.),
        "semantic_terminal_boundary": np.array(48.),
    }
    np.testing.assert_array_equal(MODULE._threshold_pool_values(row, cfg), [1., 2., 3.])
    recut = MODULE._rethreshold_derived(
        row,
        cfg,
        threshold=99.,
        threshold_scope="dataset",
        threshold_group="dataset",
        threshold_pool_points=3,
        min_skill_len=15,
    )
    np.testing.assert_array_equal(recut["boundaries"], [48])

    # A legacy fixed-length terminal display mask must not accidentally become
    # a forced semantic release boundary during the shared-threshold pass.
    fixed_mask_row = dict(row)
    fixed_mask_row.pop("semantic_terminal_boundary")
    recut = MODULE._rethreshold_derived(
        fixed_mask_row,
        cfg,
        threshold=99.,
        threshold_scope="dataset",
        threshold_group="dataset",
        threshold_pool_points=3,
        min_skill_len=15,
    )
    assert len(recut["boundaries"]) == 0


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
    monkeypatch.setattr(MODULE, "_cluster_scores", lambda descriptors, cfg, **kwargs:
                        {"covariance_gated_cosine": np.array([0., 1., 0., 3., 2., 100.]) * descriptors[0, 0, 0]})
    rendered.clear()
    MODULE.run(settings)
    assert loaded == [[0, 1, 2]]
    assert set(rendered[0]["episode_rows"]) == {1}
    assert set(json.loads((out / "summary.json").read_text())) == {"0", "1", "2"}
    stats = json.loads((out / "threshold_statistics.json").read_text())
    assert stats["gripper"]["groups"]["dataset"]["episodes"] == 3
    assert stats["no_gripper"]["groups"]["dataset"]["threshold"] == pytest.approx(stats["gripper"]["groups"]["dataset"]["threshold"] * 100)
