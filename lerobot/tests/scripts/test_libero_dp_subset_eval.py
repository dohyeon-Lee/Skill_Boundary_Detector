"""LIBERO partial selection and CPU-only cache/report reuse after the DP move."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "lerobot/skillVLA/libero/configs/train_skills/DP/eval"
ABC_SRC = ROOT / "lerobot/skillVLA/abc/configs/train_skills/DP/eval/src"
sys.path.insert(0, str(ABC_SRC))
spec = importlib.util.spec_from_file_location("libero_subset_config", EVAL / "src/config.py")
CONFIG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIG)


@pytest.fixture
def setup(tmp_path):
    dataset = tmp_path / "dataset/tiny"
    (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps({
        "fps": 20, "features": {"action": {"shape": [7], "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]}},
    }))
    pd.DataFrame({"task": ["first", "second"], "task_index": [0, 1]}).set_index("task").to_parquet(dataset / "meta/tasks.parquet")
    pd.DataFrame([{"episode_index": i, "length": 80, "tasks": ["first" if i < 2 else "second"]}
                  for i in range(4)]).to_parquet(dataset / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame([{"episode_index": ep, "frame_index": t,
                   "observation.state": np.ones(3, dtype=np.float32) * t,
                   "action": np.ones(7, dtype=np.float32) * (1 if t % 2 else -1)}
                  for ep in range(4) for t in range(80)]).to_parquet(dataset / "data/chunk-000/file-000.parquet")
    checkpoint = tmp_path / "model"; checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "type": "diffusion", "state_only": True, "n_obs_steps": 20, "horizon": 24,
        "history_encoder": "transformer", "down_dims": [128, 256, 512],
        "output_features": {"action": {"shape": [7]}},
    }))
    for name in ["model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]:
        (checkpoint / name).write_bytes(b"fixture")
    cfg = {"project_root": str(tmp_path), "dataset_root": "dataset", "target_dataset": "tiny",
           "policy_checkpoint": str(checkpoint), "task_id_space": "dataset", "task_ids": [1, 0],
           "n_episodes": 1, "output_name": "subset", "sbd_prediction_horizon": "all"}
    path = tmp_path / "eval.yaml"; path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def payload(path):
    return json.loads(CONFIG.build_settings(path)["dp_ablation_settings_json"])


def test_libero_selects_without_fsq_and_preserves_its_scoring_profile(setup):
    path, _ = setup
    settings = payload(path)
    manifest = settings["source_manifest"]
    assert [row["episode_index"] for row in manifest["evaluation_episodes"]] == [2, 0]
    assert manifest["pca_episode_ids"] == [0, 2]
    assert settings["threshold_calibration"] == {"source": "selected", "expected_episodes": 2}
    assert manifest["action"]["gripper_indices"] == [6]
    assert Path(settings["artifact_dir"]).parent == EVAL / "partial_build"
    assert Path(settings["report_root"]).name == "trans_small_subset_24_5_5_15_7_4_2_0_previous_action"
    assert settings["probe_variants"][0]["gaussian_center"] == "previous_action"
    assert settings["experiments"][0]["metric"] == "covariance_gated_cosine"
    assert settings["boundary_variants"][0]["terminal_mask_frames"] == 10


def test_libero_center_switch_keeps_pca_and_separates_probes_and_reports(setup):
    path, cfg = setup
    previous = payload(path)
    cfg['gaussian_center'] = 'current_gt'
    path.write_text(yaml.safe_dump(cfg)); current = payload(path)
    assert current['probe_variants'][0]['gaussian_center'] == 'current_gt'
    assert current['probe_variants'][0]['name'].endswith('_current_gt')
    assert previous['probe_variants'][0]['name'].endswith('_previous_action_hold_future')
    assert current['artifact_dir'] == previous['artifact_dir']
    assert current['source_manifest'] == previous['source_manifest']
    assert current['report_root'].endswith('_current_gt')
    assert previous['report_root'].endswith('_previous_action')
    cfg['gaussian_center'] = 'wrong'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='current_gt'):
        payload(path)


def test_libero_full_pca_covers_all_episodes_and_keeps_evaluation_partial(setup):
    path, cfg = setup
    partial = payload(path)
    cfg.update(pca_scope='dataset', gaussian_center='current_gt')
    path.write_text(yaml.safe_dump(cfg)); full = payload(path)
    assert full['source_manifest']['pca_episode_ids'] == [0, 1, 2, 3]
    assert full['source_manifest']['evaluation_episodes'] == partial['source_manifest']['evaluation_episodes']
    assert full['threshold_calibration']['expected_episodes'] == 2
    assert full['artifact_dir'] != partial['artifact_dir']
    assert Path(full['artifact_dir']).name.endswith('_pca_dataset')


def test_libero_reuses_original_full_pca_without_fitting(setup, tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from run_parallel_ablation import load_evaluator
    evaluator = load_evaluator()
    from action_manifold import ActionPCA
    path, cfg = setup
    cfg.update(pca_scope='dataset', gaussian_center='current_gt')
    source_dir = tmp_path / 'dataset/FSQ_dataset/tiny/FSQ_inputs/original/skillset'
    source_dir.mkdir(parents=True)
    (source_dir / 'skillset_manifest.json').write_text(json.dumps({'policy_path': cfg['policy_checkpoint']}))
    stats = {'action': np.arange(7, dtype=np.float32)}
    digest = hashlib.sha256(b'action' + stats['action'].tobytes()).hexdigest()
    metadata = {'dataset': 'tiny', 'action_dim': 7, 'action_mode': 'dataset',
                'descriptor_horizon': 24, 'descriptor_start': 0, 'stride': 3,
                'pca_scale_mode': 'std', 'variance_threshold': .95,
                'probe_action_indices': list(range(7)), 'normalizer_mode': 'MIN_MAX',
                'normalizer_stats_sha256': digest, 'action_delta_indices': list(range(24))}
    original = ActionPCA(np.zeros(7), np.ones(7), np.eye(7)[:6], np.ones(6), np.ones(6)/6, 1000, metadata)
    reference_path = source_dir / 'action_probe_pca.npz'; original.save(reference_path)
    before = reference_path.read_bytes()
    path.write_text(yaml.safe_dump(cfg)); settings = payload(path)
    manifest = settings['source_manifest']
    assert Path(manifest['pca_reference']['path']) == reference_path
    import build_skill_dataset
    monkeypatch.setattr(build_skill_dataset, '_fit_dataset_action_pca',
                        lambda **kw: pytest.fail('full dataset was read instead of reusing original PCA'))
    policy = SimpleNamespace(config=SimpleNamespace(action_feature=SimpleNamespace(shape=[7]),
                            action_delta_indices=list(range(24)), action_execution_start_index=0,
                            action_prediction_horizon=24))
    pca, indices, cached = evaluator._resolve_pca(
        variant=settings['probe_variants'][0], skillset_dir=Path(settings['skillset_dir']),
        output_dir=tmp_path/'private_pca', cache_source_dir=None, dataset_dir=tmp_path/'dataset/tiny',
        policy=policy, normalizer=SimpleNamespace(stats=stats, mode='MIN_MAX'), manifest=manifest,
        variance=.95, stride=3)
    np.testing.assert_array_equal(pca.components, original.components)
    assert pca.sample_count == 1000 and cached.is_file() and indices == tuple(range(7))
    assert reference_path.read_bytes() == before


def test_cpu_controls_share_the_build_and_invalid_selection_fails(setup):
    path, cfg = setup
    before = payload(path)
    cfg.update(gmm_components=2, nms_frames=30, score_power=1)
    path.write_text(yaml.safe_dump(cfg)); after = payload(path)
    assert before["artifact_dir"] == after["artifact_dir"]
    assert before["probe_variants"] == after["probe_variants"]
    assert before["report_root"] != after["report_root"]
    cfg["n_episodes"] = 3; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="requested 3"):
        payload(path)


@pytest.mark.parametrize("scope", ["selected", "dataset"])
def test_horizon_switch_separates_pca_roots_and_keeps_cpu_reuse(setup, scope):
    path, cfg = setup
    cfg['pca_scope'] = scope
    path.write_text(yaml.safe_dump(cfg))
    full = payload(path)
    cfg['sbd_prediction_horizon'] = 1
    path.write_text(yaml.safe_dump(cfg))
    short = payload(path)
    assert short['artifact_dir'] == full['artifact_dir'] + '_sbdh1'
    assert short['probe_variants'][0]['pca_temporal_length'] == 1
    assert short['source_manifest']['evaluation_episodes'] == full['source_manifest']['evaluation_episodes']
    assert Path(short['report_root']).name.startswith(Path(full['artifact_dir']).name + '_1_')
    cfg.update(gmm_components=2, smooth_window=9, score_power=1, gaussian_center='current_gt')
    path.write_text(yaml.safe_dump(cfg))
    recut = payload(path)
    assert recut['artifact_dir'] == short['artifact_dir']
    assert recut['report_root'] != short['report_root']


def test_horizon_change_preserves_original_full_pca_manifest(tmp_path):
    workflow = CONFIG.load_workflow('config')
    base, legacy = tmp_path/'trans_small_eval', tmp_path/'eval'
    old = {'policy_path': 'same-checkpoint', 'pca_reference': {'path': 'full_pca', 'sha256': 'original'}}
    saved = base/'source/skillset_manifest.json'
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps(old))
    before = saved.read_bytes()
    new = {'policy_path': 'same-checkpoint'}
    settings = {'source_manifest': new, 'probe_variants': [{'name': 'future24_h1_current_gt'}]}
    model = {'prediction_horizon': 24, 'sbd_prediction_horizon': 1}
    selected = workflow._select_artifact_dir(base, legacy, model, settings)
    assert selected == tmp_path/'trans_small_eval_sbdh1'
    source = workflow._load_module(EVAL/'src/dataset_source.py', 'horizon_source_test')
    source.prepare_source({**settings, 'skillset_dir': str(selected/'source')})
    assert saved.read_bytes() == before
    assert json.loads((selected/'source/skillset_manifest.json').read_text()) == new
    # Returning to full horizon keeps the original directory and PCA identity.
    assert workflow._select_artifact_dir(base, legacy, {**model, 'sbd_prediction_horizon': 24},
                                         {**settings, 'source_manifest': old}) == base


@pytest.mark.parametrize('unprefixed', [False, True])
def test_compatible_legacy_partial_horizon_cache_is_reused_in_place(tmp_path, unprefixed):
    workflow = CONFIG.load_workflow('config')
    base, legacy = tmp_path/'trans_small_eval', tmp_path/'eval'
    candidate = legacy if unprefixed else base
    manifest = candidate/'source/skillset_manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'policy_path': 'same-checkpoint'}))
    probe = 'future24_h1_current_gt'
    raw = candidate/'raw'/probe/'ep0000001.npz'
    raw.parent.mkdir(parents=True); raw.write_bytes(b'cached-probe')
    settings = {'source_manifest': {'policy_path': 'same-checkpoint'}, 'probe_variants': [{'name': probe}]}
    selected = workflow._select_artifact_dir(base, legacy,
        {'prediction_horizon': 24, 'sbd_prediction_horizon': 1}, settings)
    assert selected == candidate and raw.read_bytes() == b'cached-probe'


@pytest.mark.parametrize("center", ["previous_action", "pure_noise"])
def test_libero_cached_gmm_and_video_html_need_no_policy_or_cuda(setup, tmp_path, monkeypatch, center):
    from run_parallel_ablation import load_evaluator
    from cache_state import cache_status, sha256
    evaluator = load_evaluator()
    path, cfg = setup
    cfg["gaussian_center"] = center
    path.write_text(yaml.safe_dump(cfg))
    settings = payload(path)
    if center == "pure_noise":
        assert settings["probe_variants"][0]["name"].endswith("pure_noise_reverse10_x0step7_final10_v1")
        assert settings["report_root"].endswith("_pure_noise")
        assert settings["probe_variants"][0]["eval_at_step"] == 7
    artifact = tmp_path / "partial_build/test"
    settings.update(artifact_dir=str(artifact), skillset_dir=str(artifact / "source"),
                    report_root=str(tmp_path / "outputs/test"), output_dir=str(tmp_path / "outputs/test/src"))
    source_spec = importlib.util.spec_from_file_location("libero_subset_source", EVAL / "src/dataset_source.py")
    source = importlib.util.module_from_spec(source_spec); source_spec.loader.exec_module(source)
    source.prepare_source(settings)
    assert evaluator._selected_episode_ids(Path(settings["skillset_dir"]), task_ids=[0],
        task_id_space="dataset", target_task="", n_episodes=1) == [0]
    (artifact / "pca").mkdir(parents=True)
    pca = artifact / "pca/std_h24.npz"
    np.savez(pca, metadata=np.array(json.dumps({"source_tasks": [], "action_mode": "dataset",
              "episode_ids": [0, 2], "action_dim": 7, "action_indices": list(range(7)), "scale_mode": "std"})))
    probe = settings["probe_variants"][0]
    for episode in [0, 2]:
        raw = artifact / "raw" / probe["name"] / f"ep{episode:07d}.npz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        ts = np.arange(1, 60, 5)
        meta = {"schema": 2, "pca_sha256": sha256(pca), "descriptor_pca_sha256": sha256(pca),
                "policy_path": settings["source_manifest"]["policy_path"], "seed": settings["seed"],
                "eval_at_step": 7, "replan_interval": 5, "variant": probe}
        extra = {"final_descriptors": np.random.default_rng(episode + 100).normal(size=(len(ts), 96, 2))} if center == "pure_noise" else {}
        np.savez(raw, **extra, descriptors=np.random.default_rng(episode).normal(size=(len(ts), 96, 2)),
                 replan_ts=ts, n_frames=80, metadata_json=np.array(json.dumps(meta)))
    assert cache_status(settings, require_clusters=False)[0]
    import skill_divider
    import torch
    import dp_skillset_eval
    monkeypatch.setattr(skill_divider, "load_policy", lambda *a, **kw: pytest.fail("Policy loaded on cache path"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized on cache path"))
    def stats(**kw):
        kw["output_dir"].mkdir(parents=True, exist_ok=True)
        report = kw["output_dir"] / "index.html"
        report.write_text("<html>statistics</html>"); return report
    monkeypatch.setattr(evaluator, "_render_task_split_reports", stats)
    def video(dataset, metadata, image_key, episode, root, fps, **kw):
        (root / "src/video.mp4").touch()
        return {"video_src": "src/video.mp4", "fps": fps, "episode_start_sec": 0, "episode_end_sec": 4}
    monkeypatch.setattr(dp_skillset_eval, "_episode_video_metadata", video)
    import skillset_boundary_viz as viz
    plotted = []
    original_plot = viz.plot_boundary_curve
    def capture(ax, curve, skills):
        plotted.append((curve, skills))
        return original_plot(ax, curve, skills)
    monkeypatch.setattr(viz, "plot_boundary_curve", capture)
    hashes = {p: sha256(p) for p in artifact.glob("raw/**/*.npz")}
    report = evaluator.render_cached_report(settings)
    assert "LIBERO" in report.read_text() and "episode-player" in report.read_text()
    assert sorted(p.name for p in report.parent.iterdir()) == ["index.html", "src"]
    assert (report.parent / "src/evaluation_complete.json").is_file()
    assert hashes == {p: sha256(p) for p in hashes}
    assert cache_status(settings)[0]
    if center == "pure_noise":
        assert len(plotted) == 4  # Two graphs for each of the two selected episodes.
        for primary, comparison in zip(plotted[::2], plotted[1::2], strict=True):
            assert primary[1] == comparison[1]
            assert "final_comparison" in primary[0]
            assert len(comparison[0]["peak_ts"]) == 0  # Only shared skill cuts, no final-step peaks.
        assert "final 10: comparison only" in report.read_text()
        final_clusters = list(artifact.glob("clusters/*/*_final10/ep*.npz"))
        assert len(final_clusters) == 2
        results = json.loads((report.parent / "src/boundary_results.json").read_text())
        before = [row["boundaries"] for row in results["episodes"]]
        # Replacing only final-10 descriptors must never alter step-7 cuts.
        for raw in artifact.glob("raw/**/*.npz"):
            with np.load(raw) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            arrays["final_descriptors"] *= 100
            np.savez(raw, **arrays)
        evaluator.render_cached_report(settings)
        after = json.loads((report.parent / "src/boundary_results.json").read_text())
        assert [row["boundaries"] for row in after["episodes"]] == before
