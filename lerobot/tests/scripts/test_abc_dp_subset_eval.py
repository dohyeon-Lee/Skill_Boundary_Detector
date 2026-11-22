"""Standalone SBD must stay within the selected episodes and private outputs."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "lerobot/skillVLA/abc/configs/train_skills/DP/eval"
sys.path.insert(0, str(EVAL / "src"))
from dataset_source import prepare_source  # noqa: E402
from run_parallel_ablation import load_evaluator  # noqa: E402

spec = importlib.util.spec_from_file_location("abc_dp_subset_config", EVAL / "src/config.py")
CONFIG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIG)


@pytest.fixture
def setup(tmp_path):
    dataset = tmp_path / "dataset_ABC/abc_sim_all"
    (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
    (dataset / "native_dataset.json").write_text(json.dumps({
        "format": "sbd_abc_sim_native/v1", "action_dim": 14, "state_dim": 14,
    }))
    names = [f"joint_{i}" if i not in (6, 13) else f"gripper_{i}" for i in range(14)]
    (dataset / "meta/info.json").write_text(json.dumps({"features": {"action": {"names": names}}}))
    rows = [{"episode_index": i, "task_indices": [task], "source_task": source,
             "length": 96, "states_actions_path": f"{i}.bin"}
            for i, task, source in [(0, 59, "grab_clutter"), (1, 59, "grab_clutter"),
                                    (2, 60, "grab_clutter"), (3, 60, "grab_clutter"),
                                    (4, 59, "other_task")]]
    pd.DataFrame(rows).to_parquet(dataset / "meta/episodes/chunk-000/file-000.parquet")
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "type": "diffusion", "state_only": True, "use_delta_joint_actions": True,
        "n_obs_steps": 60, "horizon": 1, "action_sequence_mode": "future_only",
        "history_encoder": "transformer", "down_dims": [128, 256, 512],
        "output_features": {"action": {"shape": [14]}},
    }))
    for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
                 "policy_preprocessor_step_4_normalizer_processor.safetensors"):
        (checkpoint / name).write_bytes(b"checkpoint fixture")
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": {
        "root": str(dataset), "repo_id": "lerobot/abc_sim_all", "tasks": ["grab_clutter"],
        "episodes": None, "episode_selection_manifest": None,
    }}))
    cfg = yaml.safe_load((EVAL / "eval_config.yaml").read_text())
    cfg.pop("dp_run_name", None)
    cfg.pop("dp_checkpoint", None)
    cfg.update(project_root=str(tmp_path), policy_checkpoint="model", task_ids=[60, 59], n_episodes=1,
               pca_scope="selected", score_power=1, sbd_prediction_horizon=1,
               replan_interval=8, gmm_components=5, nms_frames=45, smooth_window=7, polyorder=4, peak_curvature_scale=0)
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_config_needs_no_fsq_and_selects_first_episode_per_prompt(setup):
    path, cfg = setup
    settings = CONFIG.build_settings(str(path))
    payload = json.loads(settings["dp_ablation_settings_json"])
    assert [r["episode_index"] for r in payload["source_manifest"]["evaluation_episodes"]] == [2, 0]
    assert payload["source_manifest"]["pca_episode_ids"] == [0, 2]
    assert payload["threshold_calibration"] == {"source": "selected", "expected_episodes": 2}
    assert Path(payload["report_root"]).parent == EVAL / "outputs"
    assert Path(payload["output_dir"]) == Path(payload["report_root"]) / "src"
    assert Path(payload["artifact_dir"]).parent == EVAL / "partial_build"
    assert not (Path(cfg["project_root"]) / "dataset_ABC/FSQ_dataset").exists()
    assert payload["source_manifest"]["action"]["gripper_indices"] == [6, 13]


@pytest.mark.parametrize("absolute,step", [(False, 100000), (True, 10000), (False, "010000")])
def test_short_model_selection_uses_global_outputs_root(setup, absolute, step):
    import shutil
    path, cfg = setup
    root = Path(cfg["project_root"])
    outputs = root / "custom_outputs"
    model = outputs / "DP/my_dp_run/checkpoints" / str(step).zfill(6) / "pretrained_model"
    shutil.copytree(root / "model", model)
    (root / "global_config.yaml").write_text(yaml.safe_dump({
        "server": "rllab", "outputs_root": str(outputs) if absolute else "custom_outputs",
    }))
    cfg.pop("policy_checkpoint")
    cfg.update(dp_run_name="my_dp_run", dp_checkpoint=step)
    path.write_text(yaml.safe_dump(cfg))
    resolved = CONFIG.build_settings(str(path))
    payload = json.loads(resolved["dp_ablation_settings_json"])
    assert Path(payload["source_manifest"]["policy_path"]) == model
    assert Path(payload["policy_checkpoint_override"]) == model


def test_short_model_selection_rejects_conflicting_old_path(setup):
    path, cfg = setup
    cfg.update(dp_run_name="other_model", dp_checkpoint=100000, outputs_root="outputs_custom")
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="paths disagree"):
        CONFIG.build_settings(str(path))


@pytest.mark.parametrize("encoder,mode,tag", [
    ("transformer", "iid", "trans"), ("gru", "iid", "GRUiid"), ("gru", "epi", "GRUepi"),
])
@pytest.mark.parametrize("size,dims", [("small", [128, 256, 512]), ("base", [512, 1024, 2048])])
def test_output_prefix_comes_from_checkpoint_architecture(setup, encoder, mode, tag, size, dims):
    path, cfg = setup
    checkpoint = Path(cfg["project_root"]) / "model/config.json"
    policy = json.loads(checkpoint.read_text())
    policy.update(history_encoder=encoder, history_gru_training=mode, down_dims=dims)
    checkpoint.write_text(json.dumps(policy))
    settings = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    assert settings["dp_model_prefix"] == f"{tag}_{size}"
    assert Path(settings["report_root"]).name.startswith(f"{tag}_{size}_{cfg['output_name']}_")
    assert Path(settings["artifact_dir"]).name == f"{tag}_{size}_{cfg['output_name']}_pca_selected"


def test_missing_task_or_episode_count_fails_before_creating_outputs(setup):
    path, cfg = setup
    cfg["task_ids"] = [999]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Task 999 has 0 episodes"):
        CONFIG.build_settings(str(path))


def test_prepared_source_is_not_a_skillset_and_rejects_changed_selection(setup, tmp_path):
    path, _cfg = setup
    payload = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    payload["skillset_dir"] = str(tmp_path / "partial_build/experiment/source")
    prepare_source(payload)
    evaluator = load_evaluator()
    selected = evaluator._selected_episode_ids(Path(payload["skillset_dir"]), task_ids=[60],
                  task_id_space="dataset", target_task="grab_clutter", n_episodes=1)
    assert selected == [2]
    assert not (Path(payload["skillset_dir"]) / "skills").exists()
    prepare_source(payload)  # Unchanged input is resumable.
    payload["source_manifest"]["evaluation_episodes"].pop()
    with pytest.raises(ValueError, match="choose a new output_name"):
        prepare_source(payload)


def test_checkpoint_normalizer_change_invalidates_source_identity(setup):
    path, cfg = setup
    first = CONFIG.build_settings(str(path))
    processor = Path(cfg["project_root"]) / "model/policy_preprocessor_step_4_normalizer_processor.safetensors"
    processor.write_bytes(b"changed normalization")
    second = CONFIG.build_settings(str(path))
    a = json.loads(first["dp_ablation_settings_json"])["source_manifest"]
    b = json.loads(second["dp_ablation_settings_json"])["source_manifest"]
    assert a["checkpoint_sha256"] != b["checkpoint_sha256"]


@pytest.mark.parametrize("representation", ["mean", "trajectory"])
def test_pca_fits_only_selected_episodes(representation, monkeypatch, tmp_path):
    load_evaluator()
    import build_skill_dataset as builder

    calls = []
    def episodes(dataset, tasks=(), episode_ids=None):
        calls.append(episode_ids)
        assert episode_ids == (3, 8)
        for episode in episode_ids:
            values = np.random.default_rng(episode).normal(size=(96, 2)).astype(np.float32)
            yield episode, values, values
    monkeypatch.setattr(builder, "_iter_state_action_episodes", episodes)
    normalizer = type("IdentityNormalizer", (), {"normalize": lambda self, values: values})()
    fit = builder._fit_dataset_action_pca if representation == "mean" else builder._fit_dataset_trajectory_pca
    pca = fit(dataset_dir=tmp_path, action_delta_indices=(0, 1), descriptor_start=0,
              descriptor_horizon=2, stride=1, action_dim=2, action_indices=(0, 1),
              action_mode="dataset", rel_mask=None, normalizer=normalizer,
              variance_threshold=0.95, scale_mode="std", metadata={}, episode_ids=(3, 8))
    assert calls == [(3, 8)]
    assert pca.n_components > 0


def test_subset_preserves_worker_budget(setup):
    path, cfg = setup
    cfg['eval_workers'] = 4
    path.write_text(yaml.safe_dump(cfg))
    settings = CONFIG.build_settings(str(path))
    payload = json.loads(settings['dp_ablation_settings_json'])
    assert settings['dp_ablation_workers'] == payload['workers'] == 4
    assert payload['cpu_budget'] == 8
    cfg['eval_workers'] = 9
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='no greater than eval_cpus_per_task'):
        CONFIG.build_settings(str(path))


def test_compact_controls_expand_and_connect_recipes(setup):
    path, cfg = setup
    cfg.update(gmm_components=2, replan_interval=4, nms_frames=30, smooth_window=9, polyorder=3)
    path.write_text(yaml.safe_dump(cfg))
    resolved = CONFIG.build_settings(str(path))
    payload = json.loads(resolved["dp_ablation_settings_json"])
    probe, cluster, boundary = (payload[key][0] for key in ("probe_variants", "cluster_variants", "boundary_variants"))
    assert (probe["descriptor_temporal_length"], probe["replan_interval"]) == (1, 4)
    assert cluster["k"] == 2
    assert boundary["initial_mask_frames"] == 60
    assert (boundary["nms_frames"], boundary["smooth_window"], boundary["polyorder"]) == (30, 9, 3)
    assert payload["experiments"][0]["probe"] == probe["name"]
    assert payload["experiments"][0]["cluster"] == cluster["name"]
    assert payload["experiments"][0]["boundary"] == boundary["name"]
    assert (resolved["dp_ablation_n_obs_steps"], resolved["dp_ablation_prediction_horizon"]) == (60, 1)
    assert Path(payload["report_root"]).name == "trans_small_" + cfg["output_name"] + "_pca_selected_1_4_2_30_9_3_1_0_previous_action"
    assert Path(payload["artifact_dir"]).name == "trans_small_" + cfg["output_name"] + "_pca_selected"
    assert Path(payload["work_dir"]).name == "1_4_2_30_9_3_1_0_previous_action"


@pytest.mark.parametrize("length", [1, 72, "all"])
@pytest.mark.parametrize("mode", ["future_only", "history_reconstruction"])
def test_sbd_length_comes_from_checkpoint_without_changing_dp(setup, length, mode):
    path, cfg = setup
    checkpoint = Path(cfg["project_root"]) / "model/config.json"
    policy = json.loads(checkpoint.read_text())
    policy.update(horizon=72 if mode == "future_only" else 131,
                  future_action_horizon=72, action_sequence_mode=mode)
    checkpoint.write_text(json.dumps(policy))
    before = checkpoint.read_bytes()
    cfg["sbd_prediction_horizon"] = length
    path.write_text(yaml.safe_dump(cfg))
    resolved = CONFIG.build_settings(str(path))
    payload = json.loads(resolved["dp_ablation_settings_json"])
    probe = payload["probe_variants"][0]
    expected = 72 if length == "all" else length
    assert resolved["dp_ablation_prediction_horizon"] == 72
    assert resolved["dp_ablation_sbd_prediction_horizon"] == expected
    assert Path(payload["report_root"]).name.endswith(f"_{expected}_{cfg['replan_interval']}_{cfg['gmm_components']}_45_7_4_1_0_previous_action")
    assert (probe["descriptor_temporal_length"], probe["descriptor_pca_temporal_length"]) == (expected, expected)
    assert probe["gaussian_center"] == "previous_action"
    assert probe["name"].endswith("_previous_action_hold_future")
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("key,value", [
    ("sbd_prediction_horizon", 2), ("sbd_prediction_horizon", 0),
    ("gmm_components", 0), ("replan_interval", 0), ("nms_frames", -1),
    ("smooth_window", 0), ("smooth_window", 8), ("polyorder", 7),
    ("score_power", 0), ("score_power", -1), ("score_power", float("nan")),
    ("score_power", float("inf")), ("score_power", True), ("score_power", "bad"),
    ("peak_curvature_scale", -1), ("peak_curvature_scale", float("nan")),
    ("peak_curvature_scale", float("inf")), ("peak_curvature_scale", True),
])
def test_invalid_compact_controls_fail_before_build(setup, key, value):
    path, cfg = setup
    cfg[key] = value
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=key):
        CONFIG.build_settings(str(path))


@pytest.mark.parametrize("location", ["top", "advanced", "recipe"])
@pytest.mark.parametrize("center", ["current_gt", "pure_noise"])
def test_abc_centers_are_selectable_and_keep_shared_pca(setup, location, center):
    from defaults import CONTROLS, resolve_config
    path, cfg = setup
    previous = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    if location == "recipe":
        cfg, _ = resolve_config(cfg)
        for key in CONTROLS:
            cfg.pop(key, None)
        cfg["probe_variants"][0]["gaussian_center"] = center
        cfg["probe_variants"][0]["name"] = "legacy_" + center
        cfg["experiments"][0]["probe"] = "legacy_" + center
    elif location == "advanced":
        cfg.pop("gaussian_center", None)
        cfg["advanced"] = {"gaussian_center": center}
    else:
        cfg["gaussian_center"] = center
    path.write_text(yaml.safe_dump(cfg))
    settings = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    assert settings["probe_variants"][0]["gaussian_center"] == center
    assert settings["probe_variants"][0]["name"] != previous["probe_variants"][0]["name"]
    assert settings["artifact_dir"] == previous["artifact_dir"]
    assert settings["source_manifest"] == previous["source_manifest"]
    if location != "recipe":
        assert settings["report_root"].endswith("_" + center)
        assert settings["work_dir"] != previous["work_dir"]


def test_invalid_abc_center_is_rejected(setup):
    path, cfg = setup
    cfg["gaussian_center"] = "invalid"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="gaussian_center"):
        CONFIG.build_settings(str(path))


def test_abc_pca_scopes_separate_fit_data_and_preserve_evaluation_selection(setup):
    path, cfg = setup
    choices = {}
    for scope in ("selected", "source_tasks", "dataset"):
        cfg["pca_scope"] = scope
        path.write_text(yaml.safe_dump(cfg))
        choices[scope] = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
        assert "_pca_" + scope + "_" in choices[scope]["report_root"]
    selected, tasks, dataset = [choices[key] for key in ("selected", "source_tasks", "dataset")]
    assert selected["source_manifest"]["pca_episode_ids"] == [0, 2]
    assert "pca_episode_ids" not in tasks["source_manifest"]
    assert dataset["source_manifest"]["pca_episode_ids"] == [0, 1, 2, 3]
    assert dataset["source_manifest"]["pca_source_tasks"] == []
    assert dataset["source_manifest"]["source_tasks"] == ["grab_clutter"]
    assert len({item["artifact_dir"] for item in choices.values()}) == 3
    for item in choices.values():
        assert item["source_manifest"]["evaluation_episodes"] == selected["source_manifest"]["evaluation_episodes"]
        assert item["threshold_calibration"]["expected_episodes"] == 2


def test_abc_dataset_pca_fit_does_not_filter_to_evaluation_source_tasks(setup, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from cache_state import cache_status, sha256
    evaluator = load_evaluator()
    import build_skill_dataset as builder
    from action_manifold import ActionPCA, NumpyActionNormalizer
    path, cfg = setup
    cfg.update(pca_scope="dataset", gaussian_center="pure_noise")
    train_config = Path(cfg["project_root"]) / "model/train_config.json"
    train_config.write_text(json.dumps({"dataset": {"root": str(Path(cfg["project_root"]) / "dataset_ABC/abc_sim_all"),
        "tasks": ["grab_clutter", "other_task"], "episodes": [1, 4]}}))
    path.write_text(yaml.safe_dump(cfg))
    settings = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    artifact = tmp_path / "partial_build/test"
    settings.update(artifact_dir=str(artifact), skillset_dir=str(artifact / "source"))
    manifest = settings["source_manifest"]
    calls = []
    def fit(**kwargs):
        calls.append(kwargs)
        assert kwargs["source_tasks"] == ()
        assert kwargs["episode_ids"] == (1, 4)  # Only DP training episodes, including other_task.
        return ActionPCA(mean=np.zeros(14), scale=np.ones(14), components=np.eye(14),
            explained_variance=np.ones(14), explained_variance_ratio=np.full(14, 1 / 14),
            sample_count=50, metadata=kwargs["metadata"])
    monkeypatch.setattr(builder, "_fit_dataset_action_pca", fit)
    policy = SimpleNamespace(config=SimpleNamespace(action_feature=SimpleNamespace(shape=(14,)),
        action_delta_indices=[0], action_execution_start_index=0, action_prediction_horizon=1))
    variant = settings["probe_variants"][0]
    for center in ("pure_noise", "current_gt", "previous_action"):
        pca, _, pca_path = evaluator._resolve_pca(variant={**variant, "gaussian_center": center},
            skillset_dir=Path(settings["skillset_dir"]), output_dir=artifact, cache_source_dir=None,
            dataset_dir=Path(manifest["dataset_dir"]), policy=policy,
            normalizer=NumpyActionNormalizer("IDENTITY", {}), manifest=manifest, variance=.95, stride=3)
        assert pca.metadata["source_tasks"] == []
    assert len(calls) == 1  # Same action/PCA recipe shares one basis across centers.
    for episode in (0, 2):
        raw = artifact / "raw" / variant["name"] / f"ep{episode:07d}.npz"
        metadata = {"pca_sha256": sha256(pca_path), "descriptor_pca_sha256": sha256(pca_path),
            "variant": variant, "policy_path": manifest["policy_path"], "seed": settings["seed"],
            "eval_at_step": 7, "replan_interval": 8}
        evaluator._cache_save(raw, {"descriptors": np.zeros((2, 96, 14)),
            "final_descriptors": np.ones((2, 96, 14)), "n_frames": 96, "replan_ts": [60, 68]}, metadata)
    assert cache_status(settings, require_clusters=False)[0]


def test_abc_reuses_compatible_cache_before_scope_was_added_to_names(setup, tmp_path):
    path, cfg = setup
    from defaults import resolve_config
    resolved, model = resolve_config(cfg)
    settings = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    base = tmp_path / resolved["output_name"]
    legacy = base.with_name(base.name.removeprefix(model["model_prefix"] + "_"))
    old = base.with_name(base.name.removesuffix("_pca_selected"))
    (old / "source").mkdir(parents=True)
    (old / "source/skillset_manifest.json").write_text(json.dumps(settings["source_manifest"]))
    assert CONFIG._select_artifact_dir(base, legacy, model, settings) == old


def test_dataset_pca_uses_saved_random_episode_selection_and_fails_if_it_is_missing(setup):
    path, cfg = setup
    root = Path(cfg["project_root"])
    selection = root / "dataset_ABC/abc_sim_all/meta/selected.json"
    selection.write_text(json.dumps({"format": "sbd_abc_sim_episode_selection/v1",
        "tasks": ["grab_clutter"], "episodes": [1, 3], "episode_count": 2}))
    train_config = root / "model/train_config.json"
    train = json.loads(train_config.read_text())
    train["dataset"]["episode_selection_manifest"] = "meta/selected.json"
    train_config.write_text(json.dumps(train))
    cfg["pca_scope"] = "dataset"
    path.write_text(yaml.safe_dump(cfg))
    settings = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    assert settings["source_manifest"]["pca_episode_ids"] == [1, 3]
    assert "episode_selection_sha256" in settings["source_manifest"]["pca_training_source"]
    assert [row["episode_index"] for row in settings["source_manifest"]["evaluation_episodes"]] == [2, 0]
    selection.unlink()
    with pytest.raises(FileNotFoundError):
        CONFIG.build_settings(str(path))
    train_config.unlink()
    with pytest.raises(ValueError, match="train_config.json"):
        CONFIG.build_settings(str(path))


@pytest.mark.parametrize("history,future", [(0, 1), (0, 36), (59, 72)])
def test_previous_action_center_holds_last_action_without_future_gt(history, future):
    # Load only the numeric helper; no policy, torch or GPU is needed.
    import ast
    source = ROOT / "lerobot/skillVLA/libero/configs/train_skills/build_data/src/skill_divider.py"
    tree = ast.parse(source.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == "_previous_action_probe_center")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    shape = (history + future, 3)
    current = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    previous = -current - 1
    before = current.copy()
    center = namespace[node.name](current, previous, history, future)
    np.testing.assert_array_equal(center[:history], current[:history])
    np.testing.assert_array_equal(center[history:], np.repeat(previous[history:history + 1], future, axis=0))
    np.testing.assert_array_equal(current, before)
    current[history:] += 1000
    previous[history + 1:] -= 1000
    # Changing any current/future action must not change the probe future.
    changed = namespace[node.name](current, previous, history, future)
    np.testing.assert_array_equal(changed[history:], center[history:])


def test_old_explicit_recipe_remains_supported(setup):
    from defaults import CONTROLS, resolve_config
    path, cfg = setup
    resolved_cfg, _ = resolve_config(cfg)
    for key in CONTROLS:
        resolved_cfg.pop(key, None)
    path.write_text(yaml.safe_dump(resolved_cfg))
    legacy = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    assert legacy["probe_variants"] == resolved_cfg["probe_variants"]
    assert legacy["cluster_variants"] == resolved_cfg["cluster_variants"]
    assert legacy["boundary_variants"] == resolved_cfg["boundary_variants"]


@pytest.fixture
def cached_subset(setup, tmp_path):
    from cache_state import sha256
    path, _cfg = setup
    settings = json.loads(CONFIG.build_settings(str(path))["dp_ablation_settings_json"])
    artifact = tmp_path / "partial_build/test"
    settings.update(artifact_dir=str(artifact), skillset_dir=str(artifact / "source"),
                    report_root=str(tmp_path / "outputs/test"), output_dir=str(tmp_path / "outputs/test/src"),
                    report_video=False)
    prepare_source(settings)
    metadata = {"action_dim": 14, "action_indices": list(range(14)), "scale_mode": "std",
                "source_tasks": ["grab_clutter"], "action_mode": "delta_joint", "episode_ids": [0, 2]}
    (artifact / "pca").mkdir()
    pca = artifact / "pca/std_h1.npz"
    np.savez(pca, metadata=np.array(json.dumps(metadata)))
    pca_hash = sha256(pca)
    evaluator = load_evaluator()
    probe = settings["probe_variants"][0]
    cluster = settings["cluster_variants"][0]
    for row in settings["source_manifest"]["evaluation_episodes"]:
        name = f"ep{row['episode_index']:07d}.npz"
        raw = artifact / "raw" / probe["name"] / name
        raw.parent.mkdir(parents=True, exist_ok=True)
        meta = {"schema": 2, "pca_sha256": pca_hash, "descriptor_pca_sha256": pca_hash,
                "policy_path": settings["source_manifest"]["policy_path"], "seed": settings["seed"],
                "eval_at_step": 7, "replan_interval": 8, "variant": probe}
        descriptors = np.random.default_rng(row["episode_index"]).normal(
            size=(3, probe["count"], 2)).astype(np.float32)
        descriptors[:, :probe["count"] // 2] += 3
        np.savez(raw, descriptors=descriptors, replan_ts=np.array([64, 72, 80]),
                 n_frames=row["length"], metadata_json=np.array(json.dumps(meta)))
        dest = artifact / "clusters" / probe["name"] / cluster["name"] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        scores = evaluator._empty_cluster_scores(3)
        scores["cosine"] = np.array([0., 1., 0.], dtype=np.float32)
        np.savez(dest, **scores, metadata_json=np.array(json.dumps({
            "schema": 2, "cluster": cluster, "raw_sha256": sha256(raw)})))
    return settings


def test_complete_build_is_ready_without_any_html_output(cached_subset):
    from cache_state import cache_status
    assert not Path(cached_subset["output_dir"]).exists()
    assert cache_status(cached_subset)[0]
    cached_subset["boundary_variants"][0]["nms_frames"] = 23
    assert cache_status(cached_subset)[0]


def test_another_model_does_not_reuse_legacy_cache(cached_subset, setup, monkeypatch, tmp_path):
    path, cfg = setup
    shared = CONFIG._shared_module()
    monkeypatch.setattr(CONFIG, "_shared_module", lambda: shared)
    monkeypatch.setattr(CONFIG, "__file__", str(tmp_path / "src/config.py"))
    cfg["output_name"] = "test"
    path.write_text(yaml.safe_dump(cfg))
    checkpoint = Path(cfg["project_root"]) / "model/config.json"
    policy = json.loads(checkpoint.read_text())
    policy.update(history_encoder="gru", history_gru_training="epi")
    checkpoint.write_text(json.dumps(policy))
    resolved = CONFIG.build_settings(str(path))
    settings = json.loads(resolved["dp_ablation_settings_json"])
    assert Path(settings["artifact_dir"]).name == "GRUepi_small_test_pca_selected"
    assert settings["artifact_dir"] != cached_subset["artifact_dir"]
    assert not resolved["dp_ablation_cache_ready"]


def test_cpu_only_suffix_changes_share_the_existing_dp_cache(cached_subset, setup, monkeypatch, tmp_path):
    path, cfg = setup
    shared = CONFIG._shared_module()
    monkeypatch.setattr(CONFIG, "_shared_module", lambda: shared)
    monkeypatch.setattr(CONFIG, "__file__", str(tmp_path / "src/config.py"))
    cfg["output_name"] = "test"
    path.write_text(yaml.safe_dump(cfg))
    first = CONFIG.build_settings(str(path))
    before = json.loads(first["dp_ablation_settings_json"])
    assert first["dp_ablation_cache_ready"]
    cfg.update(gmm_components=3, nms_frames=30, smooth_window=9, polyorder=3)
    path.write_text(yaml.safe_dump(cfg))
    second = CONFIG.build_settings(str(path))
    after = json.loads(second["dp_ablation_settings_json"])
    assert second["dp_ablation_cache_ready"]
    assert before["report_root"] != after["report_root"]
    assert Path(after["report_root"]).name == "trans_small_test_pca_selected_1_8_3_30_9_3_1_0_previous_action"
    assert before["artifact_dir"] == after["artifact_dir"] == cached_subset["artifact_dir"]


@pytest.mark.parametrize('power', [2, 2.0, 0.5])
def test_power_only_changes_reuse_dp_and_gmm_caches(cached_subset, setup, monkeypatch, tmp_path, power):
    from cache_state import cache_status
    path, cfg = setup
    shared = CONFIG._shared_module()
    monkeypatch.setattr(CONFIG, '_shared_module', lambda: shared)
    monkeypatch.setattr(CONFIG, '__file__', str(tmp_path / 'src/config.py'))
    cfg.update(output_name='test', score_power=1)
    path.write_text(yaml.safe_dump(cfg))
    first = json.loads(CONFIG.build_settings(str(path))['dp_ablation_settings_json'])
    cfg['score_power'] = power
    path.write_text(yaml.safe_dump(cfg))
    resolved = CONFIG.build_settings(str(path))
    second = json.loads(resolved['dp_ablation_settings_json'])
    assert resolved['dp_ablation_cache_ready'] and cache_status(second)[0]
    assert first['artifact_dir'] == second['artifact_dir'] == cached_subset['artifact_dir']
    assert first['probe_variants'] == second['probe_variants']
    assert first['cluster_variants'] == second['cluster_variants']
    assert first['report_root'] != second['report_root']
    assert second['boundary_variants'][0]['score_power'] == power
    assert Path(second['report_root']).name.endswith(('_2_0' if power == 2 else '_0.5_0') + '_previous_action')


def test_curvature_scale_changes_only_cpu_report_and_suffix(cached_subset, setup, monkeypatch, tmp_path):
    from cache_state import cache_status
    path, cfg = setup
    shared = CONFIG._shared_module()
    monkeypatch.setattr(CONFIG, '_shared_module', lambda: shared)
    monkeypatch.setattr(CONFIG, '__file__', str(tmp_path / 'src/config.py'))
    cfg.update(output_name='test', peak_curvature_scale=.2)
    path.write_text(yaml.safe_dump(cfg))
    settings = json.loads(CONFIG.build_settings(str(path))['dp_ablation_settings_json'])
    assert settings['boundary_variants'][0]['peak_curvature_scale'] == .2
    assert Path(settings['report_root']).name.endswith('_1_8_5_45_7_4_1_0.2_previous_action')
    assert settings['artifact_dir'] == cached_subset['artifact_dir']
    assert settings['probe_variants'] == cached_subset['probe_variants']
    assert settings['cluster_variants'] == cached_subset['cluster_variants']
    assert cache_status(settings)[0]


@pytest.mark.parametrize("change", ["missing_cluster", "probe_count", "gmm_recipe", "checkpoint"])
def test_cache_reuse_rejects_incomplete_or_changed_build(cached_subset, change):
    from cache_state import cache_status
    if change == "missing_cluster":
        next((Path(cached_subset["artifact_dir"]) / "clusters").rglob("*.npz")).unlink()
    elif change == "probe_count":
        cached_subset["probe_variants"][0]["count"] += 1
    elif change == "gmm_recipe":
        cached_subset["cluster_variants"][0]["max_iter"] += 1
    else:
        cached_subset["source_manifest"]["checkpoint_sha256"]["model.safetensors"] = "changed"
    assert not cache_status(cached_subset)[0]
    assert cache_status(cached_subset, require_clusters=False)[0] == (change in {"missing_cluster", "gmm_recipe"})


def test_deleted_outputs_are_rebuilt_without_loading_policy(cached_subset, monkeypatch):
    import shutil
    import skill_divider
    from cache_state import sha256
    evaluator = load_evaluator()
    monkeypatch.setattr(skill_divider, "load_policy", lambda *a, **kw: pytest.fail("Policy loaded during cached report"))
    monkeypatch.setattr(evaluator, "_saved_abc_terminal_boundaries", lambda *a: {})
    monkeypatch.setattr(evaluator, "_report_references", lambda *a: ({}, {0: np.array([]), 2: np.array([])}))
    def render(**kw):
        path = kw["output_dir"] / "index.html"
        assert set(kw["episode_rows"]) == {0, 2}
        path.write_text("<html>cached report</html>")
        return path
    monkeypatch.setattr(evaluator, "_render_task_split_reports", render)
    artifact = Path(cached_subset["artifact_dir"])
    before = {str(p): sha256(p) for p in artifact.rglob("*.npz")}
    assert evaluator.render_cached_report(cached_subset).is_file()
    assert (artifact / "summary.json").is_file()
    shutil.rmtree(cached_subset["output_dir"])
    assert evaluator.render_cached_report(cached_subset).is_file()
    assert before == {str(p): sha256(p) for p in artifact.rglob("*.npz")}


def test_default_runner_uses_cpu_report_when_cache_is_ready(cached_subset, monkeypatch):
    import run_eval
    from types import SimpleNamespace
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = []
    resolved = {"dp_ablation_settings_json": json.dumps(cached_subset), "dp_ablation_episode_count": 2}
    monkeypatch.setattr(run_eval, "build_settings", lambda *a: resolved)
    monkeypatch.setattr(run_eval, "load_evaluator", lambda: SimpleNamespace(render_cached_report=lambda s: calls.append(s)))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", "eval_config.yaml"])
    run_eval.main()
    assert len(calls) == 1


@pytest.mark.parametrize("report_only", [False, True])
@pytest.mark.parametrize("change", ["same_name", "new_name", "missing", "corrupt"])
def test_runner_refits_gmm_on_cpu_without_loading_policy_or_rewriting_dp(
        cached_subset, monkeypatch, report_only, change):
    import run_eval
    import skill_divider
    import torch
    from cache_state import cache_status, sha256

    settings = cached_subset
    artifact = Path(settings["artifact_dir"])
    probe = settings["probe_variants"][0]["name"]
    cluster = settings["cluster_variants"][0]
    old_name = cluster["name"]
    if change in {"same_name", "new_name"}:
        cluster["k"] = 2
    if change == "new_name":
        cluster["name"] = "k2_unanchored"
        settings["experiments"][0]["cluster"] = cluster["name"]
    if change in {"missing", "corrupt"}:
        for path in (artifact / "clusters").rglob("*.npz"):
            path.unlink() if change == "missing" else path.write_bytes(b"bad cache")
    protected = [*artifact.glob("raw/**/*.npz"), *artifact.glob("pca/*.npz")]
    before = {path: (sha256(path), path.stat().st_mtime_ns) for path in protected}
    assert not cache_status(settings)[0]
    assert cache_status(settings, require_clusters=False)[0]

    evaluator = load_evaluator()
    monkeypatch.setattr(skill_divider, "load_policy", lambda *a, **kw: pytest.fail("Policy loaded on CPU path"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized on CPU path"))
    monkeypatch.setattr(evaluator, "_saved_abc_terminal_boundaries", lambda *a: {})
    monkeypatch.setattr(evaluator, "_report_references", lambda *a: ({}, {0: np.array([]), 2: np.array([])}))
    def render(**kw):
        path = kw["output_dir"] / "index.html"
        path.write_text("<html>CPU GMM report</html>")
        return path
    monkeypatch.setattr(evaluator, "_render_task_split_reports", render)
    fits = []
    original = evaluator._cluster_scores
    def fit(*args, **kwargs):
        fits.append(args[1]["k"])
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluator, "_cluster_scores", fit)
    resolved = {"dp_ablation_settings_json": json.dumps(settings), "dp_ablation_episode_count": 2}
    monkeypatch.setattr(run_eval, "build_settings", lambda *a: resolved)
    monkeypatch.setattr(run_eval, "load_evaluator", lambda: evaluator)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    argv = ["run_eval.py", "--config", "eval_config.yaml"]
    monkeypatch.setattr(sys, "argv", argv + (["--report-only"] if report_only else []))
    run_eval.main()
    assert fits == [cluster["k"], cluster["k"]]
    assert cache_status(settings)[0]
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    for path in (artifact / "clusters" / probe / cluster["name"]).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            assert json.loads(str(data["metadata_json"]))["cluster"] == cluster
            assert np.all(data["selected_k"] == cluster["k"])
    if change == "new_name":
        assert len(list((artifact / "clusters" / probe / old_name).glob("*.npz"))) == 2
    run_eval.main()
    assert len(fits) == 2  # Repeating the report reuses the new GMM cache.
    assert before == {path: (sha256(path), path.stat().st_mtime_ns) for path in protected}
