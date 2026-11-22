#!/usr/bin/env python3
"""ABC wrapper around the maintained shared SBD evaluator configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_SOURCE_DIR = Path(__file__).resolve().parent


def _shared_module():
    path = next(
        parent
        / "libero/configs/train_skills/DP/eval/src/boundary_config.py"
        for parent in Path(__file__).resolve().parents
        if (
            parent
            / "libero/configs/train_skills/DP/eval/src/boundary_config.py"
        ).is_file()
    )
    spec = importlib.util.spec_from_file_location("sbd_shared_boundary_eval_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared SBD evaluator config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_artifact_dir(base, legacy, model, payload):
    """Separate horizon-dependent PCA/probes, preserving compatible old caches.

    Full-horizon evaluations retain their existing base path. Shorter SBD
    horizons get their own source manifest and PCA directory, since importing
    the original full-build PCA may be valid for only one of those horizons.
    CPU post-processing controls deliberately do not participate in this key.
    """
    horizon = int(model["sbd_prediction_horizon"])
    shortened = horizon != int(model["prediction_horizon"])
    preferred = base.with_name(f"{base.name}_sbdh{horizon}") if shortened else base
    if preferred.exists():
        return preferred
    scope_tag = model.get("pca_scope_tag")
    old_scope_paths = []
    if scope_tag:
        old_scope_paths = [path.with_name(path.name.removesuffix("_" + scope_tag))
                           for path in (base, legacy)]
    for candidate in dict.fromkeys(([base, legacy] if shortened else [legacy]) + old_scope_paths):
        manifest = candidate / "source/skillset_manifest.json"
        try:
            if json.loads(manifest.read_text()) != payload["source_manifest"]:
                continue
        except (OSError, ValueError):
            continue
        if shortened:
            # A matching dataset/checkpoint alone does not identify a horizon.
            # Reuse a legacy shared root only if it has this horizon's cache.
            has_probe = any(
                any((candidate / "raw" / probe["name"]).glob("ep*.npz"))
                for probe in payload["probe_variants"]
            )
            if not has_probe and not any((candidate / "pca").glob(f"*_h{horizon}.npz")):
                continue
        return candidate
    return preferred


def build_settings(config_path: str, *, analysis_domain="abc", eval_dir_override=None) -> dict:
    shared = _shared_module()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    defaults = _load_module(_SOURCE_DIR / "defaults.py", "sbd_compact_defaults")
    cfg, model = defaults.resolve_config(shared.load_config(config_path), domain=analysis_domain)
    standalone = shared.as_bool(shared.get_value(cfg, "standalone", False))
    if not standalone:
        raise ValueError("DP/eval requires standalone: true")
    eval_dir = Path(eval_dir_override) if eval_dir_override else Path(__file__).resolve().parents[1]
    settings = shared.build_settings(
        config_path, analysis_domain=analysis_domain, dataset_source=standalone,
        output_root_override=eval_dir / "outputs",
        config_override=cfg,
    )
    workers = int(shared.get_value(cfg, "eval_workers", 1))
    if workers < 1 or workers > settings["dp_ablation_cpus"]:
        raise ValueError("eval_workers must be positive and no greater than eval_cpus_per_task")
    payload = json.loads(settings["dp_ablation_settings_json"])
    payload["policy_checkpoint_override"] = str(cfg.get("policy_checkpoint", payload["policy_checkpoint_override"]))
    payload.update(workers=workers, cpu_budget=settings["dp_ablation_cpus"], analysis_domain=analysis_domain)
    if standalone:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        source_path = _SOURCE_DIR / "dataset_source.py" if analysis_domain == "abc" else eval_dir / "src/dataset_source.py"
        source = _load_module(source_path, f"sbd_{analysis_domain}_source")

        device = str(cfg.get("eval_device", "cuda"))
        if device not in {"cuda", "cpu"}:
            raise ValueError("eval_device must be cuda|cpu")
        relative_output = Path(payload["output_dir"]).relative_to(eval_dir / "outputs")
        artifact_dir = eval_dir / "partial_build" / relative_output
        payload["artifact_dir"] = str(artifact_dir)
        payload["skillset_dir"] = str(artifact_dir / "source")
        settings["dp_ablation_skillset_dir"] = payload["skillset_dir"]
        settings["dp_ablation_artifact_dir"] = str(artifact_dir)
        payload["device"] = device
        payload["source_manifest"] = source.make_source(cfg, payload)
        # Preserve compatible older caches in place, including unprefixed
        # ones. Horizon-dependent PCA provenance must not collide with them.
        legacy_relative = relative_output.with_name(relative_output.name.removeprefix(model["model_prefix"] + "_"))
        legacy_artifact = eval_dir / "partial_build" / legacy_relative
        artifact_dir = _select_artifact_dir(artifact_dir, legacy_artifact, model, payload)
        payload["artifact_dir"] = str(artifact_dir)
        payload["skillset_dir"] = str(artifact_dir / "source")
        settings["dp_ablation_artifact_dir"] = str(artifact_dir)
        settings["dp_ablation_skillset_dir"] = payload["skillset_dir"]
        payload["dp_model_prefix"] = model["model_prefix"]
        for probe in payload["probe_variants"]:
            if probe.get("mode", "manifest") == "manifest":
                probe["mode"] = "std_without_gripper"
            if probe.get("descriptor_mode") == "manifest":
                probe["descriptor_mode"] = "std_without_gripper"
        settings["dp_ablation_device"] = device
        settings["dp_ablation_episode_count"] = len(payload["source_manifest"]["evaluation_episodes"])
        report_root = Path(payload["output_dir"])
        if model.get("output_suffix"):
            # Share expensive caches under the YAML's base name while keeping
            # each public hyperparameter combination's report independent.
            report_root = report_root.with_name(f"{report_root.name}_{model['output_suffix']}")
            payload["work_dir"] = str(artifact_dir / "workers" / model["output_suffix"])
        payload["report_root"] = str(report_root)
        settings["dp_ablation_output_dir"] = str(report_root)
        payload["output_dir"] = str(Path(payload["report_root"]) / "src")
        payload["report_video"] = True
    settings["dp_ablation_workers"] = workers
    settings["dp_ablation_n_obs_steps"] = model["n_obs_steps"]
    settings["dp_ablation_prediction_horizon"] = model["prediction_horizon"]
    settings["dp_ablation_sbd_prediction_horizon"] = model["sbd_prediction_horizon"]
    from cache_state import cache_status
    ready, reason = cache_status(payload, require_clusters=False)
    settings["dp_ablation_cache_ready"] = ready
    settings["dp_ablation_cache_status"] = reason
    settings["dp_ablation_settings_json"] = json.dumps(payload, separators=(",", ":"))
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    shared = _shared_module()
    settings = build_settings(args.config)
    if args.shell:
        shared.print_shell(settings)
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
