"""Check whether the requested report can be built from private CPU caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_matches(saved, requested, pca, descriptor_pca):
    """Legacy 'manifest' PCA and explicit 'std' can describe the same basis."""
    saved = dict(saved)
    for key, basis in (("mode", pca), ("descriptor_mode", descriptor_pca)):
        if saved.get(key) == "manifest" and requested.get(key) == "std":
            if basis.get("scale_mode") != "std" or basis.get("action_indices") != list(range(basis["action_dim"])):
                return False
            saved[key] = "std"
    return saved == requested


def cache_status(settings, *, require_clusters=True):
    """Validate DP/PCA provenance; optionally require already fitted GMM scores.

    Missing or changed GMM scores can be rebuilt from valid descriptors on CPU.
    """
    if not settings.get("resume", True):
        return False, "resume is disabled"
    root = Path(settings["artifact_dir"])
    source = settings["source_manifest"]
    manifest = Path(settings["skillset_dir"]) / "skillset_manifest.json"
    try:
        if manifest.is_file() and json.loads(manifest.read_text()) != source:
            return False, "checkpoint or episode selection changed; choose a new output_name"
        pcas = {}
        for path in (root / "pca").glob("*.npz"):
            with np.load(path, allow_pickle=False) as payload:
                pcas[sha256(path)] = json.loads(str(payload["metadata"]))
        if not pcas:
            return False, "PCA cache is missing"
        probes = {item["name"]: item for item in settings["probe_variants"]}
        clusters = {item["name"]: item for item in settings["cluster_variants"]}
        pairs = set()
        for experiment in settings["experiments"]:
            if str(experiment["metric"]).startswith(("denoising_", "whitened_")) or experiment["metric"] == "delta_bic_gain":
                if not (root / "summary.json").is_file():
                    return False, "this metric requires the saved build summary"
            for probe in experiment.get("probes", [experiment.get("probe")]):
                if probe not in probes:
                    return False, "report uses a rollout/loss cache; use the full evaluation path"
                pairs.add((probe, experiment["cluster"]))
        for episode in source["evaluation_episodes"]:
            for probe_name, cluster_name in pairs:
                raw = root / "raw" / probe_name / f"ep{episode['episode_index']:07d}.npz"
                with np.load(raw, allow_pickle=False) as payload:
                    meta = json.loads(str(payload["metadata_json"]))
                    if not {"descriptors", "replan_ts", "n_frames"}.issubset(payload.files):
                        return False, f"incomplete probe: {raw.name}"
                    if probes[probe_name].get("gaussian_center") == "pure_noise" and "final_descriptors" not in payload.files:
                        return False, f"missing final-step comparison: {raw.name}"
                    if int(payload["n_frames"]) != episode["length"]:
                        return False, "episode length changed"
                probe = probes[probe_name]
                basis = pcas.get(meta.get("pca_sha256"))
                descriptor = pcas.get(meta.get("descriptor_pca_sha256"))
                if basis is None or descriptor is None:
                    return False, "probe PCA hash does not match"
                for pca in (basis, descriptor):
                    if pca.get("source_tasks") != source.get("pca_source_tasks", source["source_tasks"]) or pca.get("action_mode") != source["action"]["mode"]:
                        return False, "PCA source or action mode changed"
                    if pca.get("episode_ids") != source.get("pca_episode_ids"):
                        return False, "PCA episode scope changed"
                if not _probe_matches(meta["variant"], probe, basis, descriptor):
                    return False, "probe recipe changed"
                if Path(meta["policy_path"]).resolve() != Path(source["policy_path"]).resolve() or meta["seed"] != settings["seed"]:
                    return False, "checkpoint or probe seed changed"
                for key in ("replan_interval", "eval_at_step"):
                    if meta[key] != int(probe.get(key, source["detector"][key])):
                        return False, f"{key} changed"
                if not require_clusters:
                    continue
                cluster = root / "clusters" / probe_name / cluster_name / raw.name
                with np.load(cluster, allow_pickle=False) as payload:
                    metadata = json.loads(str(payload["metadata_json"]))
                    if "selected_k" not in payload.files:
                        return False, "GMM cache is incomplete"
                if metadata["cluster"] != clusters[cluster_name] or metadata["raw_sha256"] != sha256(raw):
                    return False, "GMM recipe or probe data changed"
        if require_clusters:
            return True, "all selected DP/PCA/GMM caches match"
        return True, "all selected DP/PCA caches match; GMM and report run on CPU"
    except (OSError, ValueError, KeyError, EOFError) as exc:
        return False, f"cache missing or invalid: {exc}"
