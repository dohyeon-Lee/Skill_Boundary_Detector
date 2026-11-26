"""Use the established LIBERO PT/FT segmentation modes for Coffee builds."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def data_mode(cfg):
    value = str(cfg.get("skillvla_data_mode", "ft_own")).strip().lower()
    if value not in {"pt", "ft", "ft_own", "ft_epi"}:
        raise ValueError("skillvla_data_mode must be pt|ft|ft_own|ft_epi")
    return value


def pt_reference(cfg):
    reference = cfg.get("skillvla_pt_reference", cfg.get("fsq_snap_reference"))
    if not isinstance(reference, dict):
        if not reference:
            raise ValueError("FT requires a skillvla_pt_reference or fsq_snap_reference")
        path = Path(reference).expanduser()
        path = path if path.is_absolute() else Path(cfg["project_root"]) / path
        return path.parent if path.suffix == ".npz" else path
    root = Path(reference.get("dataset_root", cfg["dataset_root"])).expanduser()
    if not root.is_absolute():
        root = Path(cfg["project_root"]) / root
    return root / reference.get("skillvla_root", "skillvla") / reference["source"] / reference["run"]


def pt_threshold(cfg):
    """Locate the threshold belonging to the selected PT dataset, not its FSQ run."""
    reference = pt_reference(cfg)
    if not (reference / "skill_latents.npz").is_file():
        raise FileNotFoundError(f"PT reference is not built: {reference}")
    # Coffee PT builds record their SBD source directly; historical LIBERO
    # builds instead identify the matching _work artifact by its contract.
    source_path = reference / "sbd_source.json"
    if source_path.is_file():
        source = json.loads(source_path.read_text())
        if source.get("skillvla_data_mode") != "pt":
            raise ValueError(f"The selected reference is not PT data: {reference}")
        threshold = Path(source["global_threshold_path"])
        if hashlib.sha256(threshold.read_bytes()).hexdigest() != source["global_threshold_sha256"]:
            raise ValueError(f"PT threshold has changed since its data was built: {threshold}")
        return threshold.resolve()
    contract_path = reference / "segmentation_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"PT segmentation provenance is required: {contract_path}")
    expected = json.loads(contract_path.read_text())["contract"]["segmentation"]
    matches = []
    for path in sorted((reference.parent / "_work").glob("*/skillset/global_boundary_threshold.json")):
        payload = json.loads(path.read_text())
        if (payload.get("boundary_threshold_mode") == "global_mean"
                and payload.get("segmentation_contract") == expected):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError(f"Expected one matching PT threshold for {reference}; found {matches}")
    return matches[0]


def configure_data_mode(cfg):
    # Frozen configurations without this selector keep their historical behavior.
    if "skillvla_data_mode" not in cfg:
        return cfg
    cfg = deepcopy(cfg)
    mode = data_mode(cfg)
    submitted_pca_hash = cfg.get("skillset_pca_reference_sha256", "")
    cfg["skillvla_data_mode"] = mode
    cfg["skillset_boundary_threshold_mode"] = "episode_mean" if mode == "ft_epi" else "global_mean"
    cfg["skillset_global_threshold_source"] = str(pt_threshold(cfg)) if mode == "ft" else ""
    cfg["skillset_pca_reference_path"] = ""
    cfg["skillset_pca_reference_sha256"] = ""
    if mode in {"ft", "ft_own"}:
        threshold = Path(cfg["skillset_global_threshold_source"]) if mode == "ft" else pt_threshold(cfg)
        pca = threshold.with_name("action_probe_pca.npz")
        if not pca.is_file():
            raise FileNotFoundError(f"FT requires the selected PT's PCA artifact: {pca}")
        cfg["skillset_pca_reference_path"] = str(pca)
        cfg["skillset_pca_reference_sha256"] = hashlib.sha256(pca.read_bytes()).hexdigest()
        if submitted_pca_hash and submitted_pca_hash != cfg["skillset_pca_reference_sha256"]:
            raise ValueError(f"PT PCA has changed since submission: {pca}")
    tag = "globalref" if mode == "ft" else "episodemean" if mode == "ft_epi" else "globalmean"
    for key in ("skillset_seg_name", "skillset_output_suffix"):
        if key in cfg:
            cfg[key] = str(cfg[key]).format(skillvla_data_mode=mode, skillvla_threshold_tag=tag)
            if mode in {"ft", "ft_own"} and not cfg[key].endswith("_pcapt"):
                cfg[key] += "_pcapt"
    return cfg


def validate_skillset_mode(cfg, manifest, skillset_dir):
    if "skillvla_data_mode" not in cfg:
        return
    detector = manifest["detector"]
    expected = cfg["skillset_boundary_threshold_mode"]
    if detector.get("boundary_threshold_mode") != expected:
        raise ValueError(f"SBD threshold mode does not match skillvla_data_mode={data_mode(cfg)}; rebuild SBD with the same mode")
    if data_mode(cfg) == "ft":
        actual = Path(detector.get("global_threshold_path", "")).resolve()
        target = Path(cfg["skillset_global_threshold_source"]).resolve()
        if actual != target:
            raise ValueError(f"FT SBD must reuse the selected PT threshold ({target}); got {actual}")
        if not target.is_file():
            raise FileNotFoundError(target)
    elif expected == "global_mean":
        actual = Path(detector.get("global_threshold_path", "")).resolve()
        target = (Path(skillset_dir) / "global_boundary_threshold.json").resolve()
        if actual != target:
            raise ValueError(f"{data_mode(cfg)} requires this dataset's own mean threshold; got {actual}")
    if data_mode(cfg) in {"ft", "ft_own"}:
        expected_hash = cfg["skillset_pca_reference_sha256"]
        if manifest.get("probe", {}).get("pca_reference_sha256") != expected_hash:
            raise ValueError("FT SBD must reuse the selected PT PCA; rebuild SBD in the _pcapt output")
        pca = Path(cfg["skillset_pca_reference_path"])
        if hashlib.sha256(pca.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"PT PCA has changed since submission: {pca}")
