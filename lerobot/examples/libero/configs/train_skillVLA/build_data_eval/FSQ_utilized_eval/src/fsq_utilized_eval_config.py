#!/usr/bin/env python3
"""Config resolver + shared helpers for FSQ_utilized_eval.

--shell emits bash exports (slurm keys from global_config train_*); python eval scripts
import load_targets()/load_fsq() directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR.parent
DEFAULT_CONFIG_PATH = EVAL_DIR / "fsq_utilized_eval_config.yaml"
OUTPUTS_DIR = EVAL_DIR / "outputs"


def _find_global(start: Path) -> Path | None:
    for d in [start.resolve(), *start.resolve().parents]:
        candidate = d / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    gpath = _find_global(Path(path).parent)
    if gpath is not None:
        with open(gpath, encoding="utf-8") as f:
            gcfg = yaml.safe_load(f) or {}
        cfg = {**gcfg, **cfg}
    return cfg


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg["project_root"])).expanduser()


def load_targets(cfg: dict[str, Any]) -> dict[str, dict[str, Path | None]]:
    """name → {run_dir (abs), fsq_ckpt (abs|None)}. Relative paths join project_root."""
    proot = project_root(cfg)
    out: dict[str, dict[str, Path | None]] = {}
    for name, spec in (cfg.get("targets") or {}).items():
        run_dir = Path(str(spec["run_dir"]))
        if not run_dir.is_absolute():
            run_dir = proot / run_dir
        ckpt = spec.get("fsq_ckpt")
        if ckpt is not None:
            ckpt = Path(str(ckpt))
            if not ckpt.is_absolute():
                ckpt = proot / ckpt
        out[str(name)] = {"run_dir": run_dir, "fsq_ckpt": ckpt}
    return out


def metrics_path(name: str) -> Path:
    return OUTPUTS_DIR / name / "metrics.json"


def merge_metrics(name: str, section: str, payload: dict) -> Path:
    """Read-modify-write the target's metrics.json (one section per analysis group)."""
    path = metrics_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {"name": name}
    data[section] = payload
    path.write_text(json.dumps(data, indent=2))
    return path


def section_cached(name: str, section: str) -> bool:
    path = metrics_path(name)
    return path.exists() and section in json.loads(path.read_text())


def load_fsq(run_dir: Path, fsq_ckpt: Path | None, cfg: dict[str, Any]):
    """Load the SplineFSQAE from {run_dir}/FSQ.pt (or the override) → (model, cfg_dict).
    Adds examples/libero to sys.path for the checkpoint's FSQ classes."""
    import dataclasses

    import torch

    sys.path.insert(0, str(project_root(cfg) / "lerobot" / "examples" / "libero"))
    from FSQ import SplineFSQAE  # noqa: PLC0415

    keys = {"action_dim", "enc_dim", "state_dim", "n_control", "spline_degree", "hidden_dim", "fsq_levels",
            "num_layers", "dropout", "length_min", "length_max", "action_min", "action_max", "delta_min", "delta_max", "state_min", "state_max",
            "feat_dim", "n_tokens", "image_encoder_layers", "image_encoder_heads", "terminator_use_third", "terminator_use_wrist",
            "image_model_name", "image_size", "patch_grid", "n_patch_raw", "image_token_dim", "chunk_size",
            "reconstructor_mode"}
    path = fsq_ckpt or (run_dir / "FSQ.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = dataclasses.asdict(ckpt["cfg"])
    fsq = SplineFSQAE(**{k: v for k, v in cfg_dict.items() if k in keys})
    fsq.load_state_dict(ckpt["model_state"])
    fsq.eval()
    for p in fsq.parameters():
        p.requires_grad_(False)
    return fsq, cfg_dict


def fsq_levels_of(run_dir: Path, cfg: dict[str, Any]) -> list[int]:
    """Codebook levels from the run's FSQ.pt cfg (no weights kept)."""
    import dataclasses

    import torch

    sys.path.insert(0, str(project_root(cfg) / "lerobot" / "examples" / "libero"))
    import FSQ  # noqa: F401, PLC0415  (checkpoint cfg class)

    ckpt = torch.load(run_dir / "FSQ.pt", map_location="cpu", weights_only=False)
    return [int(x) for x in dataclasses.asdict(ckpt["cfg"])["fsq_levels"]]


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [p.strip() for p in text.split(",") if p.strip()] if text else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = {
        "project_root": project_root(cfg),
        "lerobot_root": project_root(cfg) / "lerobot",
        "eval_partition": ",".join(as_list(cfg.get("train_partition", ["debug"]))) or "debug",
        "eval_qos": str(cfg.get("train_qos", "base_qos")),
        "eval_exclude_nodes": ",".join(as_list(cfg.get("train_exclude_nodes", []))),
    }
    if args.shell:
        for k, v in settings.items():
            print(f"export {k.upper()}={shlex.quote(str(v))}")
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
