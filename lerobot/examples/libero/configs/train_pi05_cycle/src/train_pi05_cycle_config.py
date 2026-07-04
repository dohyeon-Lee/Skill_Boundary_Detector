#!/usr/bin/env python3
"""Config resolver for configs/train_pi05_cycle (block-cyclic PT mini-experiments).

Emits shell exports for cycle_PT.sbatch. PT-only: FT/eval reuse the regular
configs/train_pi05 pipeline on the produced checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "cycle" / "cycle_config.yaml"


def _find_global(start: Path) -> Path | None:
    for d in [start.resolve(), *start.resolve().parents]:
        candidate = d / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    gpath = _find_global(config_path.parent)
    if gpath is not None and gpath.resolve() != config_path.resolve():
        with open(gpath, "r", encoding="utf-8") as f:
            gcfg = yaml.safe_load(f) or {}
        cfg = {**gcfg, **cfg}
    return cfg


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [part.strip() for part in text.split(",") if part.strip()] if text else []


def get_value(cfg: dict[str, Any], key: str, default: Any = None, *, env: str | None = None) -> Any:
    if env and env in os.environ:
        return os.environ[env]
    return cfg.get(key, default)


def resolve_path(project_root: "Path | str", value: Any, default: str = "") -> str:
    s = str(value if value not in (None, "", "null") else default).strip()
    if not s:
        return ""
    p = Path(s).expanduser()
    return str(p if p.is_absolute() else (Path(project_root) / p))


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, (list, tuple, dict)):
        value = json.dumps(value)
    return shlex.quote(str(value))


def print_shell(settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        print(f"export {key.upper()}={shell_value(value)}")


def fmt_num(x: float) -> str:
    """Compact number for run names: 0.5 → '05', 1.0 → '1', 0.25 → '025'."""
    s = f"{x:g}".replace(".", "")
    return s


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    lerobot_root = project_root / "lerobot"
    outputs_root = str(get_value(cfg, "outputs_root", "outputs"))
    cycle_pt_root = project_root / outputs_root / str(get_value(cfg, "pi05_cycle_outputs_root", "pi05_cycle_PT"))

    pt_dataset = str(get_value(cfg, "pt_dataset", "libero_90_full_full", env="PT_DATASET"))
    pt_dataset_root = str(get_value(cfg, "pt_dataset_root", get_value(cfg, "dataset_root", "libero_dataset"), env="PT_DATASET_ROOT"))
    pt_batch_size = int(get_value(cfg, "pt_batch_size", 16, env="PT_BATCH_SIZE"))
    pt_exp = str(get_value(cfg, "pt_exp", "", env="PT_EXP")).strip()

    n_groups = int(get_value(cfg, "cycle_n_groups", 8, env="CYCLE_N_GROUPS"))
    phase_steps = int(get_value(cfg, "cycle_phase_steps", 500, env="CYCLE_PHASE_STEPS"))
    n_cycles = int(get_value(cfg, "cycle_n_cycles", 0, env="CYCLE_N_CYCLES"))  # >0 overrides phase_steps
    delta_lambda = float(get_value(cfg, "cycle_delta_lambda", 0.0, env="CYCLE_DELTA_LAMBDA"))
    reptile_beta = float(get_value(cfg, "cycle_reptile_beta", 1.0, env="CYCLE_REPTILE_BETA"))
    iid_baseline = str(get_value(cfg, "cycle_iid_baseline", False, env="CYCLE_IID_BASELINE")).strip().lower() in {"1", "true", "yes", "on"}

    # Condition is encoded in the run name so wandb/output dirs separate ablations cleanly.
    # Encode whichever was specified: c{n_cycles} (phase auto-computed) or p{phase_steps}.
    sched_tag = f"c{n_cycles}" if n_cycles > 0 else f"p{phase_steps}"
    prefix = "PTiid" if iid_baseline else "PTcyc"
    run_name = f"{prefix}_{pt_dataset}_pi05_batch{pt_batch_size}_g{n_groups}{sched_tag}"
    if not iid_baseline and delta_lambda > 0:
        run_name += f"_lam{fmt_num(delta_lambda)}"
    if not iid_baseline and reptile_beta < 1.0:
        run_name += f"_b{fmt_num(reptile_beta)}"
    if pt_exp:
        run_name += f"_{pt_exp}"

    settings = {
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        "python_bin": project_root / ".venv" / "bin" / "python",
        "cycle_train_script": Path(__file__).resolve().parent / "lerobot_train_cycle.py",
        "pi05_cycle_outputs_root": cycle_pt_root,
        "pi_base": resolve_path(project_root, get_value(cfg, "pi_base", "models/pi05_base")),
        # PT
        "pt_dataset": pt_dataset,
        "pt_dataset_root": pt_dataset_root,
        "pt_dataset_dir": project_root / pt_dataset_root / pt_dataset,
        "pt_batch_size": pt_batch_size,
        "pt_num_workers": int(get_value(cfg, "pt_num_workers", 4, env="PT_NUM_WORKERS")),
        "pt_exp": pt_exp,
        "pt_lr": float(get_value(cfg, "pt_lr_base", 2.5e-05, env="PT_LR_BASE")),
        "pt_steps": int(get_value(cfg, "pt_steps", 20000, env="PT_STEPS")),
        "pt_save_freq": int(get_value(cfg, "pt_save_freq", 5000, env="PT_SAVE_FREQ")),
        "pt_log_freq": int(get_value(cfg, "pt_log_freq", 200, env="PT_LOG_FREQ")),
        "pt_wandb_project": str(get_value(cfg, "pt_wandb_project", "VLA_cycle", env="PT_WANDB_PROJECT")),
        "pt_run_name": run_name,
        "pt_output_dir": cycle_pt_root / run_name,
        # cycle curriculum
        "cycle_n_groups": n_groups,
        "cycle_phase_steps": phase_steps,
        "cycle_n_cycles": n_cycles,
        "cycle_group_seed": int(get_value(cfg, "cycle_group_seed", 0, env="CYCLE_GROUP_SEED")),
        "cycle_delta_lambda": delta_lambda,
        "cycle_delta_max_weight": float(get_value(cfg, "cycle_delta_max_weight", 3.0, env="CYCLE_DELTA_MAX_WEIGHT")),
        "cycle_reptile_beta": reptile_beta,
        "cycle_probe_batches": int(get_value(cfg, "cycle_probe_batches", 2, env="CYCLE_PROBE_BATCHES")),
        "cycle_probe_seed": int(get_value(cfg, "cycle_probe_seed", 12345, env="CYCLE_PROBE_SEED")),
        "cycle_probe_grad_group": int(get_value(cfg, "cycle_probe_grad_group", -1, env="CYCLE_PROBE_GRAD_GROUP")),
        "cycle_iid_baseline": iid_baseline,
    }
    # Slurm (partition/qos/nodelist/exclude canonical in global_config.yaml)
    settings.update({
        "pt_partition": ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "pt_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "pt_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
        "pt_qos": str(get_value(cfg, "train_qos", "big_qos")),
        "pt_gres": str(get_value(cfg, "pt_gres", "gpu:1")),
        "pt_cpus_per_task": int(get_value(cfg, "pt_cpus_per_task", 16)),
        "pt_mem": str(get_value(cfg, "pt_mem", "128G")),
        "pt_time": str(get_value(cfg, "pt_time", "24:00:00")),
    })
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
