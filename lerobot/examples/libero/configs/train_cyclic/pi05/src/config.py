"""IID/block-cyclic PT: inherit train_pi05, changing only identity and sample order."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import re
import shutil
import tempfile

import yaml

from lerobot.utils.pi05_checkpointing import checkpointing_settings

HERE = Path(__file__).resolve()
CONFIGS = HERE.parents[3]
PI05_PT = CONFIGS / "train_pi05/PT"
_spec = importlib.util.spec_from_file_location("_cyclic_pi05_base_config", PI05_PT / "src/train_pi05_config.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
DEFAULT_CONFIG = HERE.parents[1] / "iid_config.yaml"


def _positive_int(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'non-negative' if zero else 'positive'} integer.")
    return value


def load_config(path: Path) -> dict:
    path = Path(path).resolve()
    local = yaml.safe_load(path.read_text()) or {}
    if not isinstance(local, dict):
        raise ValueError("Curriculum config must be a YAML mapping.")
    unknown = set(local) - {"pt_config", "mode", "cycle", "logging", "gradient_checkpointing", "run_suffix"}
    if unknown:
        raise ValueError(f"Set shared training/resource settings in the PT source YAML, not here: {sorted(unknown)}")
    source = Path(local.get("pt_config", "../../train_pi05/PT/pi05_PT_config.yaml"))
    if not source.is_absolute():
        source = path.parent / source
    source = source.resolve()
    if source == path:
        raise ValueError("pt_config cannot reference itself.")
    cfg = base.load_config(source)
    if str(cfg.get("stage", "")).lower() != "pt":
        raise ValueError("pt_config must point to a pi05 PT config (stage: pt).")
    return {"pt": cfg, "source": source, "mode": local.get("mode", "iid"),
            "cycle": local.get("cycle", {}), "logging": local.get("logging", {}),
            "gradient_checkpointing": local.get("gradient_checkpointing"),
            "run_suffix": local.get("run_suffix", "")}


def build_settings(cfg: dict) -> dict:
    mode = cfg["mode"]
    if mode not in {"iid", "cyclic"}:
        raise ValueError("mode must be iid or cyclic; Reptile is intentionally unsupported.")
    cycle = cfg.get("cycle")
    if cycle is None:
        cycle = {}
    if not isinstance(cycle, dict) or set(cycle) - {"n_groups", "phase_steps", "group_seed"}:
        raise ValueError("cycle accepts only n_groups, phase_steps and group_seed.")
    settings = base.build_settings(copy.deepcopy(cfg["pt"]))
    inherited_checkpointing = {k: v for k, v in settings.items() if k.startswith("pt_gradient_checkpointing")}
    checkpointing = cfg.get("gradient_checkpointing")
    if checkpointing is not None:
        settings.update(checkpointing_settings(checkpointing, "pt"))
    run_suffix = cfg.get("run_suffix", "")
    if not isinstance(run_suffix, str) or (run_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", run_suffix)):
        raise ValueError("run_suffix must contain only letters, numbers, underscores or hyphens.")
    resolved_checkpointing = {k: settings[k] for k in inherited_checkpointing}
    if checkpointing is not None and resolved_checkpointing != inherited_checkpointing and not run_suffix:
        raise ValueError("A checkpointing variant requires a distinct run_suffix.")
    if settings["pt_num_gpus"] != 1:
        raise ValueError("The IID/cyclic comparison currently requires training.dataloader.gpus: 1 in PT.")
    groups = _positive_int(cycle.get("n_groups", 10), "cycle.n_groups")
    phase = _positive_int(cycle.get("phase_steps", 200), "cycle.phase_steps")
    seed = _positive_int(cycle.get("group_seed", 0), "cycle.group_seed", zero=True)
    diagnostics = cfg.get("logging") or {}
    if not isinstance(diagnostics, dict) or set(diagnostics) - {
        "wandb_project", "probe_batches_per_group", "probe_seed", "probe_grad_group",
    }:
        raise ValueError("Unknown cyclic logging setting.")
    project = diagnostics.get("wandb_project", "cyclic_PT")
    if not isinstance(project, str) or not project.strip():
        raise ValueError("logging.wandb_project must be a nonempty string.")
    probe_batches = _positive_int(diagnostics.get("probe_batches_per_group", 2), "logging.probe_batches_per_group")
    probe_seed = _positive_int(diagnostics.get("probe_seed", 12345), "logging.probe_seed", zero=True)
    grad_group = diagnostics.get("probe_grad_group", 0)
    if isinstance(grad_group, bool) or not isinstance(grad_group, int) or not -1 <= grad_group < groups:
        raise ValueError("logging.probe_grad_group must be -1 (off) or a valid group ID.")
    _positive_int(settings["pt_steps"], "PT steps")
    _positive_int(settings["pt_batch_size"], "PT batch size")
    parent = settings["pi05_pt_outputs_root"].parent / "pi05_cyclic_PT"
    suffix = "iid" if mode == "iid" else f"cyclic_g{groups}p{phase}_s{seed}"
    name = f"{settings['pt_run_name']}_{suffix}"
    if run_suffix:
        name += f"_{run_suffix}"
    settings.update(
        pt_run_name=name,
        pt_output_dir=parent / name,
        pi05_pt_outputs_root=parent,
        cyclic_mode=mode,
        cyclic_n_groups=groups,
        cyclic_phase_steps=phase,
        cyclic_group_seed=seed,
        cyclic_source_config=cfg["source"],
        cyclic_probe_batches_per_group=probe_batches,
        cyclic_probe_seed=probe_seed,
        cyclic_probe_grad_group=grad_group,
        pt_wandb_project=project.strip(),
    )
    settings["train_bin"] = HERE.parent / "train.py"
    return settings


def freeze_config(path: Path) -> Path:
    """Freeze BOTH the referencing YAML and inherited PT/global/server layers for queued jobs."""
    path = Path(path).resolve()
    cfg = load_config(path)
    build_settings(cfg)  # validate before leaving a snapshot bundle
    root = path.parent / ".config_snapshots"
    root.mkdir(exist_ok=True)
    bundle = Path(tempfile.mkdtemp(prefix=path.stem + ".", dir=root))
    local = yaml.safe_load(path.read_text())
    source_name = "inherited_pi05_PT_config.yaml"
    local["pt_config"] = source_name
    shutil.copy2(cfg["source"], bundle / source_name)
    global_path = base._find_global(cfg["source"].parent)
    if global_path is None:
        raise ValueError("Inherited PT YAML must have a global_config.yaml ancestor.")
    shutil.copy2(global_path, bundle / "global_config.yaml")
    servers = global_path.parent / "servers"
    if servers.is_dir():
        shutil.copytree(servers, bundle / "servers")
    frozen = bundle / path.name
    frozen.write_text(yaml.safe_dump(local, sort_keys=False))
    return frozen


def check_output(settings: dict) -> None:
    output = Path(settings["pt_output_dir"])
    if output.is_symlink():
        raise ValueError(f"Refusing a symlinked output directory: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output is not a directory: {output}")
    if output.exists() and not (output / "checkpoints/last/pretrained_model/train_config.json").is_file():
        raise ValueError(f"Output exists without a resumable checkpoint; preserve/inspect it manually: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        print(freeze_config(args.config))
        return
    settings = build_settings(load_config(args.config))
    check_output(settings)
    if args.shell:
        base.print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
