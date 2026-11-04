"""IID/block-cyclic PT: inherit train_pi05, changing only identity and sample order."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile

import yaml

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
    unknown = set(local) - {"pt_config", "mode", "cycle"}
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
    return {"pt": cfg, "source": source, "mode": local.get("mode", "iid"), "cycle": local.get("cycle", {})}


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
    if settings["pt_num_gpus"] != 1:
        raise ValueError("The IID/cyclic comparison currently requires training.dataloader.gpus: 1 in PT.")
    groups = _positive_int(cycle.get("n_groups", 8), "cycle.n_groups")
    phase = _positive_int(cycle.get("phase_steps", 250), "cycle.phase_steps")
    seed = _positive_int(cycle.get("group_seed", 0), "cycle.group_seed", zero=True)
    _positive_int(settings["pt_steps"], "PT steps")
    _positive_int(settings["pt_batch_size"], "PT batch size")
    parent = settings["pi05_pt_outputs_root"].parent / "pi05_cyclic_PT"
    suffix = "iid" if mode == "iid" else f"cyclic_g{groups}p{phase}_s{seed}"
    name = f"{settings['pt_run_name']}_{suffix}"
    settings.update(
        pt_run_name=name,
        pt_output_dir=parent / name,
        pi05_pt_outputs_root=parent,
        cyclic_mode=mode,
        cyclic_n_groups=groups,
        cyclic_phase_steps=phase,
        cyclic_group_seed=seed,
        cyclic_source_config=cfg["source"],
    )
    if mode == "cyclic":
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
