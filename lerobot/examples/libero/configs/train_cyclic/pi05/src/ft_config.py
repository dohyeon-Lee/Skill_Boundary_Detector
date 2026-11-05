"""Dedicated cyclic/IID FT configuration using the current pi05 training infrastructure."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile

import yaml

_HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("_cyclic_ft_pt_config", _HERE.with_name("config.py"))
pt_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt_config)
base = pt_config.base
DEFAULT_CONFIG = _HERE.parents[1] / "FT/ft_config.yaml"


def _name(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"{field} must be a plain run name containing letters, numbers, '_' or '-'.")
    return value


def load_config(path):
    cfg = base.load_config(Path(path))
    if cfg.get("stage") != "ft":
        raise ValueError("Cyclic FT requires stage: ft.")
    return cfg


def build_settings(cfg, *, require_source=False):
    cfg = copy.deepcopy(cfg)
    source = cfg.get("ft_source")
    if not isinstance(source, dict) or set(source) - {"model_dir", "label"}:
        raise ValueError("ft_source must contain model_dir and an optional label.")
    run = _name(source.get("model_dir"), "ft_source.model_dir")
    label = _name(source.get("label", run), "ft_source.label")
    checkpoint = cfg.get("ft_source_checkpoint")
    if not isinstance(checkpoint, str) or not re.fullmatch(r"[0-9]{6,}", checkpoint):
        raise ValueError("ft_source_checkpoint must be an explicit quoted step such as '100000'.")
    every = pt_config._positive_int(cfg.get("ft_probe_every", 250), "ft_probe_every")
    cfg.update(ft_pretrained_run_name=run, ft_pretrained_checkpoint=checkpoint,
               ft_probe_pt_forgetting=False)
    if cfg.get("ft_num_gpus", 1) != 1:
        raise ValueError("Cyclic FT supports one process/GPU only.")
    settings = base.build_settings(cfg)
    if settings["ft_normalization_stats"] != "pt":
        raise ValueError("Cyclic FT requires training.normalization_stats: pt to preserve PT coordinates.")
    pt_config._positive_int(settings["ft_steps"], "FT steps")
    pt_config._positive_int(settings["ft_batch_size"], "FT batch size")
    # Include the entire source identity: IID and cyclic must never resume into one another.
    name = f"{settings['ft_run_name']}_from_{run}"
    _name(name, "FT run name")
    settings.update(ft_run_name=name, ft_output_dir=settings["pi05_ft_outputs_root"] / name,
                    cyclic_ft_source_label=label, cyclic_ft_probe_every=every)
    checkpoint_dir = Path(settings["ft_pretrained_model_path"])
    source_run = checkpoint_dir.parents[2]
    settings["cyclic_ft_source_run"] = source_run
    settings["cyclic_ft_source_ready"] = (checkpoint_dir / "train_config.json").is_file()
    if require_source:
        for path in (checkpoint_dir / "config.json", checkpoint_dir / "train_config.json",
                     checkpoint_dir / "policy_preprocessor.json", checkpoint_dir / "policy_postprocessor.json",
                     checkpoint_dir.parent / "training_state/training_step.json",
                     source_run / "groups.json", source_run / "cyclic_probe_config.json",
                     source_run / "cyclic_probe_batches.pt"):
            if not path.is_file():
                raise FileNotFoundError(f"Cyclic FT needs the completed PT checkpoint and saved probes: {path}")
        saved = json.loads((checkpoint_dir / "config.json").read_text())
        if saved.get("type") != "pi05":
            raise ValueError("ft_source must be a pi05 checkpoint.")
        step = json.loads((checkpoint_dir.parent / "training_state/training_step.json").read_text())
        if step.get("step") != int(checkpoint):
            raise ValueError("Source checkpoint does not match its declared PT step.")
        index = checkpoint_dir / "model.safetensors.index.json"
        shards = (set(json.loads(index.read_text())["weight_map"].values())
                  if index.is_file() else {"model.safetensors"})
        if not shards or any(Path(name).name != name or not (checkpoint_dir / name).is_file()
                             or (checkpoint_dir / name).stat().st_size == 0 for name in shards):
            raise ValueError("Source PT model weights are incomplete.")
    check_output(settings)
    return settings


def check_output(settings):
    output = Path(settings["ft_output_dir"])
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError(f"Invalid FT output directory: {output}")
    required = ("cyclic_ft_diagnostics.json", "pretrained_model/train_config.json")
    if output.exists() and not all((output / "checkpoints/last" / name).is_file() for name in required):
        raise ValueError(f"FT output exists without resumable diagnostics; preserve/inspect it manually: {output}")


def freeze_config(path):
    path = Path(path).resolve()
    build_settings(load_config(path), require_source=True)
    global_path = base._find_global(path.parent)
    if global_path is None:
        raise ValueError("FT config needs a global_config.yaml ancestor.")
    root = path.parent / ".config_snapshots"
    root.mkdir(exist_ok=True)
    bundle = Path(tempfile.mkdtemp(prefix=path.stem + ".", dir=root))
    shutil.copy2(path, bundle / path.name)
    shutil.copy2(global_path, bundle / "global_config.yaml")
    if (global_path.parent / "servers").is_dir():
        shutil.copytree(global_path.parent / "servers", bundle / "servers")
    return bundle / path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        print(freeze_config(args.config))
        return
    settings = build_settings(load_config(args.config), require_source=args.check_source)
    if args.shell:
        base.print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
