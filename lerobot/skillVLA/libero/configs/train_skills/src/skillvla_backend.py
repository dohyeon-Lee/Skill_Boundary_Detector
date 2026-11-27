"""Shared PT/FT backbone selection; pi0.5 remains the default for old snapshots."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

BACKBONES = ("pi05", "groot_integrate", "groot_seperate")


def backend(config):
    value = config.get("skillvla_backbone", "pi05")
    if value not in BACKBONES:
        raise ValueError(f"skillvla_backbone must be one of {BACKBONES}, got {value!r}")
    return value


def prepare(config, path):
    selected = backend(config)
    if selected == "pi05" or not config.get("stage1_component"):
        return config
    config = copy.deepcopy(config)
    root = Path(config["project_root"])
    warm = config.setdefault("warm_start", {})
    warm["pi_base"] = warm.get("groot_base", "models/groot_n1_7_base")
    if "FT" not in Path(path).parts:
        return config
    component = config["stage1_component"]
    key = "vsa_checkpoint" if component == "VSA" else "predictor_checkpoint"
    override = warm.get("backend_sources", {}).get(selected, {}).get(key)
    if override is None:
        # Derive the PT run from its actual recipe, including crop/gripper probes.
        pt = root / "lerobot/skillVLA/libero/configs/train_skillVLA/PT"
        if component == "VSA":
            resolver, recipe = pt / "src/stage1_train_config.py", pt / "VSA/vsa_train_config.yaml"
        else:
            resolver, recipe = pt / "src/auxiliary/auxiliary_train_config.py", pt / "Predictor/predictor_train_config.yaml"
        spec = importlib.util.spec_from_file_location("_groot_pt_selector_" + component, resolver)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.load_stage1_component_config(recipe)
        cfg["skillvla_backbone"] = selected
        cfg["warm_start"]["pi_base"] = warm["pi_base"]
        settings = module.build_settings(cfg)
        override = {"run": settings.get("pt_run_name", settings.get("run_name")), "checkpoint": "100000"}
    from train_skills_config import resolve_run_checkpoint
    warm[key] = str(resolve_run_checkpoint(root / config["outputs_root"], component, override, field=f"warm_start.{selected}.{key}"))
    source = json.loads((Path(warm[key]) / "config.json").read_text())
    if source.get("skillvla_backbone", "pi05") != selected:
        raise ValueError(f"Selected {selected}, but {warm[key]} belongs to {source.get('skillvla_backbone', 'pi05')}")
    return config


def finalize(config, settings, *, run_key="run_name", output_key="output_dir"):
    selected = backend(config)
    settings["skillvla_backbone"] = selected
    if selected == "pi05":
        return settings
    name = settings[run_key]
    if selected not in name:
        name = name.replace("pi05vlm_9_9_noproprio", selected)
        if selected not in name:
            name += "_" + selected
    if len(name) > 230:
        digest = hashlib.sha256(name.encode()).hexdigest()[:10]
        name = name[:65] + "_run" + digest + "_" + name[-120:] + "_" + selected
    settings[run_key] = name
    settings[output_key] = Path(settings[output_key]).parent / name
    settings["groot_base_path"] = settings["pi_base"]
    processor = Path(settings["pi_base"]) / "processor"
    if processor.is_dir():
        # Remote launchers scan resolved inputs, including nested processor
        # assets when the destination already has the base weight directory.
        settings["groot_processor_path"] = processor
    return settings


def freeze_ft_source(path):
    """Pin auto-derived PT selectors in the private submission snapshot only."""
    from train_skills_config import load_stage1_component_config
    import yaml
    path = Path(path)
    if '.config_snapshots' not in path.parts:
        raise ValueError('FT source pinning only writes private submission snapshots')
    config = load_stage1_component_config(path)
    selected = backend(config)
    if selected == 'pi05' or config.get('stage1_component') not in {'VSA', 'Predictor'}:
        return
    key = 'vsa_checkpoint' if config['stage1_component'] == 'VSA' else 'predictor_checkpoint'
    source = Path(config['warm_start'][key])
    checkpoint = source.parent
    run = checkpoint.parent.parent.name
    raw = yaml.safe_load(path.read_text())
    sources = raw.setdefault('warm_start', {}).setdefault('backend_sources', {})
    sources.setdefault(selected, {})[key] = {'run': run, 'checkpoint': checkpoint.name}
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-ft-source', type=Path, required=True)
    freeze_ft_source(parser.parse_args().freeze_ft_source)
