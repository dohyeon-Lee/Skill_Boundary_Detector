"""Shared configuration and resource discovery for MimicGen pi05 workflows."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "setup_env.sh").is_file())
CONFIGS = CODE_ROOT / "lerobot/skillVLA/mimicgen/configs"
SHARED = CODE_ROOT / "lerobot/skillVLA/libero/configs"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def resource_root():
    """Managed worktrees may share their parent checkout's environments and assets."""
    if (CODE_ROOT / ".venv/bin/python").is_file():
        return CODE_ROOT
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=CODE_ROOT,
        check=True, capture_output=True, text=True,
    )
    common = Path(result.stdout.strip())
    common = common if common.is_absolute() else CODE_ROOT / common
    root = common.resolve().parent
    if not (root / ".venv/bin/python").is_file():
        raise FileNotFoundError("Run bash setup_env.sh in this checkout before using pi05.")
    return root


def load_config(path):
    path = Path(path)
    local = yaml.safe_load(path.read_text()) or {}
    if local.get("_resolved_pi05"):
        return local
    loader = module("pi05_global", SHARED / "src/global_config_loader.py")
    global_path = next(p / "global_config.yaml" for p in path.resolve().parents
                       if (p / "global_config.yaml").is_file())
    root = resource_root()
    profiles = loader.server_config_dir(global_path)
    requested = (yaml.safe_load(global_path.read_text()) or {}).get("server", "auto")
    server, _ = loader.select_server(profiles, root, requested)
    cfg = loader.load_global_config(global_path, server=server)
    # Preserve an explicitly configured storage root; otherwise borrow the parent's resources.
    if Path(cfg["project_root"]).resolve() == CODE_ROOT.resolve():
        cfg["project_root"] = str(root)
    cfg.update(local)
    return cfg


def _yaml_values(value):
    """Keep resolved filesystem paths portable in frozen workflow YAMLs."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_values(item) for item in value]
    return value


def command_line(default, prepare, build):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    cfg = prepare(load_config(args.config))
    settings = build(cfg)
    if args.freeze:
        if settings.get("policy_path") and not (Path(settings["policy_path"]) / "config.json").is_file():
            raise FileNotFoundError(f'FT checkpoint not ready: {settings["policy_path"]}; no job was submitted')
        args.freeze.mkdir(parents=True, exist_ok=False)
        frozen = args.freeze / args.config.name
        frozen.write_text(yaml.safe_dump(_yaml_values({**cfg, "_resolved_pi05": True}), sort_keys=False))
        # submit_job reads this global layer for the scheduler, using the frozen profile.
        (args.freeze / "global_config.yaml").write_text(yaml.safe_dump(_yaml_values({
            **cfg, "server": cfg["server"],
        }), sort_keys=False))
        settings["frozen_config"] = str(frozen)
    if args.shell:
        module("pi05_shell", SHARED / "train_pi05/PT/src/train_pi05_config.py").print_shell(settings)
    else:
        print(json.dumps(settings, default=str, indent=2))
