"""pi05 FT entrypoint: native training with GR00T-style checkpoint retention."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import sys


def _is_training_checkpoint(path: Path) -> bool:
    required = (
        "pretrained_model/config.json", "pretrained_model/train_config.json",
        "pretrained_model/policy_preprocessor.json", "pretrained_model/policy_postprocessor.json",
        "training_state/training_step.json", "training_state/optimizer_state.safetensors",
        "training_state/optimizer_param_groups.json", "training_state/scheduler_state.json",
        "training_state/rng_state.safetensors",
    )
    if not all((path / name).is_file() for name in required):
        return False
    try:
        if json.loads((path / "pretrained_model/config.json").read_text()).get("type") != "pi05":
            return False
        if json.loads((path / "training_state/training_step.json").read_text())["step"] != int(path.name):
            return False
        pretrained = path / "pretrained_model"
        index = pretrained / "model.safetensors.index.json"
        shards = (set(json.loads(index.read_text())["weight_map"].values())
                  if index.is_file() else {"model.safetensors"})
        return bool(shards) and all(
            Path(name).name == name and (pretrained / name).is_file()
            and (pretrained / name).stat().st_size > 0 for name in shards
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def prune_checkpoints(run: Path, keep: int, protected_steps) -> list[Path]:
    """Keep newest N complete saves plus protected steps; never follow checkpoint symlinks."""
    if keep == -1:
        return []
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("keep_checkpoints must be -1 or a positive integer.")
    root = Path(run) / "checkpoints"
    if not root.is_dir():
        return []
    if root.is_symlink():
        raise ValueError(f"Refusing to prune a symlinked checkpoint directory: {root}")
    paths = sorted(
        (path for path in root.iterdir() if path.name.isdigit() and not path.is_symlink()
         and path.is_dir() and _is_training_checkpoint(path)),
        key=lambda path: int(path.name),
    )
    retained = set(paths[-keep:])
    last = root / "last"
    if last.is_symlink():
        retained.update(path for path in paths if path.resolve() == last.resolve())
    protected = set(protected_steps)
    removed = []
    for path in paths:
        if path in retained or int(path.name) in protected:
            continue
        if path.parent.resolve() != root.resolve():
            raise ValueError(f"Refusing to prune outside checkpoint directory: {path}")
        logging.info("Pruning pi05 FT checkpoint %s", path)
        shutil.rmtree(path)
        removed.append(path)
    return removed


def install_retention(native, settings) -> None:
    """Run only after the native trainer has finished saving and updated `last`."""
    original = native.update_last_checkpoint
    output = Path(settings["ft_output_dir"])

    def update_last(path):
        if Path(path).parent.resolve() != (output / "checkpoints").resolve():
            raise ValueError(f"Unexpected pi05 FT checkpoint directory: {path}")
        result = original(path)
        prune_checkpoints(output, settings["ft_keep_checkpoints"], settings["ft_protect_checkpoints"])
        return result

    native.update_last_checkpoint = update_last


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PT/src"))
    from train_pi05_config import build_settings, load_config
    from lerobot.scripts import lerobot_train as native

    config_path = Path(os.environ.get("TRAIN_PI05_CONFIG", Path(__file__).resolve().parents[1] / "pi05_FT_config.yaml"))
    settings = build_settings(load_config(config_path))
    install_retention(native, settings)
    native.main()


if __name__ == "__main__":
    main()
