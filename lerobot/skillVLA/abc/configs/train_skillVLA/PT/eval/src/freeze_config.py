#!/usr/bin/env python3
"""Pin moving 'last' symlinks in the already-snapshotted YAML before queuing."""
from pathlib import Path
import argparse
import errno
import os
import shutil
import sys
import tempfile
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "train_pi05/PT/eval/src"))
from eval_config import load_eval_config


def retain_checkpoint(source, target):
    """Hard-link immutable inference files before keep=1 can prune the source.

    Optimizer/training state lives outside pretrained_model and is not retained.
    Atomic publication also permits repeated submissions of the same contract.
    """
    if target.is_dir():
        if (target / "model.safetensors").stat().st_size != (source / "model.safetensors").stat().st_size:
            raise ValueError(f"retained checkpoint differs from its source: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pin-", dir=target.parent))
    try:
        for item in source.iterdir():
            if not item.is_file():
                raise ValueError(f"unexpected directory in inference checkpoint: {item}")
            try:
                os.link(item, temporary / item.name)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(item, temporary / item.name)
        try:
            temporary.rename(target)
        except OSError:
            if not target.is_dir():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def freeze(path, *, pin_models=False):
    config = load_eval_config(path)
    if config.policy_type != "skillvla":
        raise ValueError("Use policy.type: skillvla for ABC PT evaluation")
    data = yaml.safe_load(path.read_text())
    settings = dict(config.skillvla)
    if pin_models:
        settings["source_checkpoints"] = {key: settings[key] for key in ("vsa", "predictor", "terminator") if key in settings}
        for component, original in settings["source_checkpoints"].items():
            source = Path(original)
            target = config.artifact_dir / "models" / component / source.parent.name / "pretrained_model"
            retain_checkpoint(source, target)
            settings[component] = str(target)
    data["policy"]["skillvla"] = settings
    data.setdefault("output", {})["run_name"] = config.output_dir.name
    frozen = path.with_name(path.stem + ".pinned.yaml")
    frozen.write_text(yaml.safe_dump(data, sort_keys=False))
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pin-models", action="store_true")
    args = parser.parse_args()
    print(freeze(args.config.resolve(), pin_models=args.pin_models))
