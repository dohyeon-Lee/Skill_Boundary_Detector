"""Replay-free new-task FT with current pi05 training and source PT-group probes."""
from pathlib import Path
import importlib.util
import os

from ft_config import DEFAULT_CONFIG, base, build_settings, load_config
from ft_diagnostics import make_observer_factory


def main():
    from lerobot.scripts import lerobot_train as native

    path = Path(os.environ.get("TRAIN_PI05_CONFIG", DEFAULT_CONFIG))
    settings = build_settings(load_config(path), require_source=True)
    spec = importlib.util.spec_from_file_location(
        "_cyclic_ft_retention", base.DEFAULT_CONFIG_PATH.parents[1] / "FT/src/train.py"
    )
    retention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retention)
    retention.install_retention(native, settings)
    native._run_inline_cuda_guard()
    native.register_third_party_plugins()
    native.train(training_observer_factory=make_observer_factory(settings))


if __name__ == "__main__":
    main()
