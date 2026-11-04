"""Cyclic entrypoint; all non-sampling behavior stays in the native pi05 trainer."""
from pathlib import Path
import os

from config import build_settings, load_config
from sampling import make_loader_factory


def main():
    from lerobot.scripts import lerobot_train as native

    settings = build_settings(load_config(Path(os.environ["TRAIN_PI05_CONFIG"])))
    if settings["cyclic_mode"] != "cyclic":
        raise ValueError("IID must use the unmodified native lerobot-train entrypoint.")
    native._run_inline_cuda_guard()
    native.register_third_party_plugins()
    native.train(offline_dataloader_factory=make_loader_factory(settings))


if __name__ == "__main__":
    main()
