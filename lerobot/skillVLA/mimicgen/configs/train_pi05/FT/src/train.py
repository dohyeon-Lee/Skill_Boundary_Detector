"""Native pi05 training with the same checkpoint retention as LIBERO FT."""
from pathlib import Path
import os

from config import build_settings, common


def main():
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.set_num_interop_threads(1)
    from lerobot.scripts import lerobot_train as native

    cfg = common.load_config(Path(os.environ["TRAIN_PI05_CONFIG"]))
    settings = build_settings(cfg)
    retention = common.module("pi05_retention", common.SHARED / "train_pi05/FT/src/train.py")
    retention.install_retention(native, settings)
    native.main()


if __name__ == "__main__":
    main()
