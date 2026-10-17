"""Opt-in bridge to the dependency-free server-sync archive worker."""
import logging
from pathlib import Path
import subprocess
import sys


def _command(cfg, operation, checkpoint=None):
    if not getattr(cfg, "checkpoint_archive", ""):
        return
    script = Path(__file__).resolve().parents[4] / "tools/server_sync/checkpoint_archive.py"
    command = [sys.executable, str(script), operation]
    if operation == "register":
        command += ["--checkpoint", str(checkpoint), "--keep", str(cfg.keep_checkpoints),
                    "--destination", cfg.checkpoint_archive]
    else:
        state = Path(cfg.output_dir) / ".checkpoint_archive/config.json"
        if not state.is_file():
            return
        command += ["--run", str(cfg.output_dir)]
    subprocess.run(command, check=True)


def queue_checkpoint_archive(cfg, checkpoint):
    _command(cfg, "register", checkpoint)


def finish_checkpoint_archive(cfg):
    if getattr(cfg, "checkpoint_archive", ""):
        logging.info("Training updates finished; flushing checkpoint archive before releasing the job.")
        _command(cfg, "drain")
