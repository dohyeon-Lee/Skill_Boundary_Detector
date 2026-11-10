import os
from pathlib import Path
import subprocess
import sys

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "initialize_libero.py"


def test_fresh_config_does_not_import_or_prompt_and_preserves_custom_paths(tmp_path):
    package = tmp_path / "libero"
    (package / "libero").mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('Package must not be imported')")
    (package / "libero/__init__.py").write_text("input('No prompts allowed')")
    config = tmp_path / "config/config.yaml"
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "LIBERO_CONFIG_PATH": str(config.parent)}
    subprocess.run([sys.executable, str(SCRIPT)], env=env, stdin=subprocess.DEVNULL, check=True, capture_output=True)
    assert yaml.safe_load(config.read_text())["bddl_files"] == str(package / "libero/bddl_files")
    config.write_text("datasets: /custom/dataset\n")
    subprocess.run([sys.executable, str(SCRIPT)], env=env, stdin=subprocess.DEVNULL, check=True, capture_output=True)
    assert config.read_text() == "datasets: /custom/dataset\n"
