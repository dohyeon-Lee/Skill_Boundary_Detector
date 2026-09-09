from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "drawsvg_dataset_config.py"
SPEC = importlib.util.spec_from_file_location("drawsvg_dataset_config", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_include_groups_preserves_yaml_order() -> None:
    assert MODULE.include_groups({"include_groups": ["circle", "square", "play"]}) == [
        "circle",
        "square",
        "play",
    ]


@pytest.mark.parametrize("value", [[], [""], ["same", "same"], ["../escape"], ["a/b"]])
def test_include_groups_rejects_invalid_values(value: list[str]) -> None:
    with pytest.raises(ValueError):
        MODULE.include_groups({"include_groups": value})


@pytest.mark.parametrize("value", ["", "../escape", "a/b", "."])
def test_output_name_rejects_non_folder_name(value: str) -> None:
    with pytest.raises(ValueError):
        MODULE.output_name({"output_name": value})
