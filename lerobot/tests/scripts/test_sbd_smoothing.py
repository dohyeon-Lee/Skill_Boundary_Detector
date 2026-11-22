"""Build and eval must share an explicit odd-window smoothing contract."""
import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import savgol_filter

from lerobot.utils.sbd_smoothing import smooth_savgol, validate_savgol

ROOT = Path(__file__).resolve().parents[2] / 'skillVLA/libero/configs/train_skills'


def smoothing_entry(file, name):
    # Load only the numerical entry point; the full builder also imports GPU policies.
    tree = ast.parse(file.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(file), 'exec'), namespace)
    return namespace[name]


BUILD = smoothing_entry(ROOT / 'build_data/src/build_skill_dataset.py', '_boundary_savgol')
EVAL = smoothing_entry(ROOT / 'DP/eval/src/evaluate.py', '_smooth')


@pytest.mark.parametrize('window', [0, 2, 8, 7.5])
def test_both_entries_reject_invalid_windows_without_rounding(window):
    for entry in (BUILD, EVAL):
        with pytest.raises(ValueError, match='positive odd integer'):
            entry(np.arange(20.), window, 4)


@pytest.mark.parametrize('length,window', [(30, 7), (30, 9), (6, 7), (4, 7), (0, 7), (12, 1)])
def test_build_and_eval_use_identical_smoothing_including_short_curves(length, window):
    values = np.sin(np.arange(length) * .7) + np.arange(length) ** 2
    built = BUILD(values, window, 4)
    np.testing.assert_array_equal(built, EVAL(values, window, 4))
    if length >= window > 1:
        np.testing.assert_allclose(built, savgol_filter(values, window, 4))
    elif length <= 4 or window == 1:
        np.testing.assert_array_equal(built, values)


def test_polynomial_order_is_validated_and_disabled_smoothing_is_supported():
    with pytest.raises(ValueError, match='polyorder'):
        validate_savgol(7, 7)
    assert validate_savgol(1, 4) == (1, 4)
