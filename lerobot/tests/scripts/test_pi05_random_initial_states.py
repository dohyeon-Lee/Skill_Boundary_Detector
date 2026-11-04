from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_pi05/PT/eval/src"
sys.path.insert(0, str(SRC))
from random_initial_states import scale_intervals, scale_workspace_samplers


def test_intervals_keep_centers_and_zero_widths():
    expanded = scale_intervals([[-.125, -.075], [.25, .25]], 2.)
    assert expanded[0] == pytest.approx([-.15, -.05])
    assert expanded[1] == [.25, .25]


def test_only_movable_workspace_samplers_are_scaled():
    def sampler(name):
        return SimpleNamespace(mujoco_objects=[SimpleNamespace(name=name)], x_ranges=[[-1., 1.]], y_ranges=[[3., 5.]])
    movable, fixture, conditional = sampler("can"), sampler("cabinet"), sampler("can")
    domain = SimpleNamespace(
        objects_dict={"can": object()},
        placement_initializer=SimpleNamespace(samplers={"can": movable, "cabinet": fixture}),
        conditional_placement_initializer=SimpleNamespace(samplers={"inside": conditional}),
    )
    changes = scale_workspace_samplers(domain, 2.)
    assert movable.x_ranges == [[-2., 2.]] and movable.y_ranges == [[2., 6.]]
    assert fixture.x_ranges == conditional.x_ranges == [[-1., 1.]]
    assert changes["can"]["original"]["x_ranges"] == [[-1., 1.]]
    assert set(changes) == {"can"}


def test_no_supported_placement_fails_instead_of_silent_noop():
    domain = SimpleNamespace(objects_dict={}, placement_initializer=SimpleNamespace(samplers={}))
    with pytest.raises(ValueError, match="no supported"):
        scale_workspace_samplers(domain, 2.)
