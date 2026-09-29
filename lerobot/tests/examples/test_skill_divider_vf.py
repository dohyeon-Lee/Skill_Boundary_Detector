import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "examples/libero/skill_divider.py"
SPEC = importlib.util.spec_from_file_location("skill_divider", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Scheduler:
    timesteps = torch.tensor([9])

    def add_noise(self, clean, noise, _timestep):
        return 0.5 * clean + 2.0 * noise

    def step(self, _prediction, _timestep, sample):
        return SimpleNamespace(
            prev_sample=sample + 1.0,
            pred_original_sample=sample + 10.0,
        )


class _UNet:
    def __call__(self, sample, _timestep, *, global_cond):
        assert global_cond.shape[0] == sample.shape[0]
        return torch.zeros_like(sample)


def _policy():
    return SimpleNamespace(
        diffusion=SimpleNamespace(noise_scheduler=_Scheduler(), unet=_UNet()),
        parameters=lambda: iter([torch.zeros(1)]),
    )


@pytest.mark.parametrize(
    ("denoise_output", "expected_offset"),
    [("prev_sample", 1.0), ("pred_original_sample", 10.0)],
)
def test_query_vf_chunks_selects_requested_scheduler_output(
    denoise_output: str, expected_offset: float
) -> None:
    chunks = torch.zeros(3, 4, 2)

    result = MODULE._query_vf_chunks(
        _policy(),
        torch.zeros(1, 5),
        chunks,
        eval_at_step=0,
        denoise_output=denoise_output,
    )

    np.testing.assert_allclose(result, expected_offset)


def test_query_vf_chunks_rejects_unknown_output() -> None:
    with pytest.raises(ValueError, match="denoise_output"):
        MODULE._query_vf_chunks(
            _policy(),
            torch.zeros(1, 5),
            torch.zeros(1, 4, 2),
            eval_at_step=0,
            denoise_output="unknown",
        )


def test_scheduler_gaussian_probes_use_forward_noise_and_keep_mean_reference() -> None:
    demo = np.arange(8, dtype=np.float32).reshape(4, 2)
    half = np.ones((1, 4, 2), dtype=np.float32)
    noises = np.concatenate([half, -half], axis=0)

    probes = MODULE._make_scheduler_gaussian_probes(
        _policy(), demo, noises, eval_at_step=0
    ).cpu().numpy()

    assert probes.shape == (3, 4, 2)
    np.testing.assert_allclose(probes[0], 0.5 * demo)
    np.testing.assert_allclose(probes[1], 0.5 * demo + 2.0)
    np.testing.assert_allclose(probes[2], 0.5 * demo - 2.0)
    np.testing.assert_allclose((probes[1] + probes[2]) / 2.0, probes[0])


def test_scheduler_gaussian_iid_mode_contains_only_noisy_draws() -> None:
    demo = np.arange(8, dtype=np.float32).reshape(4, 2)
    noises = np.stack(
        [np.ones_like(demo), 3.0 * np.ones_like(demo)], axis=0
    )

    probes = MODULE._make_scheduler_gaussian_probes(
        _policy(),
        demo,
        noises,
        eval_at_step=0,
        include_conditional_mean=False,
    ).cpu().numpy()

    assert probes.shape == (2, 4, 2)
    np.testing.assert_allclose(probes[0], 0.5 * demo + 2.0)
    np.testing.assert_allclose(probes[1], 0.5 * demo + 6.0)
