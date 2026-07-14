import torch

from FSQ import FSQ


def _previous_forward(levels: list[int], z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-L=2-fix quantization formula, used for non-binary regression coverage."""
    lv = torch.tensor(levels, dtype=torch.float32)
    levels_half = (lv - 1.0) / 2.0
    offset = torch.where(lv % 2 == 0, torch.full_like(lv, 0.5), torch.zeros_like(lv))
    shift = torch.atanh(offset / levels_half)
    half_width = torch.div(lv, 2, rounding_mode="floor")
    strides = torch.ones(len(levels), dtype=torch.long)
    for i in range(1, len(levels)):
        strides[i] = strides[i - 1] * int(levels[i - 1])

    bounded = torch.tanh(z + shift) * levels_half - offset
    z_int = torch.round(bounded)
    z_q = bounded + (z_int - bounded).detach()
    indices = ((z_int + half_width).long() * strides).sum(dim=-1)
    return z_q, indices


def test_binary_level_reaches_both_codes_and_normalized_endpoints() -> None:
    quantizer = FSQ([2])
    z_q, codes = quantizer(torch.linspace(-20, 20, 1001).unsqueeze(-1))

    assert torch.equal(torch.unique(codes), torch.tensor([0, 1]))
    assert torch.equal(torch.unique(z_q), torch.tensor([-1.0, 0.0]))
    assert torch.equal(torch.unique(quantizer.normalized(z_q)), torch.tensor([-1.0, 1.0]))


def test_mixed_levels_with_a_binary_axis_reach_the_full_codebook() -> None:
    quantizer = FSQ([3, 3, 2])
    values = torch.tensor([-20.0, 0.0, 20.0])
    _, codes = quantizer(torch.cartesian_prod(values, values, values))

    assert torch.unique(codes).numel() == quantizer.codebook_size == 18


def test_nonbinary_levels_preserve_the_previous_quantization() -> None:
    levels = [3, 4, 8]
    z = torch.linspace(-8, 8, 101).unsqueeze(-1).repeat(1, len(levels))

    expected_z_q, expected_codes = _previous_forward(levels, z)
    z_q, codes = FSQ(levels)(z)

    torch.testing.assert_close(z_q, expected_z_q, rtol=0, atol=0)
    assert torch.equal(codes, expected_codes)
