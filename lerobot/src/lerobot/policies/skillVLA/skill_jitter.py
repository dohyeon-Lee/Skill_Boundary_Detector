"""Stage-2 skill-transition randomization (jitter) — pure logic, no LeRobot deps.

At inference the FSQ terminator fires each skill transition slightly early/late, so the VLM's
"skill-start" observation comes from a frame that is off from the GT skill boundary. To make the
VLM robust to that, training picks a *jittered* skill start per sample (this module decides which
skill + which start-frame offset); the dataset then decodes that frame's image/state.

Inputs per frame (all from build_data columns):
  k        : skill_index (0-based; current skill the frame belongs to)
  ds, de   : distance-from-start / distance-to-end within the current skill (ds=0 at start, de=0 at last frame)
  seq_len  : skill_sequence_len = (#real skills N) + 1 (EOS).  last real skill index = seq_len-2
  pmax     : jitter half-window (ISS window = 2*pmax+1; offset stays in [-pmax, +pmax])

Three cases (mirrors the previous skill_boundary_random_p logic, but as a frame-index offset):
  early (near end,  de<p, k<last_real) : pretend the NEXT skill already started p frames early
                                          → k'=k+1, offset=-p
  late  (near start, ds<p, k>0)         : pretend the PREVIOUS skill is still running (late fire)
                                          → k'=k-1, offset=±p  (prev skill's own start, jittered)
  else                                   : jitter THIS skill's start by ±p
                                          → k'=k,   offset=±p
Both early & late eligible → coin flip. p ~ half-normal truncated at pmax (0 most likely).

Returns (k_prime, offset):
  start_frame  = IFS[k_prime] + offset      (clamp to the episode)
  iss_index    = pmax + offset              (index into the ISS window; always in [0, 2*pmax])
  skill_code   = SS[k_prime]                (VLM skill target + action-expert teacher-forced skill)
"""

from __future__ import annotations

import numpy as np


def sample_p(pmax: int, rng: np.random.Generator | None = None) -> int:
    """p ~ half-normal(std=pmax/2) truncated to [0, pmax], rounded to int (0 most likely)."""
    if pmax <= 0:
        return 0
    r = np.random if rng is None else rng
    return int(min(pmax, round(abs(r.normal(0.0, pmax / 2.0)))))


def sample_offset(pmax: int, rng: np.random.Generator | None = None) -> int:
    """Signed half-normal offset in [-pmax, pmax], shared by frame and transition-pack datasets."""
    r = np.random if rng is None else rng
    p = sample_p(pmax, rng)
    if p == 0:
        return 0
    return p if r.random() < 0.5 else -p


def choose_jitter(
    k: int,
    ds: int,
    de: int,
    seq_len: int,
    pmax: int,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Pick (k_prime, offset) for the skill-start jitter. See module docstring."""
    r = np.random if rng is None else rng
    last_real = seq_len - 2  # 0-based index of the last real skill (seq_len = N + EOS)
    p = sample_p(pmax, rng)

    # Skills use [start, end) frames. A p-frame early boundary includes de=0..p-1; a p-frame
    # late boundary leaves the previous skill active for ds=0..p-1. This also includes the exact
    # boundary frames, which the old ds/de!=0 guard accidentally excluded and shifted by one frame.
    can_early = p > 0 and de < p and k < last_real
    can_late = p > 0 and ds < p and k > 0
    if can_early and can_late:  # both eligible → coin flip
        if r.random() < 0.5:
            can_late = False
        else:
            can_early = False

    if can_early:
        return k + 1, -p          # next skill, started p early
    sign = 1 if r.random() < 0.5 else -1
    if can_late:
        return k - 1, sign * p    # still in prev skill; jitter its own start ±p
    return k, sign * p            # this skill, ±p jitter


def choose_jitter_torch(k, ds, de, seq_len, pmax: int):
    """Vectorized torch equivalent used by Stage-1 inside the training forward."""
    import torch  # local import keeps the NumPy dataset helper lightweight

    if pmax <= 0:
        return k, torch.zeros_like(k)
    p = torch.round(torch.abs(torch.randn(k.shape, device=k.device) * (pmax / 2.0))).long()
    p = p.clamp(max=pmax)
    last_real = seq_len - 2
    can_early = (p > 0) & (de < p) & (k < last_real)
    can_late = (p > 0) & (ds < p) & (k > 0)
    both = can_early & can_late
    choose_early = torch.rand(k.shape, device=k.device) < 0.5
    can_early = can_early & (~both | choose_early)
    can_late = can_late & (~both | ~choose_early)
    k_prime = torch.where(can_early, k + 1, torch.where(can_late, k - 1, k))
    sign = torch.where(
        torch.rand(k.shape, device=k.device) < 0.5,
        torch.ones_like(p),
        -torch.ones_like(p),
    )
    offset = torch.where(can_early, -p, sign * p)
    return k_prime, offset
