"""Deterministic raw-metre goal perturbations for paired Stage-1 evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_TARGET_CODES = {"end": 1, "start": 2}


@dataclass
class GoalNoisePerturber:
    """Sample one reproducible XYZ perturbation per skill occurrence.

    The cache keeps the perturbation fixed across replans within a skill. Model
    identity is deliberately absent from the seed, so raw and normalized
    checkpoints receive identical perturbations in a paired evaluation.
    """

    std_m: float = 0.0
    target: str = "none"
    seed: int = 0
    _episode_serial: int = field(default=-1, init=False)
    _cache: dict[tuple[str, int, int], np.ndarray] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.std_m = float(self.std_m)
        self.target = str(self.target).strip().lower()
        self.seed = int(self.seed)
        if self.std_m < 0:
            raise ValueError("goal noise std_m must be non-negative.")
        if self.target not in {"none", "end", "start", "both"}:
            raise ValueError("goal noise target must be none|end|start|both.")

    @property
    def enabled(self) -> bool:
        return self.std_m > 0.0 and self.target != "none"

    def reset(self) -> None:
        """Begin a new rollout while preserving deterministic paired seeding."""
        self._episode_serial += 1
        self._cache.clear()

    def applies_to(self, target: str) -> bool:
        target = str(target).strip().lower()
        if target not in _TARGET_CODES:
            raise ValueError(f"Unknown goal noise target {target!r}.")
        return self.enabled and self.target in {target, "both"}

    def noise_xyz(
        self, *, target: str, batch_index: int, skill_order: int
    ) -> np.ndarray:
        """Return a cached float32 Gaussian perturbation in raw metres."""
        target = str(target).strip().lower()
        if not self.applies_to(target):
            return np.zeros(3, dtype=np.float32)
        key = (target, int(batch_index), int(skill_order))
        if key not in self._cache:
            sequence = np.random.SeedSequence(
                [
                    self.seed,
                    max(self._episode_serial, 0),
                    int(batch_index),
                    int(skill_order),
                    _TARGET_CODES[target],
                ]
            )
            rng = np.random.default_rng(sequence)
            self._cache[key] = rng.normal(0.0, self.std_m, size=3).astype(
                np.float32
            )
        return self._cache[key].copy()
