"""Opt-in frame sampling around clean skill boundaries and episode ends."""
from dataclasses import dataclass
import math


@dataclass
class TransitionOversamplingConfig:
    enabled: bool = False
    percent: float = 25.0
    distribution: str = "uniform"
    weight: float = 2.0

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError("transition_oversampling.enabled must be boolean.")
        if not self.enabled:
            return
        if (isinstance(self.percent, bool) or not isinstance(self.percent, (int, float))
                or not math.isfinite(self.percent) or not 0 < self.percent <= 100):
            raise ValueError("transition_oversampling.percent must be in (0, 100].")
        if self.distribution != "uniform":
            raise ValueError("transition_oversampling.distribution must be uniform.")
        if (isinstance(self.weight, bool) or not isinstance(self.weight, (int, float))
                or not math.isfinite(self.weight) or self.weight < 1):
            raise ValueError("transition_oversampling.weight must be finite and >= 1.")

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            value = {}
        if isinstance(value, bool):
            value = {"enabled": value}
        if not isinstance(value, dict) or set(value) - {"enabled", "percent", "distribution", "weight"}:
            raise ValueError("transition_oversampling accepts enabled, percent, distribution and weight.")
        config = cls(**value)
        config.validate()
        return config

    @property
    def suffix(self):
        return f"_os{self.percent:g}x{self.weight:g}" if self.enabled else ""
