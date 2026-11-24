"""Read vector-environment success signals across Gymnasium autoreset modes."""

from collections.abc import Mapping

import numpy as np


def vector_successes(info: dict, num_envs: int) -> np.ndarray:
    """Include step info (NEXT_STEP/DISABLED) and final info (SAME_STEP).

    Gymnasium's underscore masks distinguish actual values from placeholders
    for environments that did not emit a field on this step.
    """
    successes = np.zeros(num_envs, dtype=bool)

    def read(values, outer_mask):
        if "is_success" in values:
            signals = np.asarray(values["is_success"], dtype=bool).reshape(num_envs)
            mask = np.asarray(values.get("_is_success", np.ones(num_envs)), dtype=bool)
            successes[:] |= signals & mask & outer_mask

    read(info, np.ones(num_envs, dtype=bool))
    final = info.get("final_info")
    if final is not None:
        mask = np.asarray(info.get("_final_info", np.ones(num_envs)), dtype=bool)
        if isinstance(final, Mapping):
            read(final, mask)
        else:
            # Older Gymnasium versions store one dict (or None) per environment.
            for index, values in enumerate(final):
                if mask[index] and isinstance(values, Mapping):
                    successes[index] |= bool(values.get("is_success", False))
    return successes
