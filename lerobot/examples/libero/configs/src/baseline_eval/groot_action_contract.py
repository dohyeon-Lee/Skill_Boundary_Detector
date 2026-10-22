"""Select gripper decoding from saved training statistics, never eval data."""
from __future__ import annotations

import json
import math
from pathlib import Path


def groot_action_decode_transform(checkpoint: Path | str) -> str:
    """Repair signed project checkpoints, preserving native zero-close checkpoints."""
    path = Path(checkpoint) / "policy_postprocessor.json"
    pipeline = json.loads(path.read_text())
    decoders = [step for step in pipeline.get("steps", []) if step.get("registry_name") in {
        "groot_n1_7_action_decode_v1", "groot_action_unpack_unnormalize_v2",
    }]
    if len(decoders) != 1:
        raise ValueError(f"Expected one saved GR00T action decoder in {path}")
    decoder = decoders[0]
    cfg = decoder.get("config", {})
    if decoder["registry_name"] == "groot_n1_7_action_decode_v1":
        stats = cfg.get("raw_stats", {}).get("action", {}).get("gripper", {})
        low = stats.get("min", stats.get("q01"))
        high = stats.get("max", stats.get("q99"))
        if not isinstance(low, list) or not isinstance(high, list) or len(low) != 1 or len(high) != 1:
            raise ValueError(f"Missing scalar gripper training range in {path}")
        low, high = low[0], high[0]
    else:
        # Fallback decoders store a padded action vector, not named groups.
        stats = cfg.get("stats", {}).get("action", {})
        low, high = stats.get("min"), stats.get("max")
        if (cfg.get("env_action_dim") != 7 or not isinstance(low, list) or not isinstance(high, list)
                or len(low) < 7 or len(high) < 7):
            raise ValueError(f"Missing seven-dimensional LIBERO action range in {path}")
        low, high = low[6], high[6]
    if (isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float))
            or not isinstance(high, (int, float)) or not math.isfinite(low) or not math.isfinite(high)):
        raise ValueError(f"Invalid gripper training range in {path}: {low}, {high}")
    if math.isclose(low, -1.0, abs_tol=1e-6) and math.isclose(high, 1.0, abs_tol=1e-6):
        return "none"
    if math.isclose(low, 0.0, abs_tol=1e-6) and math.isclose(high, 1.0, abs_tol=1e-6):
        return "libero"
    raise ValueError(f"Unsupported gripper training range in {path}: [{low}, {high}]")
