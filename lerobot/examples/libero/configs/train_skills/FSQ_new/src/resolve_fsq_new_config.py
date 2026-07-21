from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml


def nested(cfg: dict[str, Any], *keys: str, default: Any) -> Any:
    value: Any = cfg
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open() as handle:
        cfg = yaml.safe_load(handle) or {}

    values = {
        "fsq_new_context_queries": nested(cfg, "context", "queries", default=16),
        "fsq_new_context_layers": nested(cfg, "context", "layers", default=4),
        "fsq_new_context_heads": nested(cfg, "context", "heads", default=8),
        "fsq_new_context_dropout": nested(cfg, "context", "dropout", default=0.0),
        "fsq_new_a_skill_scale": nested(cfg, "conditioning", "A", "skill", default=1.0),
        "fsq_new_b_skill_scale": nested(cfg, "conditioning", "B", "skill", default=0.5),
        "fsq_new_c_skill_scale": nested(cfg, "conditioning", "C", "skill", default=0.5),
        "fsq_new_b_image_scale": nested(cfg, "conditioning", "B", "image", default=1.0),
        "fsq_new_c_image_scale": nested(cfg, "conditioning", "C", "image", default=0.5),
        "fsq_new_c_goal_scale": nested(cfg, "conditioning", "C", "goal", default=1.0),
        "fsq_new_a_weight": nested(cfg, "context_loss", "direct", "A", default=1.0),
        "fsq_new_b_weight": nested(cfg, "context_loss", "direct", "B", default=1.0),
        "fsq_new_c_weight": nested(cfg, "context_loss", "direct", "C", default=1.0),
        "fsq_new_ranking_weight": nested(cfg, "context_loss", "ranking", "weight", default=0.1),
        "fsq_new_ranking_margin": nested(
            cfg, "context_loss", "ranking", "relative_margin", default=0.05
        ),
        "fsq_new_wrong_goal_enabled": nested(
            cfg, "context_loss", "wrong_goal", "enabled", default=True
        ),
        "fsq_new_wrong_goal_weight": nested(
            cfg, "context_loss", "wrong_goal", "weight", default=0.1
        ),
        "fsq_new_wrong_goal_margin": nested(
            cfg, "context_loss", "wrong_goal", "relative_margin", default=0.05
        ),
        "fsq_new_context_lr": cfg.get("fsq_context_lr", 3e-4),
    }
    for key, value in values.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        print(f"export {key.upper()}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
