"""LIBERO profile of the shared compact SBD recipe."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("libero_dp_defaults_config", Path(__file__).with_name("config.py"))
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)

SHARED = _config.load_workflow("defaults")
DEFAULTS = {**SHARED.DEFAULTS, "target_dataset": "libero_90_full_full", "source_tasks": [],
            "target_task": "libero_90", "task_id_space": "suite", "pca_scope": "selected",
            "report_image_key": "observation.images.image", "report_consistency_tolerance_frames": 6}
CONTROLS = {**SHARED.CONTROLS, "replan_interval": 5, "nms_frames": 15, "score_power": 2}


def resolve_config(raw):
    return SHARED.resolve_config(raw, domain="libero")
