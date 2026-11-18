"""Load the shared statistics reader without importing an isolated LeRobot."""
import importlib.util
from pathlib import Path
import sys

_NAME = "sbd_ft_normalization"
if _NAME not in sys.modules:
    path = Path(__file__).resolve().parents[4] / "src/lerobot/utils/ft_normalization.py"
    spec = importlib.util.spec_from_file_location(_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
_shared = sys.modules[_NAME]
normalization_stats_mode = _shared.normalization_stats_mode
checkpoint_normalization_stats = _shared.checkpoint_normalization_stats
with_checkpoint_normalization = _shared.with_checkpoint_normalization
validate_resume_normalization = _shared.validate_resume_normalization

