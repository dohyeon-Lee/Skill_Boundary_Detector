"""Predictor choices and read-only migration of the retained historical models."""

PREDICTOR_MODES = ("mode1", "mode2")
PREDICTOR_MODE_SCHEMA = 2
PREDICTOR_START_XYZ = "skill_predictor_start_xyz"
PREDICTOR_TOP_ZOOM = 1.1
_LEGACY_RETAINED_MODES = {"mode3": "mode1", "mode3_preskill": "mode2"}


def normalize_predictor_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in PREDICTOR_MODES:
        raise ValueError(f"Unknown skill_predictor_mode {mode!r}; choose from {PREDICTOR_MODES}.")
    return value


def migrate_predictor_config(source: dict, *, required: bool = False) -> dict:
    """Migrate in memory only; never rewrite checkpoint files or directories.

    Unversioned mode1/2 meant different architectures. They must never be
    silently interpreted as the new crop-based models with the same names.
    """
    config = dict(source)
    mode = str(config.get("skill_predictor_mode", "mode1")).strip().lower()
    schema = int(config.get("skill_predictor_mode_schema", 1))
    active = (
        required
        or config.get("train_skill_predictor", False)
        or config.get("training_skill_source") == "predictor"
        or config.get("newtask_joint_enabled", False)
    )
    if schema not in (1, PREDICTOR_MODE_SCHEMA):
        raise ValueError(f"Unsupported Predictor mode schema {schema}.")
    if mode in _LEGACY_RETAINED_MODES:
        mode = _LEGACY_RETAINED_MODES[mode]
    elif schema < PREDICTOR_MODE_SCHEMA and active:
        raise ValueError(
            f"Retired legacy Predictor {mode!r}: it is not the new model with the same name. "
            "Existing mode3/mode3_preskill checkpoints remain supported."
        )
    elif not active and schema == 1:
        mode = "mode1"  # An unused legacy field owns no Predictor weights.
    config["skill_predictor_mode"] = normalize_predictor_mode(mode)
    config["skill_predictor_mode_schema"] = PREDICTOR_MODE_SCHEMA
    return config


def predictor_uses_previous_skill(mode: str) -> bool:
    return normalize_predictor_mode(mode) == "mode2"


def predictor_uses_top_crop(mode: str) -> bool:
    normalize_predictor_mode(mode)
    return True


def predictor_relative_xyz(mode: str, spatial_target: str = "xyz") -> bool:
    normalize_predictor_mode(mode)
    return spatial_target == "xyz"


def predictor_uses_wrist(mode: str) -> bool:
    normalize_predictor_mode(mode)
    return True


def predictor_uses_proprio(mode: str) -> bool:
    normalize_predictor_mode(mode)
    return True


def predictor_skill_conditions_xyz(mode: str) -> bool:
    normalize_predictor_mode(mode)
    return True


def validate_predictor_mode(mode: str, spatial_target: str, start_proprio: bool = False) -> str:
    mode = normalize_predictor_mode(mode)
    if spatial_target not in {"xyz", "uv"}:
        raise ValueError(
            "Predictor requires spatial_target xyz (relative endpoint) or uv "
            "(absolute endpoint in the cropped Top image)."
        )
    return mode
