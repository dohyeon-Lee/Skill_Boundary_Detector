"""Lightweight, checkpoint-owned Predictor contracts; no model imports at launch."""

PREDICTOR_MODES = (
    "mode1", "mode2", "mode1_preskill", "mode2_preskill",
    "mode3_preskill", "mode3_preskill_wristX", "mode3_preskill_wristX_proprioX",
    "mode3_preskill_proprioX", "mode3",
)
PREDICTOR_START_XYZ = "skill_predictor_start_xyz"
MODE3_TOP_ZOOM = 1.1


def normalize_predictor_mode(mode: str) -> str:
    names = {name.lower(): name for name in PREDICTOR_MODES}
    try:
        return names[str(mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown skill_predictor_mode {mode!r}; choose from {PREDICTOR_MODES}.") from exc


def predictor_uses_previous_skill(mode: str) -> bool:
    return "preskill" in normalize_predictor_mode(mode).split("_")


def predictor_uses_top_crop(mode: str) -> bool:
    mode = normalize_predictor_mode(mode)
    return mode == "mode3" or mode.startswith("mode3_")


def predictor_relative_xyz(mode: str, spatial_target: str = "xyz") -> bool:
    return predictor_uses_top_crop(mode) and spatial_target == "xyz"


def predictor_uses_wrist(mode: str) -> bool:
    return "wristX" not in normalize_predictor_mode(mode).split("_")


def predictor_uses_proprio(mode: str) -> bool:
    return "proprioX" not in normalize_predictor_mode(mode).split("_")


def predictor_skill_conditions_xyz(mode: str) -> bool:
    return not normalize_predictor_mode(mode).startswith("mode2")


def validate_predictor_mode(mode: str, spatial_target: str, start_proprio: bool = False) -> str:
    mode = normalize_predictor_mode(mode)
    if predictor_uses_top_crop(mode) and spatial_target not in {"xyz", "uv"}:
        raise ValueError(
            "Predictor mode3 requires spatial_target xyz (relative endpoint) or uv "
            "(absolute endpoint in the cropped Top image)."
        )
    if not predictor_uses_proprio(mode) and start_proprio:
        raise ValueError("Predictor proprioX forbids the direct start-proprio adapter as well as State tokens.")
    return mode
