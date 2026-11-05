"""Lightweight pi05 component settings shared by model and YAML resolvers."""

COMPONENTS = ("expert", "llm", "vision")


def checkpointing_components(value):
    if isinstance(value, bool):
        return dict.fromkeys(COMPONENTS, value)
    if not isinstance(value, dict) or not value or set(value) - set(COMPONENTS):
        raise ValueError("gradient_checkpointing must be a boolean or an expert/llm/vision mapping.")
    if any(not isinstance(flag, bool) for flag in value.values()):
        raise ValueError("gradient_checkpointing component values must be YAML booleans.")
    return {name: value.get(name, False) for name in COMPONENTS}


def checkpointing_settings(value, stage):
    components = checkpointing_components(value)
    return {
        f"{stage}_gradient_checkpointing": any(components.values()),
        **{f"{stage}_gradient_checkpointing_{name}": flag for name, flag in components.items()},
    }
