"""Install verified Coffee state/XML records for native policy evaluation."""
from pathlib import Path

import numpy as np


def coffee_records(path, n_episodes, repeat=False):
    if isinstance(n_episodes, bool) or not isinstance(n_episodes, int) or n_episodes < 1:
        raise ValueError("n_episodes must be a positive integer")
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"environment", "episode_index", "init_states", "model_xml"}
        if not required.issubset(data.files) or str(data["environment"].item()) != "Coffee_D0":
            raise ValueError("Coffee exact reset requires a verified Coffee_D0 state/XML bank")
        rows = sorted(zip(data["episode_index"], data["init_states"], data["model_xml"], strict=True),
                      key=lambda row: int(row[0]))
        if not rows or len({int(row[0]) for row in rows}) != len(rows):
            raise ValueError("Coffee exact bank must contain unique episode IDs")
        records = []
        for episode, state, xml in rows:
            state = np.asarray(state, dtype=np.float64)
            if state.ndim != 1 or not state.size or not np.isfinite(state).all() or "<mujoco" not in str(xml):
                raise ValueError(f"Invalid Coffee state/XML record for episode {episode}")
            records.append({"state": state.copy(), "model_xml": str(xml)})
    if len(records) < n_episodes and not repeat:
        raise ValueError(f"Coffee bank has {len(records)} exact episodes; requested {n_episodes}. Enable repeat_episodes.")
    return [records[index % len(records)] for index in range(n_episodes)]


def apply_coffee_init_states(envs, path, n_episodes, repeat=False):
    if set(envs) != {"Coffee_D0"} or set(envs["Coffee_D0"]) != {0}:
        raise ValueError("Coffee exact reset supports only Coffee_D0 task 0")
    vector = envs["Coffee_D0"][0]
    sub_envs = getattr(vector, "envs", None)
    if sub_envs is None:
        raise ValueError("Coffee exact reset requires eval.use_async_envs=false")
    records = coffee_records(path, n_episodes, repeat)
    for env in sub_envs:
        base = env.unwrapped
        base.init_states = True
        base._init_states = records
        base.init_state_id = base.episode_index
