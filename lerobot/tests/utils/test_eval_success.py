import gymnasium as gym
import numpy as np
import pytest

from lerobot.utils.eval_success import vector_successes


class OneStepEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self, success):
        self.success = success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), float(self.success), True, False, {"is_success": self.success}


@pytest.mark.parametrize("mode", list(gym.vector.AutoresetMode))
@pytest.mark.parametrize("success", [True, False])
def test_terminal_success_in_real_vector_environment(mode, success):
    # Coffee uses the default NEXT_STEP mode, whereas LIBERO uses SAME_STEP.
    env = gym.vector.SyncVectorEnv([lambda: OneStepEnv(success)], autoreset_mode=mode)
    try:
        env.reset()
        _, _, terminated, _, info = env.step(np.zeros((1, 1)))
        assert terminated.tolist() == [True]
        assert vector_successes(info, 1).tolist() == [success]
    finally:
        env.close()


def test_masks_do_not_turn_missing_values_into_successes():
    info = {
        "is_success": [True, True, False, False], "_is_success": [True, False, False, False],
        "final_info": {"is_success": [False, False, True, True], "_is_success": [False, False, True, True]},
        "_final_info": [False, False, True, False],
    }
    assert vector_successes(info, 4).tolist() == [True, False, True, False]
    assert vector_successes({}, 2).tolist() == [False, False]


def test_legacy_final_info_with_empty_slots():
    info = {"final_info": np.array([{"is_success": True}, None], dtype=object)}
    assert vector_successes(info, 2).tolist() == [True, False]
