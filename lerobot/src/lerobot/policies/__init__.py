# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy configuration exports, loaded only when requested.

Keeping this package initializer lazy lets inference-only environments import
one policy without importing every policy's optional training dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ACTConfig": ("lerobot.policies.act.configuration_act", "ACTConfig"),
    "DiffusionConfig": ("lerobot.policies.diffusion.configuration_diffusion", "DiffusionConfig"),
    "GrootConfig": ("lerobot.policies.groot.configuration_groot", "GrootConfig"),
    "MultiTaskDiTConfig": (
        "lerobot.policies.multi_task_dit.configuration_multi_task_dit",
        "MultiTaskDiTConfig",
    ),
    "PI0Config": ("lerobot.policies.pi0.configuration_pi0", "PI0Config"),
    "PI0FastConfig": ("lerobot.policies.pi0_fast.configuration_pi0_fast", "PI0FastConfig"),
    "PI05Config": ("lerobot.policies.pi05.configuration_pi05", "PI05Config"),
    "SkillAuxConfig": ("lerobot.policies.skill_aux.configuration_skill_aux", "SkillAuxConfig"),
    "SkillExpertConfig": (
        "lerobot.policies.skill_expert.configuration_skill_expert",
        "SkillExpertConfig",
    ),
    "SmolVLAConfig": ("lerobot.policies.smolvla.configuration_smolvla", "SmolVLAConfig"),
    "SmolVLANewLineProcessor": (
        "lerobot.policies.smolvla.processor_smolvla",
        "SmolVLANewLineProcessor",
    ),
    "TDMPCConfig": ("lerobot.policies.tdmpc.configuration_tdmpc", "TDMPCConfig"),
    "VQBeTConfig": ("lerobot.policies.vqbet.configuration_vqbet", "VQBeTConfig"),
    "WallXConfig": ("lerobot.policies.wall_x.configuration_wall_x", "WallXConfig"),
    "XVLAConfig": ("lerobot.policies.xvla.configuration_xvla", "XVLAConfig"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
