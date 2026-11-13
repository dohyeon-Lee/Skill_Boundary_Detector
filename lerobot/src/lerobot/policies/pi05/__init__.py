#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PI05Config",
    "PI05Policy",
    "make_pi05_pre_post_processors",
]

_EXPORTS = {
    "PI05Config": ("lerobot.policies.pi05.configuration_pi05", "PI05Config"),
    "PI05Policy": ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
    "make_pi05_pre_post_processors": (
        "lerobot.policies.pi05.processor_pi05",
        "make_pi05_pre_post_processors",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
