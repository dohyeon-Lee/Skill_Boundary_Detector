#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import patch

import pytest
import torch

from lerobot.datasets.video_utils import decode_episode_video_frames


def test_decode_episode_video_frames_selects_each_target_timestamp():
    decoded = torch.zeros(3, 3, 8, 8)
    with patch("lerobot.datasets.video_utils.decode_video_frames", return_value=decoded) as decode:
        frames = decode_episode_video_frames(
            "packed.mp4",
            from_timestamp=12.3,
            to_timestamp=12.45,
            length=3,
            fps=20,
        )

    assert frames is decoded
    args, kwargs = decode.call_args
    assert args[0] == "packed.mp4"
    assert args[1] == pytest.approx([12.3, 12.35, 12.4])
    assert args[2] == 1e-4
    assert args[3] == "pyav"
    assert kwargs == {"decoder_num_threads": 1}


def test_decode_episode_video_frames_rejects_wrong_duration():
    with pytest.raises(ValueError, match="Episode duration mismatch"):
        decode_episode_video_frames(
            "packed.mp4",
            from_timestamp=10.0,
            to_timestamp=10.1,
            length=10,
            fps=20,
        )
