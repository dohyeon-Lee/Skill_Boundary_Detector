#!/usr/bin/env python

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
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dino_feature_dataset import DinoFrameFeatureDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.datasets.skillvla_dino_token_dataset import SkillVLADinoTokenDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            # DINO / state-only modes condition on observation.state only — skip windowing other obs.
            vision_off = getattr(cfg, "use_dino_features", False) or getattr(cfg, "state_only", False)
            if vision_off and key != OBS_STATE:
                continue
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        # DINO uses precomputed tokens, state-only uses no vision → skip loading video frames.
        _no_video = getattr(cfg.policy, "use_dino_features", False) or getattr(cfg.policy, "state_only", False)
        video_keys_to_load = [] if _no_video else None
        if not cfg.dataset.streaming:
            # Stage-2 SkillVLA adds the (jittered) skill-start image/state + skill code per item.
            dataset_cls = LeRobotDataset
            if getattr(cfg.policy, "type", None) == "skill_vla":
                from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset

                dataset_cls = SkillVLADataset
            elif getattr(cfg.policy, "type", None) == "skill_expert" and getattr(cfg.policy, "use_connector", False):
                # Stage-1 connector needs the skill's END frame (3rd + wrist image + state) per item.
                from lerobot.policies.skill_expert.dataset_skill_expert import SkillExpertDataset

                dataset_cls = SkillExpertDataset
            dataset = dataset_cls(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
                video_keys_to_load=video_keys_to_load,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    if getattr(cfg.policy, "use_dino_features", False):
        if not getattr(cfg.policy, "dino_feature_dir", None):
            raise ValueError("policy.dino_feature_dir is required when policy.use_dino_features=true")
        dataset = DinoFrameFeatureDataset(
            dataset,
            feature_dir=cfg.policy.dino_feature_dir,
            image_keys=list(cfg.policy.dino_image_keys),
            output_key=cfg.policy.dino_token_key,
            observation_delta_indices=cfg.policy.observation_delta_indices,
            cache_size=cfg.policy.dino_cache_size,
        )

    if getattr(cfg.policy, "skill_decoder_dino_tokens_path", None):
        dataset = SkillVLADinoTokenDataset(
            dataset,
            tokens_path=cfg.policy.skill_decoder_dino_tokens_path,
            output_key=cfg.policy.skill_decoder_dino_output_key,
            cache_path=cfg.policy.skill_decoder_dino_cache_path,
            build_cache=cfg.policy.skill_decoder_dino_build_cache,
        )

    # Stage-1 dual terminator (FSQ terminator_use_wrist=True): attach the wrist-camera tokens too.
    if getattr(cfg.policy, "skill_decoder_dino_wrist_tokens_path", None):
        dataset = SkillVLADinoTokenDataset(
            dataset,
            tokens_path=cfg.policy.skill_decoder_dino_wrist_tokens_path,
            output_key=cfg.policy.skill_decoder_dino_wrist_output_key,
            cache_path=cfg.policy.skill_decoder_dino_wrist_cache_path,
            build_cache=getattr(cfg.policy, "skill_decoder_dino_build_cache", True),
        )

    return dataset
