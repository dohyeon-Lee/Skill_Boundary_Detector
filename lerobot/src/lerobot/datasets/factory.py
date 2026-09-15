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
            # The auxiliary recurrent terminator needs only a proprio history.
            # Keep camera observations at the current frame so it can still be
            # co-trained with any visual auxiliary without decoding T videos.
            if (
                getattr(cfg, "type", None) == "skill_aux"
                and getattr(cfg, "train_state_rnn_terminator", False)
                and key != OBS_STATE
            ):
                continue
            # DINO / state-only modes condition on observation.state only — skip windowing other obs.
            vision_off = getattr(cfg, "use_dino_features", False) or getattr(cfg, "state_only", False)
            if vision_off and key != OBS_STATE:
                continue
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _foveated_vision_config(policy: PreTrainedConfig) -> dict:
    """Translate the checkpoint's flat config into the dataset transform."""
    return {
        "enabled": bool(getattr(policy, "foveated_vision_enabled", False)),
        "randomization_enabled": bool(
            getattr(policy, "foveation_randomization_enabled", False)
        ),
        "mode": str(getattr(policy, "foveation_mode", "partial_fov")),
        "crop_size": int(getattr(policy, "foveation_crop_size", 128)),
        "output_size": int(getattr(policy, "foveation_output_size", 224)),
        "inner_box_enabled": bool(
            getattr(policy, "foveation_inner_box_enabled", True)
        ),
        "inner_box_mode": str(
            getattr(policy, "foveation_inner_box_mode", "blur")
        ),
        "inner_box_size": int(getattr(policy, "foveation_inner_box_size", 32)),
        "inner_box_line_width": int(
            getattr(policy, "foveation_inner_box_line_width", 3)
        ),
        "shape": str(getattr(policy, "foveation_shape", "square")),
        "sharp_size": int(getattr(policy, "foveation_sharp_size", 96)),
        "feather": int(getattr(policy, "foveation_feather", 20)),
        "peripheral_blur_radius": float(
            getattr(policy, "foveation_peripheral_blur_radius", 8.0)
        ),
        "color_enabled": bool(
            getattr(policy, "foveation_color_enabled", False)
        ),
        "brightness": (
            float(getattr(policy, "foveation_brightness_min", 0.8)),
            float(getattr(policy, "foveation_brightness_max", 1.2)),
        ),
        "contrast": (
            float(getattr(policy, "foveation_contrast_min", 0.8)),
            float(getattr(policy, "foveation_contrast_max", 1.2)),
        ),
        "saturation": (
            float(getattr(policy, "foveation_saturation_min", 0.8)),
            float(getattr(policy, "foveation_saturation_max", 1.2)),
        ),
        "hue": (
            float(getattr(policy, "foveation_hue_min", -0.15)),
            float(getattr(policy, "foveation_hue_max", 0.15)),
        ),
        "crop_enabled": bool(getattr(policy, "foveation_crop_enabled", False)),
        "crop_offset_px": (
            int(getattr(policy, "foveation_crop_offset_min_px", -24)),
            int(getattr(policy, "foveation_crop_offset_max_px", 24)),
        ),
        "inner_box_offset_px": (
            int(getattr(policy, "foveation_inner_box_offset_min_px", -4)),
            int(getattr(policy, "foveation_inner_box_offset_max_px", 4)),
        ),
        "input_blur_enabled": bool(
            getattr(policy, "foveation_input_blur_enabled", False)
        ),
        "input_blur_radius": (
            float(getattr(policy, "foveation_input_blur_min_radius", 0.0)),
            float(getattr(policy, "foveation_input_blur_max_radius", 4.0)),
        ),
    }


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
        _no_video = (
            getattr(cfg.policy, "use_dino_features", False)
            or getattr(cfg.policy, "state_only", False)
            or getattr(cfg.policy, "state_only_auxiliary", False)
            # Predictor-only auxiliary items query their jittered transition
            # top/wrist frames explicitly as skill_start_* below. Loading the
            # arbitrary base-row camera pair as well only decodes duplicates.
            or getattr(cfg.policy, "predictor_transition_sampling", False)
        )
        video_keys_to_load = [] if _no_video else None
        if not cfg.dataset.streaming:
            # Skill policies add the (jittered) skill-start image/state + skill code per item.
            dataset_cls = LeRobotDataset
            policy_type = getattr(cfg.policy, "type", None)
            if policy_type in {"skill_aux", "skill_expert", "skill_vla_stage2"}:
                if policy_type == "skill_aux":
                    needs_predictor_start = bool(
                        getattr(cfg.policy, "train_skill_predictor", False)
                    )
                    needs_previous_action = bool(
                        getattr(cfg.policy, "train_terminator", False)
                    )
                    if needs_predictor_start or needs_previous_action:
                        from functools import partial

                        from lerobot.policies.skillVLA.dataset_skillVLA import (
                            SkillVLADataset,
                        )

                        dataset_cls = partial(
                            SkillVLADataset,
                            include_predictor_start_inputs=needs_predictor_start,
                        )
                    else:
                        # Legacy state-only auxiliaries need neither extra field.
                        dataset_cls = LeRobotDataset
                else:
                    from functools import partial

                    from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset

                    dataset_cls = (
                        partial(
                            SkillVLADataset,
                            include_canonical_skill_actions=bool(
                                (
                                    getattr(cfg.policy, "skill_flow_enabled", False)
                                    and getattr(
                                        cfg.policy,
                                        "skill_flow_target",
                                        "canonical",
                                    )
                                    == "canonical"
                                )
                                or (
                                    policy_type == "skill_vla_stage2"
                                    and getattr(
                                        cfg.policy,
                                        "dsbc_latent_predictor_enabled",
                                        False,
                                    )
                                    and getattr(
                                        cfg.policy,
                                        "dsbc_latent_supervision",
                                        "main_chunk",
                                    )
                                    == "skill_only"
                                )
                            ),
                            canonical_skill_action_max_length=int(
                                getattr(cfg.policy, "skill_flow_max_length", 0)
                            )
                            or None,
                            jitter_pmax=int(
                                getattr(cfg.policy, "transition_jitter_pmax", 0)
                            ),
                            jitter_early_start_pmax=int(
                                getattr(
                                    cfg.policy,
                                    "transition_jitter_early_start_pmax",
                                    -1,
                                )
                            ),
                            jitter_late_start_pmax=int(
                                getattr(
                                    cfg.policy,
                                    "transition_jitter_late_start_pmax",
                                    -1,
                                )
                            ),
                            jitter_early_end_pmax=int(
                                getattr(
                                    cfg.policy,
                                    "transition_jitter_early_end_pmax",
                                    -1,
                                )
                            ),
                            jitter_late_end_pmax=int(
                                getattr(
                                    cfg.policy,
                                    "transition_jitter_late_end_pmax",
                                    -1,
                                )
                            ),
                            foveated_vision_config=_foveated_vision_config(
                                cfg.policy
                            ),
                        )
                        if policy_type in {"skill_expert", "skill_vla_stage2"}
                        else SkillVLADataset
                    )
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
            if (
                isinstance(dataset, LeRobotDataset)
                and getattr(cfg.policy, "type", None) == "skill_aux"
                and getattr(cfg.policy, "train_state_rnn_terminator", False)
            ):
                dataset.cache_delta_columns([OBS_STATE])
                logging.info(
                    "Cached %s in RAM for auxiliary RNN history windows ",
                    OBS_STATE,
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

    # SkillVLA datasets materialize grounding during their own build stage.
    # This runtime view is specifically for raw-dataset Diffusion training;
    # applying it to another policy with the same config field would subtract
    # the episode reference twice.
    proprio_grounding = getattr(cfg.policy, "proprio_grounding", "none")
    if getattr(cfg.policy, "type", None) == "diffusion" and proprio_grounding != "none":
        if cfg.dataset.streaming:
            raise ValueError("Episode-start proprio grounding is not supported for streaming datasets.")
        if not isinstance(dataset, LeRobotDataset):
            raise TypeError(
                "Episode-start proprio grounding requires a frame-level LeRobotDataset, "
                f"got {type(dataset).__name__}."
            )
        from lerobot.datasets.proprio_grounding import (
            EpisodeStartXYZGroundedDataset,
            normalize_proprio_grounding,
        )

        proprio_grounding = normalize_proprio_grounding(proprio_grounding)
        if proprio_grounding == "episode_start_xyz":
            # Dense state windows otherwise perform hundreds of tiny HF gathers.
            dataset.cache_delta_columns([OBS_STATE])
            dataset = EpisodeStartXYZGroundedDataset(dataset)
            logging.info(
                "Applied episode-start XYZ grounding before policy normalization."
            )

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

    # (skill_decoder_dino 토큰 래퍼 은퇴 — terminator co-training은 배치의 현재 프레임 이미지를
    #  fsq._prepare_decoder_tokens의 raw 분기로 ONLINE 토큰화한다. 디스크 DINO precompute 제거.)

    return dataset
