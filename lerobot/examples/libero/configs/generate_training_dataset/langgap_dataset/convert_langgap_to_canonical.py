#!/usr/bin/env python3
# Inputs:
#   LangGap staging : {langgap_root}/_hf/{set}   (HF LeRobot v3.0, fps-label 10, image/image2)
#   config          : ./langgap_dataset_config.yaml
#   orientation     : {staging}/.orientation/verdict.json  (flip=auto일 때)
# Reference model:
#   none
# Outputs:
#   LeRobot data    : {langgap_root}/{output_name}   (canonical v3.0, 20 Hz, image/wrist_image)
"""Rewrite a LangGap HF LeRobot dataset into the local canonical convention.

Why a full rewrite (decode+re-encode) instead of metadata patching:
  - fps 라벨 10 → 실제 20 Hz 로 바꾸려면 frame timestamp 컬럼과 비디오 컨테이너 PTS가
    함께 바뀌어야 한다(안 맞으면 timestamp 기반 프레임 조회가 어긋남) → 어차피 재작성.
  - 방향 플립이 필요하면 재인코딩이 필수.
The rewrite keeps the current SBD conventions:
  - 20 Hz episodes (LangGap collect는 control_freq=20, frame_skip=1 — 라벨만 10이었음)
  - observation.images.image  <- image   (flip per orientation verdict)
  - observation.images.wrist_image <- image2
  - state = LangGap observation.state 그대로 (eef pos3 + axis-angle3 + gripper qpos2)
  - action = LangGap OSC delta action, gripper kept in {-1, 1}
  - observation.states.* 서브 피처는 생략 (LangGap에 joint_state 없음; 학습 미사용 확인)

Examples:
  python convert_langgap_to_canonical.py --set langgap_56_full_full
  python convert_langgap_to_canonical.py --set langgap_6_smoke \
    --flip-image none --flip-wrist none --max-episodes 2 --output-name langgap_smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))
from langgap_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    langgap_root,
    load_config,
    project_root,
)

REPO_PREFIX = "dohyeon"
EMITTED_FEATURES = (
    "observation.images.image",
    "observation.images.wrist_image",
    "observation.state",
    "action",
)
FLIPS = {
    "none": lambda a: a,
    "h": lambda a: a[::-1],
    "w": lambda a: a[:, ::-1],
    "hw": lambda a: a[::-1, ::-1],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--set", required=True, help="Staging set name under {langgap_root}/_hf")
    parser.add_argument("--output-name", default=None, help="Output folder name. Default: {set}")
    parser.add_argument("--flip-image", choices=[*FLIPS, "auto"], default=None,
                        help="Agentview flip. Default: yaml convert_flip_image")
    parser.add_argument("--flip-wrist", choices=[*FLIPS, "auto"], default=None,
                        help="Wrist flip. Default: yaml convert_flip_wrist")
    parser.add_argument("--vcodec", default=None)
    parser.add_argument("--image-writer-threads", type=int, default=None)
    parser.add_argument("--image-writer-processes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug limit.")
    return parser.parse_args()


def resolve_flip(cli: str | None, cfg_value: str, staging: Path, camera: str) -> str:
    mode = (cli or str(cfg_value or "auto")).strip().lower()
    if mode != "auto":
        return mode
    verdict_path = staging / ".orientation" / "verdict.json"
    if not verdict_path.exists():
        raise SystemExit(
            f"flip={camera}:auto 인데 {verdict_path} 가 없음 — "
            "verify_image_orientation.py 를 먼저 실행하거나 flip을 명시하세요."
        )
    verdict = json.loads(verdict_path.read_text())
    value = str(verdict.get(camera, "unknown"))
    if value not in FLIPS:
        raise SystemExit(
            f"orientation verdict for {camera!r} = {value!r} (not confident). "
            f"{verdict_path.parent}/*_compare.png 를 눈으로 확인한 뒤 yaml convert_flip_* "
            "또는 --flip-* 로 명시하세요."
        )
    return value


def reference_feature_specs(reference_root: Path) -> dict[str, Any]:
    """Load the exact feature contract (emitted keys only) from the canonical dataset."""
    info_path = reference_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Schema reference is missing meta/info.json: {info_path}. "
            "Set convert_schema_reference to an existing canonical LeRobot dataset."
        )
    features = json.loads(info_path.read_text()).get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Schema reference has no valid features mapping: {info_path}")
    selected = {key: value for key, value in features.items() if key in EMITTED_FEATURES}
    missing = set(EMITTED_FEATURES) - set(selected)
    if missing:
        raise ValueError(f"Schema reference is missing required features: {sorted(missing)}")
    for key in ("observation.images.image", "observation.images.wrist_image"):
        if selected[key].get("dtype") != "video":
            raise ValueError(
                f"Schema reference must be video-backed, but {key} has "
                f"dtype={selected[key].get('dtype')!r}: {info_path}"
            )
    return selected


def normalize_feature_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    if isinstance(out.get("shape"), list):
        out["shape"] = tuple(out["shape"])
    if isinstance(out.get("info"), dict):
        out["info"] = dict(out["info"])
    return out


def codec_metadata_name(vcodec: str) -> str:
    if vcodec == "libsvtav1":
        return "av1"
    if vcodec.startswith("h264"):
        return "h264"
    if vcodec.startswith("hevc"):
        return "hevc"
    return vcodec


def validated_reference_features(
    base_features: dict[str, Any], image_size: int, resolved_vcodec: str, fps: int
) -> dict[str, Any]:
    features = {key: normalize_feature_spec(value) for key, value in base_features.items()}
    expected_codec = codec_metadata_name(resolved_vcodec)
    for key in ("observation.images.image", "observation.images.wrist_image"):
        feature = features[key]
        shape = tuple(feature.get("shape", ()))
        info = feature.get("info") or {}
        if shape != (image_size, image_size, 3):
            raise ValueError(f"Configured image size {image_size} does not match {key} reference shape {shape}.")
        if info.get("video.codec") != expected_codec:
            raise ValueError(
                f"Configured codec {expected_codec!r} does not match {key} reference codec "
                f"{info.get('video.codec')!r}."
            )
        if info.get("video.fps") != fps:
            raise ValueError(f"Reference {key} fps={info.get('video.fps')!r}, expected {fps}.")
    return features


def normalized_action(action: np.ndarray) -> np.ndarray:
    out = action.astype(np.float32, copy=True)
    grip = out[..., 6]
    unique = np.unique(np.round(grip, decimals=6))
    if set(unique.tolist()).issubset({0.0, 1.0}):
        out[..., 6] = np.where(grip > 0.5, 1.0, -1.0)
    elif set(unique.tolist()).issubset({-1.0, 1.0}):
        out[..., 6] = np.where(grip > 0.0, 1.0, -1.0)
    return out


def to_uint8_hwc(chw) -> np.ndarray:
    arr = (chw.permute(1, 2, 0).numpy() * 255.0).round()
    return np.clip(arr, 0, 255).astype(np.uint8)


def validate_task_order(staging: Path, output_dir: Path) -> None:
    """출력의 task_index 순서가 소스와 동일한지 검증.

    build_training_dataset.py 의 --task-range 는 연속 task_index 구간으로 서브셋을 만들므로
    (예: langgap_56 에서 확장 16태스크 = 40to55), 변환이 소스의 task_index 배치를 바꾸면
    하류 분할이 조용히 다른 태스크를 집게 된다. 순서가 다르면 여기서 시끄럽게 실패한다."""
    import pandas as pd

    src = pd.read_parquet(staging / "meta" / "tasks.parquet").sort_values("task_index")
    out = pd.read_parquet(output_dir / "meta" / "tasks.parquet").sort_values("task_index")
    src_tasks = list(src.index)
    out_tasks = list(out.index)
    # 디버그 리밋(--max-episodes)이면 출력이 소스의 prefix 만 커버할 수 있다.
    if out_tasks != src_tasks[: len(out_tasks)]:
        mismatch = next(i for i, (a, b) in enumerate(zip(out_tasks, src_tasks)) if a != b)
        raise RuntimeError(
            "Converted task_index order differs from source — downstream --task-range "
            f"splits would select wrong tasks. First mismatch at task_index {mismatch}: "
            f"output={out_tasks[mismatch]!r} vs source={src_tasks[mismatch]!r}"
        )
    print(f"  task order     : preserved ({len(out_tasks)}/{len(src_tasks)} tasks, "
          f"task_index 00..{len(out_tasks) - 1:02d} identical to source)")


def validate_written_dataset(output_dir: Path, expected_features: dict[str, Any], fps: int) -> None:
    info_path = output_dir / "meta" / "info.json"
    stats_path = output_dir / "meta" / "stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        raise RuntimeError(f"Converted dataset is missing v3 metadata under {output_dir / 'meta'}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(f"Expected LeRobot codebase_version='v3.0', got {info.get('codebase_version')!r}")
    if int(info.get("fps", 0)) != fps:
        raise RuntimeError(f"Converted fps={info.get('fps')!r}, expected {fps}")
    actual_features = info.get("features", {})
    for key, expected in expected_features.items():
        if key not in actual_features:
            raise RuntimeError(f"Converted v3 metadata is missing feature {key!r}")
        actual = normalize_feature_spec(actual_features[key])
        for field in ("dtype", "shape", "names"):
            if actual.get(field) != expected.get(field):
                raise RuntimeError(
                    f"Converted feature {key!r} field {field!r} does not match schema reference: "
                    f"actual={actual.get(field)!r}, expected={expected.get(field)!r}"
                )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    proot = project_root(cfg)
    root = langgap_root(cfg)
    staging = root / "_hf" / args.set
    output_name = args.output_name or args.set
    output_dir = root / output_name
    fps = int(cfg.get("convert_fps", 20))
    image_size = int(cfg.get("convert_image_size", 256))
    vcodec = args.vcodec or str(cfg.get("convert_vcodec", "libsvtav1"))

    if not (staging / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Staging set not found (download first): {staging}")
    if output_dir.exists():
        if not (args.overwrite or bool(cfg.get("convert_overwrite", False))):
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    flip_image = resolve_flip(args.flip_image, cfg.get("convert_flip_image", "auto"), staging, "image")
    flip_wrist = resolve_flip(args.flip_wrist, cfg.get("convert_flip_wrist", "auto"), staging, "wrist")

    schema_reference = Path(str(cfg.get("convert_schema_reference", ""))).expanduser()
    if not schema_reference.is_absolute():
        schema_reference = proot / schema_reference

    lerobot_src = proot / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_episode_video_frames, resolve_vcodec

    src_info = json.loads((staging / "meta" / "info.json").read_text())

    print("Convert LangGap LeRobot to canonical")
    print(f"  set          : {args.set}")
    print(f"  source       : {staging} (episodes={src_info.get('total_episodes')}, "
          f"fps-label={src_info.get('fps')}, real 20 Hz)")
    print(f"  output       : {output_dir}")
    print(f"  flip         : image={flip_image} wrist={flip_wrist}")
    print(f"  schema ref   : {schema_reference}")

    base_features = reference_feature_specs(schema_reference)
    resolved_vcodec = resolve_vcodec(vcodec)
    print(f"  codec        : requested={vcodec} resolved={resolved_vcodec}")
    features = validated_reference_features(base_features, image_size, resolved_vcodec, fps)

    dataset = LeRobotDataset.create(
        repo_id=f"{REPO_PREFIX}/{output_name}",
        fps=fps,
        root=output_dir,
        robot_type="franka",
        features=features,
        image_writer_threads=args.image_writer_threads or int(cfg.get("convert_image_writer_threads", 10)),
        image_writer_processes=args.image_writer_processes or int(cfg.get("convert_image_writer_processes", 5)),
        vcodec=resolved_vcodec,
    )

    # ── Fast source reading ─────────────────────────────────────────
    # State/action/task are read directly from parquet. Each camera is still decoded only once per
    # episode, but every output frame is selected by its encoded timestamp. A raw read_video range
    # can include the preceding episode's final frame at float/seek boundaries.
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    src_fps = float(src_info.get("fps", 10))
    data_df = pd.concat(
        [pd.read_parquet(p) for p in sorted((staging / "data").glob("**/*.parquet"))],
        ignore_index=True,
    ).sort_values(["episode_index", "frame_index"])
    episodes_df = pd.concat(
        [pd.read_parquet(p) for p in sorted((staging / "meta" / "episodes").glob("**/*.parquet"))],
        ignore_index=True,
    ).sort_values("episode_index")
    tasks_df = pd.read_parquet(staging / "meta" / "tasks.parquet")
    task_by_index = {int(row.task_index): name for name, row in tasks_df.iterrows()}
    frames_by_ep = {int(ep): g for ep, g in data_df.groupby("episode_index")}

    ep_rows = episodes_df.to_dict("records")
    if args.max_episodes is not None:
        ep_rows = ep_rows[: args.max_episodes]

    SRC_CAMERAS = ("observation.images.image", "observation.images.image2")

    def decode_episode(row) -> dict[str, np.ndarray]:
        """Decode both packed-video slices by exact source timestamps (THWC uint8)."""
        length = int(row["length"])
        out = {}
        for src_key in SRC_CAMERAS:
            chunk = int(row[f"videos/{src_key}/chunk_index"])
            file = int(row[f"videos/{src_key}/file_index"])
            from_ts = float(row[f"videos/{src_key}/from_timestamp"])
            to_ts = float(row[f"videos/{src_key}/to_timestamp"])
            path = staging / "videos" / src_key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
            frames = decode_episode_video_frames(
                path,
                from_ts,
                to_ts,
                length,
                src_fps,
                backend="pyav",
                # Offline conversion is not subject to the persistent
                # DataLoader-worker AV1 deadlock, so use the decoder's native
                # thread pool here.
                decoder_num_threads=None,
            )
            out[src_key] = (
                (frames.permute(0, 2, 3, 1) * 255.0)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .numpy()
            )
        return out

    flip_img_fn = FLIPS[flip_image]
    flip_wri_fn = FLIPS[flip_wrist]
    total_frames = 0
    total_episodes = 0
    # 프리페치: 다음 에피소드 디코드를 백그라운드에서 미리 수행해 인코딩과 겹친다.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(decode_episode, row) for row in ep_rows[:2]]
        for i, row in enumerate(ep_rows):
            videos = futures[i].result()
            if i + 2 < len(ep_rows):
                futures.append(pool.submit(decode_episode, ep_rows[i + 2]))

            ep = int(row["episode_index"])
            group = frames_by_ep[ep]
            length = int(row["length"])
            if len(group) != length:
                raise ValueError(f"episode {ep}: parquet rows {len(group)} != episode length {length}")
            states = np.stack(group["observation.state"].to_numpy()).astype(np.float32)
            actions = normalized_action(np.stack(group["action"].to_numpy()))
            task = task_by_index[int(group["task_index"].iloc[0])]
            images = videos["observation.images.image"]
            wrists = videos["observation.images.image2"]

            for t in range(length):
                dataset.add_frame(
                    {
                        "observation.images.image": flip_img_fn(images[t]).copy(),
                        "observation.images.wrist_image": flip_wri_fn(wrists[t]).copy(),
                        "observation.state": states[t],
                        "action": actions[t],
                        "task": task,
                    }
                )
            dataset.save_episode()
            total_frames += length
            total_episodes += 1
            if total_episodes % 50 == 0:
                print(f"    saved episodes={total_episodes}/{len(ep_rows)} frames={total_frames}")

    dataset.finalize()
    validate_written_dataset(output_dir, features, fps)
    validate_task_order(staging, output_dir)
    print("")
    print("DONE")
    print("  LeRobot        : v3.0 (schema reference verified, 20 Hz)")
    print(f"  output         : {output_dir}")
    print(f"  total episodes : {total_episodes}")
    print(f"  total frames   : {total_frames}")


if __name__ == "__main__":
    main()
