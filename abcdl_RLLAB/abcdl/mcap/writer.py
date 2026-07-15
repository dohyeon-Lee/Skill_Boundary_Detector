"""Write an Episode to an ABC-130k-compatible episode.mcap."""

from __future__ import annotations

import numpy as np
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap_protobuf.writer import Writer

from abcdl.episode import Episode, StateLayout
from abcdl.mcap import schemas

# Camera name → MCAP topic (must match the reverse map in reader.py).
# reader._CAM_TOPICS = {"top": "/top-camera", "left_wrist": "/left-wrist-camera", ...}
_CAM_NAME_TO_TOPIC: dict[str, str] = {
    "top": "/top-camera",
    "top_left": "/top-left-camera",
    "top_right": "/top-right-camera",
    "left_wrist": "/left-wrist-camera",
    "right_wrist": "/right-wrist-camera",
}


def _ts(ns: int):
    """Return a google.protobuf.Timestamp from an integer nanosecond value."""
    from google.protobuf.timestamp_pb2 import Timestamp
    t = Timestamp()
    t.FromNanoseconds(int(ns))
    return t


def write_mcap(episode: Episode, path: str, layout: StateLayout = StateLayout.YAM) -> None:
    """Write *episode* to *path* as an ABC-130k-compatible episode.mcap.

    Per-timestep topics emitted:
      /{side}-arm-{state,action}  →  RobotState  (6 arm joints; EE pose on state when available)
      /{side}-ee-{state,action}   →  GripperState (1 gripper value)

    Once per file:
      /instruction                →  Instructions
      episode-metadata record     →  session_id, operator_id, task_name

    Per-frame camera topics follow the naming in reader._CAM_TOPICS so that a
    subsequent read_mcap() maps them back to the same camera names.
    """
    episode.validate()
    arm = layout.arm_dof      # 6 for YAM
    grip = layout.gripper_dof  # 1 for YAM
    sides = ["left", "right"]
    ts = np.asarray(episode.timestamps, np.int64)

    with open(path, "wb") as f, Writer(f) as w:
        for i in range(episode.num_steps):
            t_ns = int(ts[i])
            timestamp = _ts(t_ns)

            for s, side in enumerate(sides):
                base = s * (arm + grip)

                for kind, vec in (("state", episode.states[i]), ("action", episode.actions[i])):
                    # --- arm joints (RobotState) ---
                    rs = schemas.RobotState(timestamp=timestamp)
                    rs.position.extend(vec[base:base + arm].tolist())
                    if (kind == "state"
                            and episode.ee_poses is not None
                            and side in episode.ee_poses):
                        # ee_poses[side] is (T, 4, 4); flatten to 16 floats.
                        rs.pose.extend(
                            np.asarray(episode.ee_poses[side][i]).reshape(-1).tolist()
                        )
                    w.write_message(
                        topic=f"/{side}-arm-{kind}",
                        message=rs,
                        log_time=t_ns,
                        publish_time=t_ns,
                    )

                    # --- gripper (GripperState) ---
                    gs = schemas.GripperState(timestamp=timestamp)
                    gs.position.append(float(vec[base + arm]))
                    w.write_message(
                        topic=f"/{side}-ee-{kind}",
                        message=gs,
                        log_time=t_ns,
                        publish_time=t_ns,
                    )

        # --- instruction (once) ---
        ins = schemas.Instructions(timestamp=_ts(int(ts[0])), data=episode.meta.task)
        w.write_message(
            topic="/instruction",
            message=ins,
            log_time=int(ts[0]),
            publish_time=int(ts[0]),
        )

        # --- camera frames ---
        for name in episode.meta.cameras:
            cam = episode.cameras[name]
            topic = _CAM_NAME_TO_TOPIC.get(name)
            if topic is None:
                # Fallback for unknown names: best-effort topic derivation.
                if "wrist" in name:
                    topic = f"/{name.replace('_', '-')}-camera"
                else:
                    topic = f"/{name}-camera"
            frames, frame_ts, eff_codec = _encoded_frames(cam)
            for j, chunk in enumerate(frames):
                cv = CompressedVideo(
                    data=chunk,
                    format=eff_codec,
                    frame_id=f"{name}-images-rgb",
                )
                cv.timestamp.FromNanoseconds(int(frame_ts[j]))
                w.write_message(
                    topic=topic,
                    message=cv,
                    log_time=int(frame_ts[j]),
                    publish_time=int(frame_ts[j]),
                )

        # --- episode-metadata record (via underlying mcap.writer.Writer) ---
        if not hasattr(getattr(w, "_writer", None), "add_metadata"):
            raise RuntimeError(
                "mcap_protobuf.Writer internal API changed: no `_writer.add_metadata`. "
                "Pin mcap-protobuf-support<0.6 or update abcdl.mcap.writer."
            )
        w._writer.add_metadata(
            name="episode-metadata",
            data={
                "session_id": episode.meta.session_id or "",
                "operator_id": episode.meta.operator_id or "",
                "task_name": episode.meta.task,
            },
        )


def encode_frame_to_annexb(frame_hwc) -> bytes:
    """Encode one (H,W,3) uint8 RGB frame to an H.264 byte blob."""
    import os
    import tempfile
    import numpy as np
    from abcdl.format.encode import encode_strict_h264

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        encode_strict_h264(np.asarray(frame_hwc, np.uint8)[None], tmp)
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(tmp)


def _encoded_frames(cam):
    """Return (list[bytes] Annex-B frames, timestamps np.int64, effective_codec str).

    If frames are already bytes/bytearray, pass them through directly and
    report the camera's own codec.  Decoded (ndarray) frames are encoded
    per-frame via encode_frame_to_annexb and the effective codec is "h264".
    """
    if len(cam.frames) and isinstance(cam.frames[0], (bytes, bytearray)):
        return [bytes(x) for x in cam.frames], np.asarray(cam.timestamps, np.int64), cam.codec
    return (
        [encode_frame_to_annexb(fr) for fr in cam.frames],
        np.asarray(cam.timestamps, np.int64),
        "h264",
    )
