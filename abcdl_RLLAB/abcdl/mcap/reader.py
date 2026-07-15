"""Read an ABC-130k episode.mcap into the Episode IR (embedded-schema decode)."""

from __future__ import annotations

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from abcdl.episode import CameraStream, Episode, EpisodeMeta, StateLayout

# Camera topics present in ABC-130k episodes (discover at read-time; these are candidates).
_CAM_TOPICS = {
    "top": "/top-camera",
    "top_left": "/top-left-camera",
    "top_right": "/top-right-camera",
    "left_wrist": "/left-wrist-camera",
    "right_wrist": "/right-wrist-camera",
}
# Reverse map: topic → name
_TOPIC_TO_CAM = {v: k for k, v in _CAM_TOPICS.items()}


def ch_time(dec) -> int:
    """Return absolute nanosecond timestamp from a decoded protobuf message."""
    return int(dec.timestamp.seconds) * 1_000_000_000 + int(dec.timestamp.nanos)


def _floor_idx(src_ts: np.ndarray, tgt_ts: np.ndarray) -> np.ndarray:
    """For each element of tgt_ts, return the index of the latest src_ts at-or-before it."""
    return np.clip(np.searchsorted(src_ts, tgt_ts, side="right") - 1, 0, len(src_ts) - 1)


def _read_metadata(reader) -> tuple[str | None, str | None, str | None]:
    """Return (session_id, operator_id, task) tolerant of the two known naming variants.

    Variant A (episode-metadata): keys session_id, operator_id, task_name.
    Variant B (session-metadata):  keys session-uuid, operator-id, instruction.
    """
    for m in reader.iter_metadata():
        md = dict(m.metadata)
        if m.name in ("episode-metadata", "session-metadata"):
            return (
                md.get("session_id") or md.get("session-uuid"),
                md.get("operator_id") or md.get("operator-id"),
                md.get("task_name") or md.get("instruction"),
            )
    return None, None, None


def read_mcap(path: str, layout: StateLayout = StateLayout.YAM) -> Episode:
    """Read *path* (an ABC-130k episode.mcap) into an :class:`~abcdl.episode.Episode`.

    State/action layout follows *layout* (default YAM):
    ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`` → 14-D.

    Gripper values are floor-sampled onto the arm clock (latest gripper sample
    at-or-before each arm timestamp).  Right-arm streams are aligned onto the
    left-arm-state clock via the same floor-index method.

    Camera frames are kept as encoded Annex-B ``bytes`` at their native
    nanosecond timestamps (no decoding or resampling here).

    EE poses (``RobotState.pose``, length 16) are reshaped to ``(T,4,4)`` and
    aligned to the left-state clock.
    """
    # Pass 1: read metadata (needs a separate file handle so we can rewind).
    with open(path, "rb") as f:
        sid, oid, mtask = _read_metadata(make_reader(f))

    # Buffers: keyed by (side, kind).
    arms: dict[tuple[str, str], list[tuple[int, list[float]]]] = {}
    grips: dict[tuple[str, str], list[tuple[int, float]]] = {}
    poses: dict[str, list[tuple[int, list[float]]]] = {}
    cams: dict[str, list[tuple[int, bytes, str]]] = {}
    task: str | None = None

    # Pass 2: decode all messages.
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, _, dec in reader.iter_decoded_messages():
            t = ch.topic
            if t == "/instruction":
                task = dec.data
            elif t.endswith("-arm-state") or t.endswith("-arm-action"):
                side = "left" if "left" in t else "right"
                kind = "state" if t.endswith("state") else "action"
                ts = ch_time(dec)
                arms.setdefault((side, kind), []).append((ts, list(dec.position)))
                if kind == "state" and len(dec.pose) == 16:
                    poses.setdefault(side, []).append((ts, list(dec.pose)))
            elif t.endswith("-ee-state") or t.endswith("-ee-action"):
                side = "left" if "left" in t else "right"
                kind = "state" if t.endswith("state") else "action"
                grips.setdefault((side, kind), []).append((ch_time(dec), float(dec.position[0])))
            elif t in _TOPIC_TO_CAM:
                name = _TOPIC_TO_CAM[t]
                cams.setdefault(name, []).append((ch_time(dec), bytes(dec.data), dec.format))

    # ---------------------------------------------------------------------------
    # Build per-(side, kind) arrays: arm joints concatenated with floor-sampled
    # gripper, all on the arm-state clock.
    # ---------------------------------------------------------------------------
    def stack(side: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, array) of shape (T, 7) for *side*/*kind*."""
        if (side, kind) not in arms:
            raise ValueError(f"MCAP missing required topic /{side}-arm-{kind}")
        if (side, kind) not in grips:
            raise ValueError(f"MCAP missing required topic /{side}-ee-{kind}")
        a = sorted(arms[(side, kind)])
        g = sorted(grips[(side, kind)])
        a_ts = np.array([x[0] for x in a], np.int64)
        a_v = np.array([x[1] for x in a], np.float64)          # (Ta, 6)
        g_ts = np.array([x[0] for x in g], np.int64)
        g_v = np.array([x[1] for x in g], np.float64)[:, None]  # (Tg, 1)
        g_on_a = g_v[_floor_idx(g_ts, a_ts)]
        return a_ts, np.concatenate([a_v, g_on_a], axis=1)      # (Ta, 7)

    lt, lstate = stack("left", "state")
    la_ts, laction_v = stack("left", "action")
    rs_ts, rstate_v = stack("right", "state")
    ra_ts, raction_v = stack("right", "action")

    # The four streams (left/right x state/action) are polled on independent clocks
    # and therefore have different lengths. Align every stream onto the left-arm-state
    # clock (lt) using each stream's OWN timestamps — never index one stream with
    # another stream's indices.
    laction = laction_v[_floor_idx(la_ts, lt)]
    rstate = rstate_v[_floor_idx(rs_ts, lt)]
    raction = raction_v[_floor_idx(ra_ts, lt)]
    states = np.concatenate([lstate, rstate], axis=1)    # (T, 14)
    actions = np.concatenate([laction, raction], axis=1)  # (T, 14)

    # ---------------------------------------------------------------------------
    # EE poses: reshape flat 16-D pose to (4,4) and align to left-state clock.
    # ---------------------------------------------------------------------------
    ee: dict[str, np.ndarray] = {}
    for side, lst in poses.items():
        lst_s = sorted(lst)
        p_ts = np.array([x[0] for x in lst_s], np.int64)
        p = np.array([x[1] for x in lst_s], np.float64).reshape(-1, 4, 4)
        ee[side] = p[_floor_idx(p_ts, lt)]

    # ---------------------------------------------------------------------------
    # Camera streams: sorted by timestamp, frames kept as raw bytes.
    # ---------------------------------------------------------------------------
    # Build canonical camera order: known cameras in _CAM_TOPICS key order, then unknowns sorted.
    _canonical_order = list(_CAM_TOPICS.keys())
    _known = [n for n in _canonical_order if n in cams]
    _unknown = sorted(n for n in cams if n not in _CAM_TOPICS)
    cam_names: list[str] = _known + _unknown

    cam_streams: dict[str, CameraStream] = {}
    cam_res: dict[str, tuple[int, int]] = {}
    cam_codecs: dict[str, str] = {}
    for name in cam_names:
        lst = cams[name]
        lst_s = sorted(lst, key=lambda x: x[0])
        ts = np.array([x[0] for x in lst_s], np.int64)
        frames = [x[1] for x in lst_s]
        codec = lst_s[0][2]
        cam_streams[name] = CameraStream(frames=frames, timestamps=ts, width=0, height=0, codec=codec)
        cam_res[name] = (0, 0)
        cam_codecs[name] = codec

    meta = EpisodeMeta(
        task=task or mtask or "",
        fps=0.0,
        cameras=cam_names,
        camera_resolutions=cam_res,
        camera_codecs=cam_codecs,
        operator_id=oid,
        alignment="native",
        session_id=sid,
    )
    return Episode(
        states=states,
        actions=actions,
        timestamps=lt,
        cameras=cam_streams,
        meta=meta,
        ee_poses=ee or None,
    )
