import numpy as np

from abcdl.mcap.reader import read_mcap


def test_read_mcap_handles_mismatched_stream_lengths(tmp_path):
    """State and action streams are polled on independent clocks → different lengths.

    read_mcap must align every stream onto the left-arm-state clock using each
    stream's own timestamps, never indexing one stream with another's indices.
    Regression for the IndexError on raw ABC-130k episodes (unequal stream lengths).
    """
    from google.protobuf.timestamp_pb2 import Timestamp
    from mcap_protobuf.writer import Writer

    from abcdl.mcap import schemas

    path = str(tmp_path / "mismatch.mcap")

    def ts(ns: int) -> Timestamp:
        t = Timestamp()
        t.FromNanoseconds(int(ns))
        return t

    # Deliberately different per-stream counts; left-arm-state (6) is the reference.
    arm_counts = {"/left-arm-state": 6, "/left-arm-action": 4,
                  "/right-arm-state": 5, "/right-arm-action": 3}
    ee_counts = {"/left-ee-state": 6, "/left-ee-action": 4,
                 "/right-ee-state": 5, "/right-ee-action": 3}
    with open(path, "wb") as f, Writer(f) as w:
        for topic, n in arm_counts.items():
            for i in range(n):
                rs = schemas.RobotState(timestamp=ts(i * 1_000_000))
                rs.position.extend([float(i)] * 6)
                w.write_message(topic=topic, message=rs, log_time=i * 1_000_000,
                                publish_time=i * 1_000_000)
        for topic, n in ee_counts.items():
            for i in range(n):
                gs = schemas.GripperState(timestamp=ts(i * 1_000_000))
                gs.position.append(0.5)
                w.write_message(topic=topic, message=gs, log_time=i * 1_000_000,
                                publish_time=i * 1_000_000)
        ins = schemas.Instructions(timestamp=ts(0), data="t")
        w.write_message(topic="/instruction", message=ins, log_time=0, publish_time=0)

    ep = read_mcap(path)  # must not raise IndexError
    assert ep.num_steps == 6  # reference = left-arm-state count
    assert ep.states.shape == (6, 14)
    assert ep.actions.shape == (6, 14)
    np.testing.assert_array_equal(ep.timestamps, np.arange(6, dtype=np.int64) * 1_000_000)


def test_read_sample_episode(sample_mcap):
    ep = read_mcap(str(sample_mcap))
    assert ep.meta.task == "open the umbrella"
    assert ep.states.shape[1] == 14 and ep.actions.shape[1] == 14
    assert ep.num_steps > 100
    assert set(ep.cameras) >= {"top", "left_wrist", "right_wrist"}
    # cameras carry encoded h264 chunks (bytes) at native timing
    top = ep.cameras["top"]
    assert top.codec == "h264" and isinstance(top.frames[0], (bytes, bytearray))
    assert ep.meta.operator_id  # metadata-drift tolerant read populated this
    assert ep.ee_poses is not None and ep.ee_poses["left"].shape[1:] == (4, 4)
