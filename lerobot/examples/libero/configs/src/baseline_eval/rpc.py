"""Length-prefixed messages over private parent/child pipes (never a network service).

Pickle is only used between our own two processes, not for user-supplied files.
Numpy arrays retain their dtype/shape without JSON or image compression losses.
"""
import pickle
import select
import struct
import time

import numpy as np

MAX_MESSAGE = 128 * 1024 * 1024


def _pack(value):
    # Different venvs can use NumPy 1.x/2.x. Do not pickle NumPy's module names.
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not valid inference tensors")
        return ("__baseline_ndarray__", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pack(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unpack(value):
    if isinstance(value, tuple) and len(value) == 4 and value[0] == "__baseline_ndarray__":
        return np.frombuffer(value[3], dtype=value[1]).reshape(value[2]).copy()
    if isinstance(value, dict):
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unpack(item) for item in value)
    return value


def write_message(stream, value):
    data = pickle.dumps(_pack(value), protocol=5)
    if len(data) > MAX_MESSAGE:
        raise ValueError("Inference message exceeds 128 MiB")
    # FileIO.write is allowed to return a short write on a large pipe payload.
    packet = memoryview(struct.pack("!Q", len(data)) + data)
    while packet:
        written = stream.write(packet)
        if not written:
            raise BrokenPipeError("Inference worker pipe closed")
        packet = packet[written:]
    stream.flush()


def read_message(stream, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout

    def exact(count):
        data = bytearray()
        while len(data) < count:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                    raise TimeoutError("Inference worker timed out; see the job stderr")
            chunk = stream.read(count - len(data))
            if not chunk:
                raise EOFError("Inference worker exited; see its traceback in the job stderr")
            data.extend(chunk)
        return data

    size = struct.unpack("!Q", exact(8))[0]
    if size > MAX_MESSAGE:
        raise ValueError("Invalid/oversized inference message")
    return _unpack(pickle.loads(exact(size)))
