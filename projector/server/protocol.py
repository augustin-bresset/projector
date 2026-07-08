"""JSON (de)serialization — no FastAPI dependency.

numpy arrays are encoded as `{dtype, shape, data(base64)}`: compact, lossless, and
trivially decodable on the JS side (`atob` + a `TypedArray` over the buffer). The
server ships raw channels; colorization happens on the front.
"""

from __future__ import annotations

import base64
import json
import struct

import numpy as np

from ..core.poses import pose_to_matrix
from ..core.source import ChannelSpec, Frame, SequenceSpec


# --------------------------------------------------------------- arrays
def _wire_array(arr) -> np.ndarray:
    """Fit an array for the JS side. 64-bit integers arrive as BigInt64Array there,
    whose elements are BigInts — label ids then miss every Map lookup silently. Narrow
    to 32 bits whenever the values fit (labels always do); genuinely huge ints pass
    through untouched."""
    arr = np.ascontiguousarray(arr)
    if arr.dtype in (np.int64, np.uint64) and arr.size:
        lo, hi = arr.min(), arr.max()
        if np.iinfo(np.int32).min <= lo and hi <= np.iinfo(np.int32).max:
            return arr.astype(np.int32)
    elif arr.dtype in (np.int64, np.uint64):
        return arr.astype(np.int32)  # empty: dtype consistency across frames
    return arr


def encode_array(arr) -> dict | None:
    if arr is None:
        return None
    arr = _wire_array(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(d: dict | None):
    if d is None:
        return None
    buf = base64.b64decode(d["data"])
    return np.frombuffer(buf, dtype=np.dtype(d["dtype"])).reshape(d["shape"])


# --------------------------------------------------------------- specs
def channelspec_to_dict(spec: ChannelSpec) -> dict:
    placement = None if spec.placement is None else pose_to_matrix(spec.placement).tolist()
    return {
        "name": spec.name,
        "kind": spec.kind.value,
        "dtype": None if spec.dtype is None else str(spec.dtype),
        "shape": None if spec.shape is None else list(spec.shape),
        "of": spec.of,
        "labelset": None if spec.labelset is None else spec.labelset.to_dict(),
        "placement": placement,  # 4x4 ego-frame pose, or null
    }


def sequencespec_to_dict(spec: SequenceSpec) -> dict:
    return {"id": spec.id, "n_frames": spec.n_frames}


# --------------------------------------------------------------- frames
def frame_to_dict(seq: str, index: int, n_frames: int, frame: Frame) -> dict:
    """One frame message: raw channels + position. Labelings ride along untouched;
    normalization (dtype fitting for the wire) keeps semantics: labels stay ints."""
    return {
        "seq": seq,
        "index": index,
        "n_frames": n_frames,
        "timestamps": frame.timestamps,
        "channels": {k: encode_array(v) for k, v in frame.channels.items()},
    }


# --------------------------------------------------------------- binary frames (websocket)
# One message = <u32 header length> + JSON header + concatenated raw buffers.
# Header and every buffer are padded to 8 bytes so the JS side can map TypedArrays
# straight over the message buffer (a Float64Array view needs 8-byte alignment).
_ALIGN = 8


def _pad(n: int) -> int:
    return (-n) % _ALIGN


def pack_frame(seq: str, index: int, n_frames: int, frame: Frame, back: int = 0, fwd: int = 0) -> bytes:
    header: dict = {
        "seq": seq,
        "index": index,
        "n_frames": n_frames,
        # accumulation echoed so the client can match pipelined replies to requests
        "back": back,
        "fwd": fwd,
        "timestamps": frame.timestamps,
        "channels": [],
    }
    blobs: list[bytes] = []
    offset = 0
    for name, value in frame.channels.items():
        arr = _wire_array(value)
        header["channels"].append(
            {"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape), "offset": offset}
        )
        buf = arr.tobytes()
        buf += b"\x00" * _pad(len(buf))
        blobs.append(buf)
        offset += len(buf)

    hjson = json.dumps(header).encode()
    hjson += b" " * _pad(4 + len(hjson))  # header + length prefix end on an 8-byte boundary
    return struct.pack("<I", len(hjson)) + hjson + b"".join(blobs)


def unpack_frame(payload: bytes) -> dict:
    """Inverse of `pack_frame` (tests + Python clients)."""
    (hlen,) = struct.unpack_from("<I", payload, 0)
    header = json.loads(payload[4 : 4 + hlen].decode())
    base = 4 + hlen
    channels = {}
    for ch in header["channels"]:
        dtype = np.dtype(ch["dtype"])
        count = int(np.prod(ch["shape"], dtype=np.int64)) if ch["shape"] else 1
        start = base + ch["offset"]
        arr = np.frombuffer(payload, dtype=dtype, count=count, offset=start)
        channels[ch["name"]] = arr.reshape(ch["shape"])
    header["channels"] = channels
    return header
