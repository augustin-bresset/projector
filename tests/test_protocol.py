import numpy as np

from projector.core.source import Frame
from projector.server.protocol import decode_array, encode_array, pack_frame, unpack_frame


def _frame():
    return Frame(
        channels={
            "lidar": np.random.default_rng(0).normal(size=(101, 4)).astype(np.float32),
            "gt": np.arange(101, dtype=np.int32) % 3,
            "camera": np.zeros((7, 9, 3), dtype=np.uint8),
            "pose": np.eye(4, dtype=np.float64),
        },
        timestamps={"lidar": 12.5, "gt": 12.5, "camera": 12.5, "pose": 12.5},
    )


def test_base64_roundtrip():
    arr = np.random.default_rng(1).normal(size=(50, 3)).astype(np.float32)
    out = decode_array(encode_array(arr))
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == arr.dtype


def test_pack_unpack_roundtrip():
    frame = _frame()
    payload = pack_frame("seq_a", 5, 40, frame)
    out = unpack_frame(payload)
    assert out["seq"] == "seq_a" and out["index"] == 5 and out["n_frames"] == 40
    assert out["timestamps"]["lidar"] == 12.5
    for k, v in frame.channels.items():
        np.testing.assert_array_equal(out["channels"][k], v)
        assert out["channels"][k].dtype == v.dtype


def test_int64_narrowed_for_the_wire():
    # int64 labels become BigInt64Array in JS and miss every Map lookup — the wire
    # narrows them to int32 whenever the values fit.
    frame = Frame(channels={"gt": np.array([0, 1, 2], dtype=np.int64)})
    out = unpack_frame(pack_frame("s", 0, 1, frame))
    assert out["channels"]["gt"].dtype == np.int32
    np.testing.assert_array_equal(out["channels"]["gt"], [0, 1, 2])
    # huge values stay int64 (honest, even if JS gets BigInts)
    frame = Frame(channels={"big": np.array([2**40], dtype=np.int64)})
    out = unpack_frame(pack_frame("s", 0, 1, frame))
    assert out["channels"]["big"].dtype == np.int64


def test_pack_alignment():
    # Header and every blob end on 8-byte boundaries, whatever the shapes.
    for n in (1, 2, 3, 50, 101):
        frame = Frame(
            channels={
                "a": np.ones(n, dtype=np.uint8),  # odd byte counts
                "b": np.ones((n, 3), dtype=np.float64),  # needs 8-byte alignment
            }
        )
        payload = pack_frame("s", 0, 1, frame)
        out = unpack_frame(payload)
        np.testing.assert_array_equal(out["channels"]["b"], np.ones((n, 3)))
        import struct

        (hlen,) = struct.unpack_from("<I", payload, 0)
        assert (4 + hlen) % 8 == 0
