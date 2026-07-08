import numpy as np
import pytest

from projector.adapters.array_source import ArraySource
from projector.core.labels import LabelSet
from projector.core.source import ChannelKind, ChannelSpec
from projector.engine.player import Player


def _source(n_points=1000):
    specs = [
        ChannelSpec("lidar", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 4)),
        ChannelSpec(
            "gt", ChannelKind.LABELS, np.dtype("int32"), (None,), of="lidar", labelset=LabelSet.default()
        ),
    ]
    frames = [
        {
            "lidar": np.random.default_rng(i).normal(size=(n_points, 4)).astype(np.float32),
            "gt": (np.arange(n_points) % 3).astype(np.int32),
        }
        for i in range(4)
    ]
    return ArraySource(specs, {"a": frames, "b": frames[:2]})


def test_clamping_and_fallback():
    p = Player(_source())
    assert p.clamp("a", -5) == ("a", 0)
    assert p.clamp("a", 999) == ("a", 3)
    assert p.clamp("nope", 1) == ("a", 1)
    assert p.clamp(None, 2) == ("a", 2)


def test_frame_remembers_position():
    p = Player(_source())
    seq, idx, _ = p.frame("b", 1)
    assert (seq, idx) == ("b", 1)
    assert (p.seq, p.index) == ("b", 1)


def test_decimation_keeps_labels_aligned():
    p = Player(_source(n_points=1000), max_points=100)
    _, _, frame = p.frame("a", 0)
    assert len(frame.channels["lidar"]) == 100
    assert len(frame.channels["gt"]) == 100
    # decimated labels follow the same indices as the cloud
    full = _source(n_points=1000).frame("a", 0)
    idx = np.linspace(0, 999, 100).astype(np.int64)
    np.testing.assert_array_equal(frame.channels["gt"], full.channels["gt"][idx])


def test_no_decimation_below_cap():
    p = Player(_source(n_points=50), max_points=100)
    _, _, frame = p.frame("a", 0)
    assert len(frame.channels["lidar"]) == 50


def _posed_source(n_points=100, n_frames=4, step=2.0):
    """Ego advances `step` in +x per frame; each frame's cloud is local (around 0)."""
    specs = [
        ChannelSpec("lidar", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 4)),
        ChannelSpec(
            "gt", ChannelKind.LABELS, np.dtype("int32"), (None,), of="lidar", labelset=LabelSet.default()
        ),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
    ]
    frames = []
    for i in range(n_frames):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = step * i
        frames.append(
            {
                "lidar": np.ones((n_points, 4), dtype=np.float32),
                "gt": np.full(n_points, i, dtype=np.int32),  # frame id as label → traceable
                "pose": pose,
            }
        )
    return ArraySource(specs, {"a": frames})


def test_accumulation_registers_and_labels_follow():
    p = Player(_posed_source(n_points=100, step=2.0))
    _, _, frame = p.frame("a", 1, back=1, fwd=1)  # window = frames 0..2
    lidar, gt = frame.channels["lidar"], frame.channels["gt"]
    assert len(lidar) == 300 and len(gt) == 300
    # reference part first (frame 1), then neighbors 0 and 2
    np.testing.assert_array_equal(np.unique(gt), [0, 1, 2])
    # frame 0's points land at ref_x - 2 (its cloud x=1 → 1 - 2 = -1), frame 2's at +3
    xs_by_label = {int(lab): frame.channels["lidar"][gt == lab][:, 0] for lab in (0, 1, 2)}
    np.testing.assert_allclose(xs_by_label[1], 1.0, atol=1e-5)  # ref stays put
    np.testing.assert_allclose(xs_by_label[0], -1.0, atol=1e-5)
    np.testing.assert_allclose(xs_by_label[2], 3.0, atol=1e-5)


def test_accumulation_then_decimation_stays_aligned():
    p = Player(_posed_source(n_points=100), max_points=60)
    _, _, frame = p.frame("a", 1, back=1, fwd=1)
    assert len(frame.channels["lidar"]) == 60
    assert len(frame.channels["gt"]) == 60


def test_trajectory():
    p = Player(_posed_source(n_frames=5, step=2.0))
    traj = p.trajectory("a")
    assert traj.shape == (5, 3)
    np.testing.assert_allclose(traj[:, 0], [0, 2, 4, 6, 8])


def test_empty_source_is_idle():
    # An empty source is the valid "launched bare, waiting for /api/open" state;
    # only frame access is an error.
    p = Player(ArraySource([], {}))
    assert p.sequences == [] and p.seq is None
    with pytest.raises(ValueError):
        p.frame()
