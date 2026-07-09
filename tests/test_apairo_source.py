"""ApairoSource against duck-typed fakes — no apairo install needed."""

import numpy as np

from projector.adapters.apairo_source import ApairoSource
from projector.core.source import ChannelKind


class _Sample:
    def __init__(self, data, timestamp=None):
        self.data = data
        self.timestamp = timestamp


class _FakeSyncDataset:
    """Synchronized view: every sample carries all channels, with a reference ts."""

    keys = ["lidar", "camera", "pose", "ground_truth", "intensity_like"]

    def __init__(self, n=6, n_points=200):
        rng = np.random.default_rng(0)
        self._frames = []
        for i in range(n):
            self._frames.append(
                _Sample(
                    {
                        "lidar": rng.normal(size=(n_points, 4)).astype(np.float32),
                        "camera": np.zeros((8, 12, 3), dtype=np.uint8),
                        "pose": np.eye(4, dtype=np.float32),
                        "ground_truth": (np.arange(n_points) % 3).astype(np.int64),
                        "intensity_like": rng.normal(size=n_points).astype(np.float32),
                    },
                    timestamp=100.0 + i * 0.1,
                )
            )

    is_synchronous = True

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


class _FakeEventDataset:
    """Async events: one channel per sample, own timestamp."""

    keys = ["lidar", "camera"]
    is_synchronous = False

    def __init__(self, n_points=100):
        rng = np.random.default_rng(1)
        cloud = rng.normal(size=(n_points, 4)).astype(np.float32)
        img = np.zeros((4, 6, 3), dtype=np.uint8)
        self._events = [
            _Sample({"lidar": cloud}, 10.0),
            _Sample({"camera": img}, 10.03),
            _Sample({"lidar": cloud}, 10.1),
            _Sample({"camera": img}, 10.13),
        ]

    def __len__(self):
        return len(self._events)

    def __getitem__(self, i):
        return self._events[i]


def test_sync_classification_and_labels_detection():
    src = ApairoSource({"seq": _FakeSyncDataset()})
    kinds = {s.name: s.kind for s in src.channels()}
    assert kinds["lidar"] is ChannelKind.POINTCLOUD
    assert kinds["camera"] is ChannelKind.IMAGE
    assert kinds["pose"] is ChannelKind.POSE
    assert kinds["ground_truth"] is ChannelKind.LABELS
    assert kinds["intensity_like"] is ChannelKind.SCALAR  # float (N,) stays a measure

    gt = next(s for s in src.channels() if s.name == "ground_truth")
    assert gt.of == "lidar"
    assert sorted(c.id for c in gt.labelset.classes) == [0, 1, 2]


def test_sync_frame_carries_reference_timestamps():
    src = ApairoSource({"seq": _FakeSyncDataset()})
    f = src.frame("seq", 2)
    assert set(f.channels) == {"lidar", "camera", "pose", "ground_truth", "intensity_like"}
    assert f.timestamps["lidar"] == 100.2


def test_sequences_lengths():
    src = ApairoSource({"a": _FakeSyncDataset(n=6), "b": _FakeSyncDataset(n=3)})
    assert [(s.id, s.n_frames) for s in src.sequences()] == [("a", 6), ("b", 3)]


def test_events_mode_partial_frames():
    src = ApairoSource({"seq": _FakeEventDataset()})
    kinds = {s.name: s.kind for s in src.channels()}
    assert kinds == {"lidar": ChannelKind.POINTCLOUD, "camera": ChannelKind.IMAGE}

    f0 = src.frame("seq", 0)
    assert set(f0.channels) == {"lidar"}
    assert f0.timestamps == {"lidar": 10.0}
    f1 = src.frame("seq", 1)
    assert set(f1.channels) == {"camera"}
    assert f1.timestamps == {"camera": 10.03}


def test_keys_filter_and_order():
    src = ApairoSource({"seq": _FakeSyncDataset()}, keys=["camera", "lidar"])
    assert [s.name for s in src.channels()] == ["camera", "lidar"]
    f = src.frame("seq", 0)
    assert set(f.channels) == {"camera", "lidar"}


def test_event_timeline_groups_cotimestamped_events():
    from projector.adapters.apairo_source import _EventTimeline

    n_pts = 2000  # cloud and labels above the carry threshold — they stay strict

    class _OneChannel:
        """Single-channel dataset: index = event index."""

        def __init__(self, name, values, ts):
            self._name, self._values, self.ts = name, values, ts

        def __getitem__(self, j):
            return _Sample({self._name: self._values[j]}, self.ts[j])

    cloud = np.random.default_rng(0).normal(size=(n_pts, 4)).astype(np.float32)
    labels = (np.arange(n_pts) % 2).astype(np.int32)
    # labels share the cloud's timestamps (apairo timestamps_from); camera has its own
    lidar_ts = np.array([0.0, 0.1, 0.2])
    cam_ts = np.array([0.05, 0.15])
    tl = _EventTimeline(
        {
            "lidar": (_OneChannel("lidar", [cloud] * 3, lidar_ts), lidar_ts),
            "trav": (_OneChannel("trav", [labels] * 3, lidar_ts), lidar_ts),
            "camera": (_OneChannel("camera", [np.zeros((4, 6, 3), np.uint8)] * 2, cam_ts), cam_ts),
        }
    )
    assert len(tl) == 5  # 3 lidar+trav ticks interleaved with 2 camera ticks
    s0 = tl[0]
    assert set(s0.data) == {"lidar", "trav"}  # co-timestamped events ride together
    assert s0.channel_timestamps == {"lidar": 0.0, "trav": 0.0}
    assert set(tl[1].data) == {"camera"}
    # the tiny fake camera (< 4 KB) is carried onto later ticks with its own ts —
    # the "latest state" semantics small channels (pose, IMU) get in events mode
    s2 = tl[2]
    assert set(s2.data) == {"lidar", "trav", "camera"}
    assert s2.channel_timestamps["camera"] == 0.05
    assert s2.channel_timestamps["lidar"] == 0.1
    # per-channel event counters (apairo_rr-style): each sensor keeps its own
    assert s2.channel_indices == {"lidar": 1, "trav": 1, "camera": 0}

    # end to end: the labeling classifies as LABELS of lidar without synchronizing
    src = ApairoSource({"seq": tl})
    spec = next(s for s in src.channels() if s.name == "trav")
    assert spec.kind is ChannelKind.LABELS and spec.of == "lidar"
    f2 = src.frame("seq", 2)
    assert f2.indices == {"lidar": 1, "trav": 1, "camera": 0}


def test_mount_tf_applied_to_clouds():
    src = ApairoSource({"seq": _FakeSyncDataset()})
    # 90° about x + a lift: y→z, z→-y, +1 on z
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    T[2, 3] = 1.0
    raw = src.frame("seq", 0).channels["lidar"].copy()
    src.mount = T
    out = src.frame("seq", 0).channels["lidar"]
    np.testing.assert_allclose(out[:, 0], raw[:, 0], atol=1e-5)
    np.testing.assert_allclose(out[:, 1], -raw[:, 2], atol=1e-5)
    np.testing.assert_allclose(out[:, 2], raw[:, 1] + 1.0, atol=1e-5)
    np.testing.assert_allclose(out[:, 3], raw[:, 3], atol=1e-5)  # intensity untouched
    # labelings still aligned (length unchanged)
    assert len(src.frame("seq", 0).channels["ground_truth"]) == len(out)


def test_sparse_overlay_view():
    from projector.adapters.apairo_source import _SparseSync

    sync = _FakeSyncDataset(n=6)  # ticks at 100.0, 100.1, ... 100.5
    n_pts = sync[0].data["lidar"].shape[0]

    class _SparseChannel:
        """Single-channel dataset: 2 labeling events, at ticks 1 and 4."""

        def __init__(self):
            self._ts = np.array([100.1, 100.4])

        def __getitem__(self, j):
            return _Sample({"sparse_gt": np.full(n_pts, j, dtype=np.int64)}, self._ts[j])

    view = _SparseSync(sync, {"sparse_gt": (_SparseChannel(), np.array([100.1, 100.4]))}, tolerance=0.02)
    assert len(view) == 6
    assert "sparse_gt" not in view[0].data  # no event near tick 0
    assert "sparse_gt" in view[1].data  # event 0 sits on tick 1
    assert view[1].data["sparse_gt"][0] == 0
    assert view[1].channel_timestamps["sparse_gt"] == 100.1
    assert "sparse_gt" not in view[2].data
    assert view[4].data["sparse_gt"][0] == 1  # event 1 sits on tick 4

    # extra_samples: one merged sample at the first event's tick, so classification
    # sees the sparse labeling next to its cloud
    extras = view.extra_samples()
    assert len(extras) == 1 and "sparse_gt" in extras[0] and "lidar" in extras[0]

    # end to end: the sparse channel classifies as LABELS of lidar
    src = ApairoSource({"seq": view})
    spec = next(s for s in src.channels() if s.name == "sparse_gt")
    assert spec.kind is ChannelKind.LABELS and spec.of == "lidar"
    f1 = src.frame("seq", 1)
    assert "sparse_gt" in f1.channels and f1.timestamps["sparse_gt"] == 100.1
    f2 = src.frame("seq", 2)
    assert "sparse_gt" not in f2.channels
