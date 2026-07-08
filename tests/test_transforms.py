import numpy as np
import pytest

from projector.core.source import ChannelKind, Frame
from projector.engine.transforms import ActiveTransform, apply_transforms, load_script, virtual_specs

SCRIPT = '''
import numpy as np

def high_points(pts, threshold: float = 1.0):
    """Label points above threshold as class 1."""
    return (pts[:, 2] > threshold).astype(np.int32)

class RangeFilter:
    """Drop points farther than max_range."""
    def __init__(self, max_range: float = 10.0):
        self.max_range = max_range
    def __call__(self, pts):
        return pts[np.linalg.norm(pts[:, :2], axis=1) <= self.max_range]

def _private_helper(x):
    return x

not_a_transform = 42
'''


@pytest.fixture
def script(tmp_path):
    p = tmp_path / "my_transforms.py"
    p.write_text(SCRIPT)
    return str(p)


def _frame(n=100):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(n, 4)).astype(np.float32) * 5
    return Frame(channels={"lidar": pts})


def test_discovery(script):
    specs = load_script(script)
    names = {s.name for s in specs}
    assert names == {"high_points", "RangeFilter"}
    hp = next(s for s in specs if s.name == "high_points")
    assert hp.params == {"threshold": 1.0} and not hp.is_factory
    rf = next(s for s in specs if s.name == "RangeFilter")
    assert rf.params == {"max_range": 10.0} and rf.is_factory
    assert "Label points" in hp.doc


def test_labels_output_becomes_virtual_channel(script):
    spec = next(s for s in load_script(script) if s.name == "high_points")
    t = ActiveTransform(spec=spec, cloud="lidar", params={"threshold": 0.0})
    t.bind()
    frame = _frame()
    out = apply_transforms(frame, [t])
    assert "high_points" in out.channels
    labels = out.channels["high_points"]
    assert labels.dtype == np.int32 and len(labels) == 100
    np.testing.assert_array_equal(labels, (frame.channels["lidar"][:, 2] > 0).astype(np.int32))
    # virtual spec advertises it as LABELS of the cloud
    vs = virtual_specs([t])
    assert vs[0].kind is ChannelKind.LABELS and vs[0].of == "lidar"


def test_cloud_output_replaces_channel(script):
    spec = next(s for s in load_script(script) if s.name == "RangeFilter")
    t = ActiveTransform(spec=spec, cloud="lidar", params={"max_range": 3.0})
    t.bind()
    out = apply_transforms(_frame(), [t])
    assert len(out.channels["lidar"]) < 100
    assert np.all(np.linalg.norm(out.channels["lidar"][:, :2], axis=1) <= 3.0)


def test_user_error_is_captured_not_raised(script, tmp_path):
    p = tmp_path / "boom.py"
    p.write_text("def broken(pts):\n    raise RuntimeError('nope')\n")
    spec = load_script(str(p))[0]
    t = ActiveTransform(spec=spec, cloud="lidar", params={})
    t.bind()
    out = apply_transforms(_frame(), [t])  # must not raise
    assert "broken" not in out.channels
    assert "nope" in t.last_error


def test_endpoints(script):
    from fastapi.testclient import TestClient

    from projector.demo import make_demo_source
    from projector.server.app import create_app
    from projector.server.protocol import unpack_frame

    c = TestClient(create_app(make_demo_source()))
    d = c.post("/api/plugins/load", json={"path": script}).json()
    assert {s["name"] for s in d["specs"]} == {"high_points", "RangeFilter"}

    hp_id = next(s["id"] for s in d["specs"] if s["name"] == "high_points")
    d = c.post("/api/plugins/active", json={"active": [{"id": hp_id, "params": {"threshold": 0.5}}]}).json()
    assert any(ch["name"] == "high_points" and ch["kind"] == "labels" for ch in d["session"]["channels"])

    with c.websocket_connect("/ws/frames") as ws:
        ws.send_json({"seq": "seq_a", "index": 0})
        out = unpack_frame(ws.receive_bytes())
        assert "high_points" in out["channels"]
        assert len(out["channels"]["high_points"]) == len(out["channels"]["lidar"])
