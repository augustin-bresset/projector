import numpy as np
from fastapi.testclient import TestClient

from projector.demo import make_demo_source
from projector.server.app import create_app
from projector.server.protocol import decode_array, unpack_frame


def _client():
    return TestClient(create_app(make_demo_source(), title="test"))


def test_session():
    c = _client()
    s = c.get("/api/session").json()
    assert [q["id"] for q in s["sequences"]] == ["seq_a", "seq_b"]
    gt = next(ch for ch in s["channels"] if ch["name"] == "gt")
    assert gt["kind"] == "labels" and gt["of"] == "lidar"
    assert gt["labelset"]["classes"][1]["name"] == "traversable"


def test_rest_frame_clamps():
    c = _client()
    f = c.get("/api/frame?seq=seq_b&index=9999").json()
    assert f["seq"] == "seq_b" and f["index"] == f["n_frames"] - 1
    lidar = decode_array(f["channels"]["lidar"])
    gt = decode_array(f["channels"]["gt"])
    assert lidar.ndim == 2 and len(gt) == len(lidar)


def test_websocket_stream():
    c = _client()
    with c.websocket_connect("/ws/frames") as ws:
        ws.send_json({"seq": "seq_a", "index": 3})
        out = unpack_frame(ws.receive_bytes())
        assert out["seq"] == "seq_a" and out["index"] == 3 and out["n_frames"] == 40
        assert set(out["channels"]) == {"lidar", "camera_front", "pose", "gt", "pred"}
        assert out["channels"]["lidar"].dtype == np.float32

        # pipelined requests come back in order
        for i in (4, 5, 6):
            ws.send_json({"seq": "seq_a", "index": i})
        got = [unpack_frame(ws.receive_bytes())["index"] for _ in range(3)]
        assert got == [4, 5, 6]


def test_empty_launch_and_fs(tmp_path):
    from projector.server.app import create_app as mk

    (tmp_path / "plain").mkdir()
    (tmp_path / "ds" / ".apairo").mkdir(parents=True)
    c = TestClient(mk())  # no source: idle viewer
    s = c.get("/api/session").json()
    assert s["sequences"] == [] and s["channels"] == []
    assert c.get("/api/frame").status_code == 422

    listing = c.get(f"/api/fs?path={tmp_path}").json()
    flags = {d["name"]: d["is_dataset"] for d in listing["dirs"]}
    assert flags == {"plain": False, "ds": True}
    assert c.get("/api/fs?path=/nonexistent-dir-xyz").status_code == 400


def test_open_rejects_bad_path():
    c = _client()
    r = c.post("/api/open", json={"path": "/nonexistent-dir-xyz"})
    assert r.status_code == 422
    # the previous source is untouched
    assert c.get("/api/session").json()["sequences"] != []


def test_trajectory_endpoint():
    c = _client()
    d = c.get("/api/trajectory?seq=seq_a").json()
    pts = decode_array(d["points"])
    assert pts.shape == (40, 3)  # demo: ego advances in +x every frame
    assert pts[1, 0] > pts[0, 0]


def test_websocket_accum():
    c = _client()
    with c.websocket_connect("/ws/frames") as ws:
        ws.send_json({"seq": "seq_a", "index": 5})
        base = unpack_frame(ws.receive_bytes())
        ws.send_json({"seq": "seq_a", "index": 5, "back": 2, "fwd": 2})
        acc = unpack_frame(ws.receive_bytes())
        assert acc["back"] == 2 and acc["fwd"] == 2
        assert len(acc["channels"]["lidar"]) > len(base["channels"]["lidar"]) * 3
        assert len(acc["channels"]["gt"]) == len(acc["channels"]["lidar"])


def test_state_sidecar_roundtrip(tmp_path, monkeypatch):
    from projector.server.app import create_app as mk

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    c = TestClient(mk(make_demo_source(), state_key="test-dataset"))
    assert c.get("/api/state").json()["state"] is None
    payload = {"position": {"seq": "seq_a", "index": 7}, "bookmarks": [{"seq": "seq_a", "index": 3}]}
    assert c.post("/api/state", json=payload).json()["ok"] is True
    assert c.get("/api/state").json()["state"] == payload
    # no key → saves are dropped, reads are null
    c2 = TestClient(mk(make_demo_source()))
    assert c2.post("/api/state", json=payload).json()["ok"] is False


def test_timeline_endpoint_nulls_without_timestamps():
    c = _client()
    d = c.get("/api/timeline?seq=seq_a").json()
    assert d["channels"] is None and d["ticks"] is None


def test_websocket_decimation():
    app = create_app(make_demo_source(), max_points=500)
    c = TestClient(app)
    with c.websocket_connect("/ws/frames") as ws:
        ws.send_json({"seq": "seq_a", "index": 0})
        out = unpack_frame(ws.receive_bytes())
        assert len(out["channels"]["lidar"]) == 500
        assert len(out["channels"]["gt"]) == 500
