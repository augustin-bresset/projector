"""FastAPI app: serves a `Source` over REST + websocket and hosts the web front.

Frames are stateless (`GET /api/frame`, or the binary `/ws/frames` stream), so any
number of clients can scrub independently. The websocket path is the playback one:
JSON request in, binary frame out (see `protocol.pack_frame`); frame reads run in a
thread executor so disk-bound sources never stall the event loop.

The served source is swappable at runtime (`POST /api/open`): the viewer can start
empty and open apairo datasets browsed via `/api/fs` + `/api/probe`.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from functools import partial
from pathlib import Path

# fastapi imported at module level on purpose: with `from __future__ import annotations`
# the `ws: WebSocket` annotation is a string, and FastAPI can only resolve it from the
# module namespace — a function-local import silently turns `ws` into a query param
# (the websocket then closes with 1008). Only cli/desktop import this module, so the
# numpy-only core stays importable without fastapi.
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..adapters.array_source import ArraySource
from ..engine.player import Player
from .files import list_dir, probe
from .protocol import channelspec_to_dict, frame_to_dict, pack_frame, sequencespec_to_dict

# Web front (vanilla, zero build) served as-is — packaged at `projector/web/`.
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app(
    source=None,
    *,
    title: str = "Projector",
    max_points: int | None = None,
    state_key: str | None = None,
    open_params: dict | None = None,
):
    # Mutable holder so /api/open can swap the served dataset; every endpoint —
    # including the long-lived websocket loop — reads through it.
    box = {
        "player": Player(source if source is not None else ArraySource([], {}), max_points=max_points),
        "title": title,
        "state_key": state_key,  # identity of the served dataset (session sidecar)
        "open_params": open_params,  # how the dataset was opened (lets /api/sync re-open)
        "was_async": bool(getattr(source, "was_async", False)),
        # user transforms loaded from scripts (in-memory only, see engine/transforms.py)
        "tf_specs": {},  # id → TransformSpec
        "tf_active": [],  # [ActiveTransform]
    }
    if open_params:
        from .state import save_last

        save_last(dict(open_params))
    app = FastAPI(title=f"{title} API", version="0.1")

    # The front may be served from another origin (dev): allow everything.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def session_dict() -> dict:
        from ..engine.transforms import virtual_specs

        player = box["player"]
        params = box.get("open_params")
        sync = None
        if params and box.get("was_async"):
            sync = {
                "reference": params.get("reference"),
                "tolerance": float(params.get("tolerance") or 0.1),
            }
        channels = [channelspec_to_dict(s) for s in player.source.channels()]
        channels += [channelspec_to_dict(s) for s in virtual_specs(box["tf_active"])]
        return {
            "title": box["title"],
            "sequences": [sequencespec_to_dict(s) for s in player.sequences],
            "channels": channels,
            "position": {"seq": player.seq, "index": player.index},
            "sync": sync,  # non-null when the dataset can be resampled (/api/sync)
        }

    def _framed(seq, index, back, fwd):
        """One frame through the player + the active user transforms."""
        from ..engine.transforms import apply_transforms

        player = box["player"]
        seq, index, frame = player.frame(seq, index, back=back, fwd=fwd)
        if box["tf_active"]:
            frame = apply_transforms(frame, box["tf_active"])
        return seq, index, frame

    # LRU of PACKED payloads: revisiting a frame (scrub back, loop wrap) skips
    # disk reads, accumulation, transforms and packing entirely. Cleared whenever
    # the served content changes (open/sync/plugins).
    frame_cache: OrderedDict[tuple, bytes] = OrderedDict()
    _CACHE_ENTRIES = 64
    _CACHE_BYTES = 128 * 1024 * 1024

    def _invalidate_frames() -> None:
        frame_cache.clear()

    def _packed(seq, index, back, fwd) -> bytes:
        key = (seq, index, back, fwd)
        hit = frame_cache.get(key)
        if hit is not None:
            frame_cache.move_to_end(key)
            return hit
        seq, index, frame = _framed(seq, index, back, fwd)
        payload = pack_frame(seq, index, box["player"].n_frames(seq), frame, back=back, fwd=fwd)
        frame_cache[(seq, index, back, fwd)] = payload
        while len(frame_cache) > _CACHE_ENTRIES or (
            len(frame_cache) > 1 and sum(len(v) for v in frame_cache.values()) > _CACHE_BYTES
        ):
            frame_cache.popitem(last=False)
        return payload

    @app.get("/api/session")
    def get_session() -> dict:
        return session_dict()

    @app.get("/api/frame")
    def get_frame(seq: str | None = None, index: int = 0, back: int = 0, fwd: int = 0) -> dict:
        try:
            seq, index, frame = _framed(seq, index, back, fwd)
        except (KeyError, IndexError, ValueError) as e:
            raise HTTPException(422, f"cannot fetch frame: {e}") from e
        return frame_to_dict(seq, index, box["player"].n_frames(seq), frame)

    @app.get("/api/timeline")
    def get_timeline(seq: str) -> dict:
        """Per-channel event timestamps + tick times (footer tracks), or nulls."""
        from .protocol import encode_array

        player = box["player"]
        try:
            tl = player.timeline(seq)
        except (KeyError, ValueError) as e:
            raise HTTPException(422, str(e)) from e
        if not tl:
            return {"seq": seq, "channels": None, "ticks": None}
        import numpy as np

        return {
            "seq": seq,
            "channels": {k: encode_array(np.asarray(v, np.float64)) for k, v in tl["channels"].items()},
            "ticks": None if tl["ticks"] is None else encode_array(np.asarray(tl["ticks"], np.float64)),
        }

    @app.get("/api/trajectory")
    def get_trajectory(seq: str) -> dict:
        """Ego positions (N, 3) of a sequence's pose channel, or points=null."""
        from .protocol import encode_array

        player = box["player"]
        try:
            traj = player.trajectory(seq)
        except (KeyError, ValueError) as e:
            raise HTTPException(422, str(e)) from e
        import numpy as np

        pts = None if traj is None else encode_array(np.asarray(traj, dtype=np.float32))
        return {"seq": seq, "points": pts}

    @app.websocket("/ws/frames")
    async def ws_frames(ws: WebSocket) -> None:
        """Binary frame stream: `{"seq": id, "index": i}` in, packed frame out.

        Requests are served in order on one connection; the client matches replies
        by the (seq, index) echoed in the header, so it can pipeline prefetches.
        """
        await ws.accept()
        loop = asyncio.get_running_loop()
        try:
            while True:
                msg = await ws.receive_json()
                back, fwd = int(msg.get("back", 0)), int(msg.get("fwd", 0))
                call = partial(_packed, msg.get("seq"), int(msg.get("index", 0)), back, fwd)
                try:
                    payload = await loop.run_in_executor(None, call)
                except (KeyError, IndexError, ValueError):
                    continue  # e.g. request for the previous dataset — drop it
                await ws.send_bytes(payload)
        except WebSocketDisconnect:
            pass

    # -------------------------------------------------------- open-dataset flow
    @app.get("/api/fs")
    def fs_list(path: str | None = None, files: str | None = None) -> dict:
        try:
            return list_dir(path, files=files)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/probe")
    def fs_probe(path: str) -> dict:
        try:
            return probe(path)
        except ImportError as e:
            raise HTTPException(400, f"apairo is not installed: {e}") from e
        except Exception as e:  # noqa: BLE001 — surface whatever apairo raised
            raise HTTPException(422, f"cannot probe {path}: {e}") from e

    def _open(params: dict) -> dict:
        """Open an apairo dataset and swap the served player. `params` is remembered
        so /api/sync can re-open with a different time model."""
        from ..adapters.apairo_source import ApairoSource

        try:
            src = ApairoSource.from_path(
                params["path"],
                keys=params.get("channels") or None,
                reference=params.get("reference") or None,
                tolerance=float(params.get("tolerance") or 0.1),
                sequences=params.get("sequences") or None,
                every=int(params.get("every") or 1),
                upright=bool(params.get("upright", True)),
            )
            box["player"] = Player(src, max_points=max_points)
            box["title"] = f"Projector — {Path(params['path']).name}"
            box["state_key"] = str(Path(params["path"]).expanduser().resolve())
            box["open_params"] = dict(params)
            box["was_async"] = bool(getattr(src, "was_async", False))
            _invalidate_frames()
            from .state import save_last

            save_last(dict(params))
        except KeyError as e:
            raise HTTPException(422, f"missing field: {e}") from e
        except ImportError as e:
            raise HTTPException(400, f"apairo is not installed: {e}") from e
        except Exception as e:  # noqa: BLE001 — synchronize/classification failures
            raise HTTPException(422, f"cannot open dataset: {e}") from e
        return session_dict()

    @app.post("/api/open")
    def open_dataset(payload: dict = Body(...)) -> dict:
        return _open(payload)

    @app.post("/api/sync")
    def resync(payload: dict = Body(...)) -> dict:
        """Re-open the current dataset with another time model: a reference channel
        (resampled clock) or none (event timeline)."""
        params = box.get("open_params")
        if not params:
            raise HTTPException(400, "no reopenable dataset (started from a custom source)")
        params = dict(params)
        params["reference"] = payload.get("reference") or None
        if payload.get("tolerance") is not None:
            params["tolerance"] = float(payload["tolerance"])
        return _open(params)

    # -------------------------------------------------------- user transforms
    def _plugins_dict() -> dict:
        return {
            "specs": [
                {"id": s.id, "name": s.name, "source": s.source, "params": s.params, "doc": s.doc}
                for s in box["tf_specs"].values()
            ],
            "active": [
                {"id": t.spec.id, "cloud": t.cloud, "params": t.params, "error": t.last_error}
                for t in box["tf_active"]
            ],
        }

    @app.get("/api/plugins")
    def get_plugins() -> dict:
        return _plugins_dict()

    @app.post("/api/plugins/load")
    def plugins_load(payload: dict = Body(...)) -> dict:
        """Import a script and register its transform candidates (nothing active yet)."""
        from ..engine.transforms import load_script

        try:
            specs = load_script(payload["path"])
        except KeyError as e:
            raise HTTPException(422, f"missing field: {e}") from e
        except Exception as e:  # noqa: BLE001 — user script errors go to the UI
            raise HTTPException(422, f"cannot load script: {e}") from e
        for s in specs:
            box["tf_specs"][s.id] = s
        return _plugins_dict()

    @app.post("/api/plugins/unload")
    def plugins_unload(payload: dict = Body(...)) -> dict:
        """Forget a script: drop its specs and deactivate the transforms it provided."""
        path = payload.get("path")
        if not path:
            raise HTTPException(422, "missing field: path")
        source = str(Path(path).expanduser().resolve())
        box["tf_specs"] = {k: s for k, s in box["tf_specs"].items() if s.source != source}
        box["tf_active"] = [t for t in box["tf_active"] if t.spec.source != source]
        _invalidate_frames()
        return {"session": session_dict(), "plugins": _plugins_dict()}

    @app.post("/api/plugins/active")
    def plugins_active(payload: dict = Body(...)) -> dict:
        """Set the active transform list: [{id, cloud?, params?}] — in order."""
        from ..engine.transforms import ActiveTransform

        player = box["player"]
        specs = player.source.channels()
        clouds = [s.name for s in specs if s.kind.value == "pointcloud"]
        default_cloud = clouds[0] if clouds else None
        active = []
        for entry in payload.get("active", []):
            spec = box["tf_specs"].get(entry.get("id"))
            if spec is None:
                raise HTTPException(422, f"unknown transform: {entry.get('id')}")
            cloud = entry.get("cloud") or default_cloud
            if cloud is None:
                raise HTTPException(422, "no point-cloud channel to transform")
            params = dict(spec.params)
            params.update({k: v for k, v in (entry.get("params") or {}).items() if k in params})
            t = ActiveTransform(spec=spec, cloud=cloud, params=params)
            t.bind()
            active.append(t)
        box["tf_active"] = active
        _invalidate_frames()
        return {"session": session_dict(), "plugins": _plugins_dict()}

    @app.get("/api/last")
    def get_last() -> dict:
        """Open params of the most recently opened dataset (any run) — lets the
        viewer offer a reopen prompt when it starts with nothing loaded."""
        from .state import load_last

        return {"params": load_last()}

    # -------------------------------------------------------- session sidecar
    @app.get("/api/state")
    def get_state() -> dict:
        from .state import load_state

        key = box["state_key"]
        return {"key": key, "state": load_state(key) if key else None}

    @app.post("/api/state")
    def post_state(payload: dict = Body(...)) -> dict:
        from .state import save_state

        key = box["state_key"]
        if key:
            save_state(key, payload)
        return {"ok": bool(key)}

    # Web front mounted last (on "/"): /api/* and /docs, registered before, keep priority.
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    app.state.box = box
    return app


def serve(
    source=None,
    *,
    host: str = "127.0.0.1",
    port: int = 8088,
    title: str = "Projector",
    max_points: int | None = None,
    state_key: str | None = None,
    open_params: dict | None = None,
):
    import uvicorn

    app = create_app(source, title=title, max_points=max_points, state_key=state_key, open_params=open_params)
    uvicorn.run(app, host=host, port=port)
    return app
