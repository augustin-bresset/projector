"""Per-dataset session sidecars — layout, bookmarks, position, display settings.

Stored under the user data dir (XDG), keyed by the dataset identity: the viewer
never writes inside a dataset directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    d = Path(base) / "projector"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sessions_dir() -> Path:
    d = _data_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(key: str) -> Path:
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return _sessions_dir() / f"{digest}.json"


def load_state(key: str) -> dict | None:
    p = _path(key)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save_state(key: str, state: dict) -> None:
    _path(key).write_text(json.dumps(state))


def _last_path() -> Path:
    return _data_dir() / "last_opened.json"


def load_last() -> dict | None:
    """The open params of the most recently opened (real, path-backed) dataset —
    across all projector runs, not per-dataset. Used to offer a reopen prompt
    when the viewer starts with nothing loaded."""
    p = _last_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save_last(params: dict) -> None:
    _last_path().write_text(json.dumps(params))
