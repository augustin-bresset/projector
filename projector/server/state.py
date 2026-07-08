"""Per-dataset session sidecars — layout, bookmarks, position, display settings.

Stored under the user data dir (XDG), keyed by the dataset identity: the viewer
never writes inside a dataset directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sessions_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    d = Path(base) / "projector" / "sessions"
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
