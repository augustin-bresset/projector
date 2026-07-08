"""Filesystem browsing for the "Open dataset…" flow — directories only.

A directory is *openable* when it carries a `.apairo` marker (dataset root or lone
sequence dir). `probe` peeks inside one so the front can offer channel checkboxes
and, on an async dataset, a reference-channel picker before opening.
"""

from __future__ import annotations

from pathlib import Path


def list_dir(path: str | None = None, files: str | None = None) -> dict:
    """Subdirectories of `path` (default: cwd), each flagged if it is an apairo
    dataset. With `files` (a suffix, e.g. ".py"), matching files are listed too."""
    p = Path(path).expanduser() if path else Path.cwd()
    p = p.resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"not a directory: {p}")
    dirs, file_list = [], []
    for child in sorted(p.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            dirs.append({"name": child.name, "is_dataset": (child / ".apairo").is_dir()})
        elif files and child.suffix == files:
            file_list.append(child.name)
    return {
        "path": str(p),
        "parent": None if p == p.parent else str(p.parent),
        "is_dataset": (p / ".apairo").is_dir(),
        "dirs": dirs,
        "files": file_list,
    }


def probe(path: str) -> dict:
    """What the open form needs: channels, synchronicity, sequences of a dataset dir.

    On a dataset root, synchronicity is read from the *first sequence* — the root
    view reports synchronous even when every sequence is an async event stream, and
    the adapter opens sequences individually anyway.
    """
    import apairo  # lazy — the `apairo` extra

    p = Path(path).expanduser().resolve()
    ds = apairo.RawDataset(str(p))
    try:
        sequences = list(ds.sequence_ids)
    except Exception:  # noqa: BLE001 — a lone sequence dir
        sequences = []
    sync_probe = apairo.RawDataset(str(p / sequences[0])) if sequences else ds
    return {
        "path": str(p),
        "channels": list(getattr(ds, "keys", [])),
        "is_synchronous": bool(getattr(sync_probe, "is_synchronous", True)),
        "sequences": sequences,
    }
