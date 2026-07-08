"""Projector CLI.

    projector                                       # empty viewer: Open… a dataset from the UI
    projector demo                                  # desktop app on the synthetic source
    projector demo --serve                          # same app in the browser → http://127.0.0.1:8088
    projector /path/to/dataset                      # apairo dataset (auto-detected via .apairo)
    projector /path/to/ds --channels lidar,camera,pose --reference lidar --tolerance 0.15
    projector /path/to/ds --sequences seq_a,seq_b --every 5 --max-points 120000

An async apairo dataset without --reference is served in events mode: each frame is
one sensor event, channels refresh at their own rate (rerun-style).
Extras: `app` (desktop), `api` (browser/--serve), `apairo` (the adapter).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _make_source(args):
    """Build the requested `Source`, or (None, ...) if the invocation is invalid.

    Returns (source, title, open_params); `open_params` is non-None for apairo
    datasets and lets the server's /api/sync re-open with another time model.
    """
    if args.path is None:
        from .adapters.array_source import ArraySource

        return ArraySource([], {}), "Projector", None  # empty: open a dataset from the UI
    if args.path == "demo":
        from .demo import make_demo_source

        return make_demo_source(), "Projector — demo", None
    if args.path and (args.adapter == "apairo" or (Path(args.path) / ".apairo").is_dir()):
        from .adapters.apairo_source import ApairoSource

        keys = [k.strip() for k in args.channels.split(",")] if args.channels else None
        seqs = [s.strip() for s in args.sequences.split(",")] if args.sequences else None
        params = {
            "path": str(args.path),
            "channels": keys,
            "reference": args.reference,
            "tolerance": args.tolerance,
            "sequences": seqs,
            "every": args.every,
            "upright": not args.raw,
        }
        src = ApairoSource.from_path(
            args.path,
            keys=keys,
            reference=args.reference,
            tolerance=args.tolerance,
            sequences=seqs,
            every=args.every,
            upright=not args.raw,
        )
        return src, f"Projector — {Path(args.path).name}", params
    return None, None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="projector", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", nargs="?", help="'demo', or an apairo dataset root / sequence dir")
    parser.add_argument("--adapter", choices=["apairo"], help="input adapter (auto-detected via .apairo)")
    parser.add_argument("--channels", help="comma-separated channels to load (default: all)")
    parser.add_argument("--reference", help="reference channel to synchronize an async dataset")
    parser.add_argument("--tolerance", type=float, default=0.1, help="sync tolerance in seconds")
    parser.add_argument("--sequences", help="comma-separated sequence ids (default: all)")
    parser.add_argument("--every", type=int, default=1, help="keep one frame out of N")
    parser.add_argument(
        "--raw", action="store_true", help="keep clouds in their native sensor frame (skip the mount TF)"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="decimate clouds above N points server-side (labelings follow)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="run the headless web server (open it in a browser) instead of the desktop app",
    )
    parser.add_argument("--host", default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=int, default=8088, help="server port (--serve mode)")
    parser.add_argument(
        "--quiet", action="store_true", help="desktop app: hush QtWebEngine/Chromium console logs"
    )
    args = parser.parse_args(argv)

    try:
        source, title, open_params = _make_source(args)
    except ImportError:
        parser.error("apairo is not installed — install the `apairo` extra")
        return 2
    except (ValueError, OSError, KeyError) as e:
        parser.error(str(e))
        return 2

    if source is None:
        parser.error("pass 'demo' or an apairo dataset path (--adapter apairo to force)")
        return 2

    state_key = None
    if args.path == "demo":
        state_key = "demo"
    elif args.path:
        state_key = str(Path(args.path).expanduser().resolve())

    if args.serve:
        from .server.app import serve

        print(f"Projector → http://{args.host}:{args.port}  (web front + API, docs at /docs)")
        serve(
            source,
            host=args.host,
            port=args.port,
            title=title,
            max_points=args.max_points,
            state_key=state_key,
            open_params=open_params,
        )
        return 0

    from .server.desktop import run_desktop

    run_desktop(
        source,
        title=title,
        quiet=args.quiet,
        max_points=args.max_points,
        state_key=state_key,
        open_params=open_params,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
