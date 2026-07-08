"""Desktop app: the web front rendered in a native window (Spotify/Electron style).

Starts the FastAPI server in a background thread, then opens the OS webview
(pywebview) on the local URL. If no native webview backend is available, it falls
back to opening the URL in the default browser and keeps serving.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
import webbrowser


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_ready(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.05)


def _preferred_gui() -> str | None:
    """Pick a webview backend explicitly to skip pywebview's probing (and its noisy
    GTK import traceback when GTK bindings are absent). Returns None to auto-select."""
    try:
        import qtpy  # noqa: F401  (provided by the `app` extra → Qt WebEngine)

        return "qt"
    except Exception:
        return None


def run_desktop(
    source,
    *,
    title: str = "Projector",
    host: str = "127.0.0.1",
    port: int = 0,
    width: int = 1480,
    height: int = 880,
    quiet: bool = False,
    max_points: int | None = None,
    state_key: str | None = None,
    open_params: dict | None = None,
) -> None:
    import uvicorn

    from .app import create_app

    app = create_app(source, title=title, max_points=max_points, state_key=state_key, open_params=open_params)
    if port == 0:
        port = _free_port(host)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    _wait_ready(host, port)
    url = f"http://{host}:{port}"

    try:
        gui = _preferred_gui()
        if gui == "qt":
            # Qt's event loop swallows SIGINT; restore the default so Ctrl+C quits at once.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if quiet:
                os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--log-level=2")
                os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

        import webview

        webview.create_window(title, url, width=width, height=height)
        webview.start(gui=gui)  # blocks until the window is closed
    except Exception as exc:
        # pywebview missing, or no native backend (GTK/Qt) — fall back to the browser.
        print(f"[projector] native window unavailable ({type(exc).__name__}); opening browser.")
        print(
            "[projector] for a native window, install a webview backend "
            "(system GTK+WebKit, or `pip install 'pywebview[qt]'`)."
        )
        print(f"[projector] serving at {url} — press Ctrl+C to stop.")
        webbrowser.open(url)
        try:
            while not server.should_exit:
                time.sleep(0.3)
        except KeyboardInterrupt:
            pass

    server.should_exit = True
