"""Projector — interactive viewer for multi-sensor robotics sequences.

Also the home of the shared three.js point-cloud engine. ``web_engine_dir()``
returns the path to that engine so sibling tools (toaster, apairo_visu studio)
serve it straight from projector's install instead of keeping a private copy —
one fix here reaches every tool.
"""

from importlib.resources import files

from .adapters.array_source import ArraySource
from .core import ChannelKind, ChannelSpec, Frame, LabelClass, LabelSet, SequenceSpec, Source

__version__ = "0.1.0"


def web_dir() -> str:
    """Filesystem path to projector's vanilla web front.

    Holds ``index.html``, ``src/`` and ``vendor/three``. Consumers wanting the
    whole viewer, or projector's vendored ``three``, serve from here.
    """
    return str(files("projector").joinpath("web"))


def web_engine_dir() -> str:
    """Filesystem path to the shipped three.js point-cloud engine.

    (``octree.js`` / ``viewer.js`` / ``overlays.js`` / ``octree-worker.js``.)
    Mount it statically and provide the ``three`` / ``three/addons/`` importmap —
    the engine imports only ``three`` and its own siblings.
    """
    return str(files("projector").joinpath("web", "src", "engine"))


__all__ = [
    "ArraySource",
    "ChannelKind",
    "ChannelSpec",
    "Frame",
    "LabelClass",
    "LabelSet",
    "SequenceSpec",
    "Source",
    "__version__",
    "web_dir",
    "web_engine_dir",
]
