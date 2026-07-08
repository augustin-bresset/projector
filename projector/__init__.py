"""Projector — interactive viewer for multi-sensor robotics sequences."""

from .adapters.array_source import ArraySource
from .core import ChannelKind, ChannelSpec, Frame, LabelClass, LabelSet, SequenceSpec, Source

__version__ = "0.1.0"

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
]
