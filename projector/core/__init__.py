"""Numpy-only domain: source protocol, label sets, poses. No UI dependency."""

from .labels import LabelClass, LabelSet
from .source import (
    ChannelKind,
    ChannelSpec,
    Frame,
    SequenceSpec,
    Source,
    channels_of_kind,
    labelings_of,
)

__all__ = [
    "ChannelKind",
    "ChannelSpec",
    "Frame",
    "LabelClass",
    "LabelSet",
    "SequenceSpec",
    "Source",
    "channels_of_kind",
    "labelings_of",
]
