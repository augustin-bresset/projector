"""Projector's input protocol — the central point of genericity.

A `Source` is a set of *sequences*; each sequence yields, at every time index, a
`Frame` = a pack of named channels (numpy arrays), each typed by a `ChannelKind`.
That is all the viewer requires. apairo, files, in-memory arrays… are just ways to
produce a `Source`.

Labelings are first-class channels (`ChannelKind.LABELS`): a `(N,)` integer array
aligned to a point-cloud channel (`ChannelSpec.of`). One cloud can carry any number
of them (ground truth, model predictions, corridors…) — the front colors by one,
overlays several, or compares two (confusion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from .labels import LabelSet


class ChannelKind(Enum):
    """Nature of a channel — drives the view that shows it and how it is colored."""

    POINTCLOUD = "pointcloud"  # (N, 3) or (N, 3+C): [x, y, z, ...]
    IMAGE = "image"  # (H, W) or (H, W, C) uint8
    POSE = "pose"  # (4, 4), (3, 4), (7,) [x,y,z, qx,qy,qz,qw], or (3,): ego world pose
    LABELS = "labels"  # (N,) int, aligned to the cloud channel named by `of`
    SCALAR = "scalar"  # other array (reserved / extensible)

    def __repr__(self) -> str:  # compact display
        return f"ChannelKind.{self.name}"


@dataclass(frozen=True)
class ChannelSpec:
    """Channel metadata.

    `of` is only meaningful for LABELS: the point-cloud channel the labeling is
    aligned to (same N on every frame). `labelset` gives ids their names/colors.
    `placement` is the sensor's pose in the ego frame — `(4, 4)`, `(7,)` or `(3,)` —
    or `None` for non-sensor channels.
    """

    name: str
    kind: ChannelKind
    dtype: np.dtype | None = None
    shape: tuple[int | None, ...] | None = None
    of: str | None = None
    labelset: LabelSet | None = None
    placement: np.ndarray | None = None


@dataclass(frozen=True)
class SequenceSpec:
    """One replayable sequence: an id and its length in reference-clock ticks."""

    id: str
    n_frames: int


@dataclass
class Frame:
    """One time step: named channels, plus optional per-channel timestamps.

    `timestamps` is `None` for a synchronous source. On an async rig each present
    channel maps to its own timestamp, and a channel may be absent from a frame
    (the front keeps its last value on screen).
    """

    channels: dict[str, np.ndarray]
    timestamps: dict[str, float] | None = None

    def __getitem__(self, key: str) -> np.ndarray:
        return self.channels[key]

    def __contains__(self, key: str) -> bool:
        return key in self.channels

    def keys(self):
        return self.channels.keys()


@runtime_checkable
class Source(Protocol):
    """Sequences + channel description + indexed access to frames."""

    def sequences(self) -> list[SequenceSpec]: ...

    def channels(self) -> list[ChannelSpec]: ...

    def frame(self, seq: str, index: int) -> Frame: ...


def channels_of_kind(source_or_specs, kind: ChannelKind) -> list[str]:
    """Names of the channels of a given `ChannelKind`, from a `Source` or a spec list."""
    specs = source_or_specs.channels() if hasattr(source_or_specs, "channels") else source_or_specs
    return [s.name for s in specs if s.kind == kind]


def labelings_of(source_or_specs, cloud: str) -> list[ChannelSpec]:
    """LABELS channels aligned to a given cloud channel."""
    specs = source_or_specs.channels() if hasattr(source_or_specs, "channels") else source_or_specs
    return [s for s in specs if s.kind == ChannelKind.LABELS and s.of == cloud]


@dataclass
class _SeqIndex:
    """Internal helper: (seq, index) navigation state with clamping."""

    sequences: list[SequenceSpec] = field(default_factory=list)

    def clamp(self, seq: str, index: int) -> tuple[str, int]:
        by_id = {s.id: s for s in self.sequences}
        if seq not in by_id:
            if not self.sequences:
                raise ValueError("source has no sequences")
            seq = self.sequences[0].id
        n = by_id[seq].n_frames
        return seq, max(0, min(index, n - 1))
