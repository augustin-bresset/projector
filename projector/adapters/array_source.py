"""`ArraySource` — a `Source` built from in-memory numpy arrays.

The default input, with no dependency: handy for demos, tests, and any pipeline that
already produces arrays. No hidden I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..core.source import ChannelSpec, Frame, SequenceSpec, Source


class ArraySource(Source):
    """In-memory multi-sequence source.

    Parameters
    ----------
    specs:
        Channel description (`ChannelSpec`). Order is preserved.
    sequences:
        `{seq_id: [frame, ...]}` where a frame is `{channel_name: np.ndarray}`.
        A plain list of frames is accepted as a single sequence named "0".
    timestamps:
        Optional `{seq_id: [per-frame {channel: ts}, ...]}`. `None` = synchronous.
    """

    def __init__(
        self,
        specs: Sequence[ChannelSpec],
        sequences: Mapping[str, Sequence[Mapping[str, np.ndarray]]] | Sequence[Mapping[str, np.ndarray]],
        timestamps: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
    ) -> None:
        self._specs = list(specs)
        if not isinstance(sequences, Mapping):
            sequences = {"0": sequences}
        self._sequences = {str(k): [dict(f) for f in v] for k, v in sequences.items()}
        self._timestamps = timestamps

        names = {s.name for s in self._specs}
        for seq, frames in self._sequences.items():
            for i, fr in enumerate(frames):
                unknown = set(fr.keys()) - names
                if unknown:
                    raise ValueError(f"sequence {seq}, frame {i}: undeclared channels {sorted(unknown)}")

    def sequences(self) -> list[SequenceSpec]:
        return [SequenceSpec(k, len(v)) for k, v in self._sequences.items()]

    def channels(self) -> list[ChannelSpec]:
        return list(self._specs)

    def frame(self, seq: str, index: int) -> Frame:
        frames = self._sequences[str(seq)]
        ts = None
        if self._timestamps is not None:
            ts = dict(self._timestamps[str(seq)][index])
        return Frame(channels=dict(frames[index]), timestamps=ts)

    def trajectory(self, seq: str) -> np.ndarray | None:
        """Ego positions (N, 3) from the first POSE channel — in-memory walk."""
        from ..core.poses import pose_to_matrix
        from ..core.source import ChannelKind

        poses = [s.name for s in self._specs if s.kind is ChannelKind.POSE]
        if not poses:
            return None
        key = poses[0]
        out = []
        for fr in self._sequences[str(seq)]:
            p = fr.get(key)
            if p is not None:
                out.append(pose_to_matrix(np.asarray(p))[:3, 3])
        return np.asarray(out) if out else None
