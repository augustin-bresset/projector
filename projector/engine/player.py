"""`Player` — headless navigation over a `Source`, with no UI dependency.

Owns nothing but position (sequence + frame index) and validated access. Frames are
served statelessly (`frame(seq, index)`) so several clients can scrub independently;
the stored position only anchors the initial view.

Server-side per-frame transforms, in order:
- **accumulation** (`accum=N`): the cloud channels of frames `[index-N, index+N]`
  registered into the current frame by the pose channel; labelings are concatenated
  alongside (absent ones fill with their ignore id) so label coloring stays aligned;
- **decimation** (`max_points`): oversized clouds are subsampled — and every labeling
  aligned to them (`ChannelSpec.of`) with the same indices.
"""

from __future__ import annotations

import numpy as np

from ..core.poses import invert, pose_to_matrix, transform_points
from ..core.source import ChannelKind, Frame, SequenceSpec, Source

MAX_ACCUM = 25


class Player:
    def __init__(self, source: Source, *, max_points: int | None = None) -> None:
        self.source = source
        self.max_points = max_points
        self._sequences = list(source.sequences())
        # An empty source is a valid idle state (viewer launched bare, waiting for
        # /api/open); only frame access is an error then.
        self.seq = self._sequences[0].id if self._sequences else None
        self.index = 0
        specs = source.channels()
        self._clouds = [s.name for s in specs if s.kind is ChannelKind.POINTCLOUD]
        self._labelings = {
            s.name: (s.of, s.labelset.ignore_id if s.labelset else 0)
            for s in specs
            if s.kind is ChannelKind.LABELS and s.of
        }
        poses = [s.name for s in specs if s.kind is ChannelKind.POSE]
        self._pose = poses[0] if poses else None

    @property
    def sequences(self) -> list[SequenceSpec]:
        return list(self._sequences)

    def n_frames(self, seq: str) -> int:
        for s in self._sequences:
            if s.id == seq:
                return s.n_frames
        raise KeyError(f"unknown sequence: {seq!r}")

    def clamp(self, seq: str | None, index: int) -> tuple[str, int]:
        if not self._sequences:
            raise ValueError("no dataset loaded")
        if seq is None or all(s.id != seq for s in self._sequences):
            seq = self._sequences[0].id
        return seq, max(0, min(int(index), self.n_frames(seq) - 1))

    def frame(
        self, seq: str | None = None, index: int | None = None, back: int = 0, fwd: int = 0
    ) -> tuple[str, int, Frame]:
        """Fetch a frame (clamped); remembers the position as the session's current one.

        `back`/`fwd` accumulate that many past/future frames into the current one
        (pose-registered) — asymmetric on purpose: past-only accumulation is the
        online view a robot would have."""
        seq, index = self.clamp(
            seq if seq is not None else self.seq, index if index is not None else self.index
        )
        self.seq, self.index = seq, index
        frame = self.source.frame(seq, index)
        back, fwd = min(int(back), MAX_ACCUM), min(int(fwd), MAX_ACCUM)
        if back > 0 or fwd > 0:
            frame = self._accumulate(seq, index, back, fwd, frame)
        return seq, index, self._decimate(frame)

    def trajectory(self, seq: str) -> np.ndarray | None:
        """Ego positions `(N, 3)` for a sequence, or None. Sources may provide an
        efficient `trajectory(seq)`; the fallback walks the pose channel frame by
        frame (fine in memory, avoided for disk-backed sources)."""
        seq, _ = self.clamp(seq, 0)
        impl = getattr(self.source, "trajectory", None)
        if impl is not None:
            out = impl(seq)
            return None if out is None else np.asarray(out, dtype=np.float64)
        return None

    def timeline(self, seq: str) -> dict | None:
        """Per-channel event timestamps (+ tick times) when the source knows them."""
        seq, _ = self.clamp(seq, 0)
        impl = getattr(self.source, "timeline", None)
        return impl(seq) if impl is not None else None

    # ------------------------------------------------------------- accumulation
    def _accumulate(self, seq: str, index: int, back: int, fwd: int, frame: Frame) -> Frame:
        if self._pose is None or not self._clouds:
            return frame
        ref_pose = frame.channels.get(self._pose)
        if ref_pose is None:
            return frame
        p_ref_inv = invert(pose_to_matrix(ref_pose))
        n = self.n_frames(seq)
        window = [j for j in range(max(0, index - back), min(n, index + fwd + 1)) if j != index]

        # One chunk list per cloud, and one per labeling — kept index-aligned with its
        # cloud's list so the concatenations cannot drift apart.
        parts = {c: [np.asarray(frame.channels[c])] for c in self._clouds if c in frame.channels}
        labs = {
            name: [frame.channels.get(name)] for name, (c_of, _ig) in self._labelings.items() if c_of in parts
        }
        for j in window:
            f = self.source.frame(seq, j)
            pj = f.channels.get(self._pose)
            if pj is None:
                continue  # cannot register this neighbor
            T = p_ref_inv @ pose_to_matrix(pj)
            appended = set()
            for c, chunks in parts.items():
                p = f.channels.get(c)
                if p is None or len(p) == 0:
                    continue
                moved = np.asarray(p, dtype=np.float64).copy()
                moved[:, :3] = transform_points(moved[:, :3], T)
                chunks.append(moved.astype(np.float32))
                appended.add(c)
            for name in labs:
                if self._labelings[name][0] in appended:
                    labs[name].append(f.channels.get(name))

        channels = dict(frame.channels)
        for c, chunks in parts.items():
            if len(chunks) > 1:
                channels[c] = np.concatenate([np.asarray(k, dtype=np.float32) for k in chunks], axis=0)
        for name, chunks in labs.items():
            c_of, ignore = self._labelings[name]
            cloud_chunks = parts[c_of]
            if len(cloud_chunks) <= 1:
                continue
            sized = [
                np.asarray(k)
                if k is not None and len(k) == len(cp)
                else np.full(len(cp), ignore, dtype=np.int32)
                for k, cp in zip(chunks, cloud_chunks, strict=True)
            ]
            channels[name] = np.concatenate(sized)
        return Frame(channels=channels, timestamps=frame.timestamps, indices=frame.indices)

    # ------------------------------------------------------------- decimation
    def _decimate(self, frame: Frame) -> Frame:
        if not self.max_points:
            return frame
        for cloud in self._clouds:
            arr = frame.channels.get(cloud)
            if arr is None or len(arr) <= self.max_points:
                continue
            n = len(arr)
            idx = np.linspace(0, n - 1, self.max_points).astype(np.int64)
            frame.channels[cloud] = arr[idx]
            for lab, (of, _ignore) in self._labelings.items():
                la = frame.channels.get(lab)
                if of == cloud and la is not None and len(la) == n:
                    frame.channels[lab] = np.asarray(la)[idx]
        return frame
