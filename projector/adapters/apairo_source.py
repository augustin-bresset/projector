"""**Optional** adapter: an apairo dataset (root or lone sequence dir) → `Source`.

apairo is only imported here — never by core/engine. The class duck-types on any
apairo-like object (`is_synchronous`, `keys`, `__len__`, `__getitem__` → object with
`.data` / `.timestamp`), so tests can feed it fakes.

Two time models, per sequence:
- **synchronized** (`reference=` given, or the dataset is already synchronous):
  every index yields a full frame (all channels present);
- **events** (async dataset, no reference): every index is one sensor event — the
  frame carries only the channel that ticked, with its timestamp, and the front
  keeps the last value of the others (rerun-style).

Labelings are auto-detected in synchronized mode: a `(N,)` integer channel whose
length matches a cloud in the same samples becomes `ChannelKind.LABELS` with a
generated palette.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..core.labels import LabelClass, LabelSet
from ..core.source import ChannelKind, ChannelSpec, Frame, SequenceSpec

# Distinct, colorblind-friendly-ish palette cycled for auto-generated labelsets.
_PALETTE = [
    (130, 130, 130),
    (50, 200, 80),
    (200, 60, 60),
    (70, 130, 220),
    (240, 180, 40),
    (170, 100, 220),
    (80, 210, 200),
    (240, 120, 60),
    (150, 200, 60),
    (220, 100, 160),
    (100, 160, 240),
    (180, 150, 100),
]

_SCAN_FRAMES = 25  # frames sampled to classify channels
_MAX_LABEL_IDS = 32  # more unique ints than this → SCALAR, not LABELS
_SPARSE_RATIO = 0.5  # channel with < ratio × reference events → per-tick overlay


def _kind_of(arr: np.ndarray) -> ChannelKind:
    """Guess the `ChannelKind` of an apairo channel from the array shape."""
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return ChannelKind.IMAGE
    if arr.shape in ((4, 4), (3, 4)) or arr.shape == (7,):
        return ChannelKind.POSE
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return ChannelKind.POINTCLOUD
    return ChannelKind.SCALAR  # (N,) measures and labelings — refined by _detect_labels

    # NOTE: (H, W) grayscale images classify as SCALAR; pass them through an explicit
    # kinds= override once the need shows up.


# Conventional frame names used to auto-resolve the lidar mount TF (apairo_rr's
# heuristic): most rigs mount the lidar tilted, so a raw scan "looks up" instead
# of forward. First calibration-tree match wins.
_LIDAR_FRAME_HINTS = ("os_sensor", "velodyne", "hesai", "rslidar", "livox", "os_lidar")
_BASE_FRAME_HINTS = ("base_link", "base_footprint")


def _auto_mount(ds) -> np.ndarray | None:
    """Static mount TF `<lidar frame> -> <base frame>` from the calibration tree,
    or None (no calibration / no conventional frames) — cloud stays native then."""
    cal = getattr(ds, "calibration", None)
    if not cal:
        return None
    try:
        nodes: set[str] = set()
        for edge in cal.keys():
            head, sep, tail = str(edge).partition("_to_")
            if sep:
                nodes.add(head)
                nodes.add(tail)
    except Exception:  # noqa: BLE001 — never let auto-resolution break the viewer
        return None
    base = next((b for b in _BASE_FRAME_HINTS if b in nodes), None)
    if base is None:
        return None
    for lf in _LIDAR_FRAME_HINTS:
        if lf in nodes and lf != base:
            try:
                return np.asarray(cal.get_tf(lf, base), dtype=np.float64)
            except Exception:  # noqa: BLE001 — no path: try the next candidate
                continue
    return None


def _auto_labelset(values: np.ndarray) -> LabelSet:
    ids = sorted(int(v) for v in np.unique(values))
    classes = [LabelClass(i, str(i), _PALETTE[k % len(_PALETTE)]) for k, i in enumerate(ids)]
    return LabelSet(classes, ignore_id=ids[0] if ids else 0)


class ApairoSource:
    """Wraps apairo sequences into a `Source`. Use `from_path` for the common case."""

    def __init__(
        self,
        views: dict[str, object],
        *,
        keys: list[str] | None = None,
        dirs: dict[str, str] | None = None,
    ) -> None:
        if not views:
            raise ValueError("no sequences to serve")
        self._views = views  # seq id → apairo dataset (one per sequence)
        self._keys = keys
        self._dirs = dirs or {}  # seq id → sequence dir (enables fast trajectory reads)
        self._traj_cache: dict[str, np.ndarray | None] = {}
        self._timeline_cache: dict[str, dict | None] = {}
        self.mount: np.ndarray | None = None  # static lidar mount TF (upright clouds)
        self._specs = self._classify()
        self._names = {s.name for s in self._specs}
        self._cloud_names = {s.name for s in self._specs if s.kind is ChannelKind.POINTCLOUD}

    # ------------------------------------------------------------- Source protocol
    def sequences(self) -> list[SequenceSpec]:
        return [SequenceSpec(sid, len(ds)) for sid, ds in self._views.items()]

    def channels(self) -> list[ChannelSpec]:
        return list(self._specs)

    def frame(self, seq: str, index: int) -> Frame:
        s = self._views[seq][index]
        channels = {
            k: np.asarray(v)
            for k, v in s.data.items()
            if k in self._names and getattr(v, "dtype", None) is not None
        }
        if self.mount is not None:
            # Lift clouds upright with the static mount TF (sensor → base frame).
            R, t = self.mount[:3, :3], self.mount[:3, 3]
            for k in self._cloud_names:
                p = channels.get(k)
                if p is not None and p.ndim == 2 and p.shape[1] >= 3:
                    p = np.array(p, dtype=np.float32, copy=True)
                    p[:, :3] = p[:, :3] @ R.T.astype(np.float32) + t.astype(np.float32)
                    channels[k] = p
        ts = getattr(s, "timestamp", None)
        per_channel = getattr(s, "channel_timestamps", None)  # _SparseSync samples
        if per_channel is not None:
            timestamps = {k: float(per_channel[k]) for k in channels if k in per_channel}
        else:
            timestamps = None if ts is None else {k: float(ts) for k in channels}
        per_index = getattr(s, "channel_indices", None)  # per-sensor event counters
        indices = None if per_index is None else {k: int(per_index[k]) for k in channels if k in per_index}
        return Frame(channels=channels, timestamps=timestamps, indices=indices)

    def timeline(self, seq: str) -> dict | None:
        """Per-channel event timestamps + tick times of a sequence (footer tracks).

        `{"channels": {name: (N,) ts}, "ticks": (M,) tick times | None}` — ticks map
        the player's frame index to time on an event timeline; None when frames are
        resampled (synchronized) and the index/time mapping is not 1:1.
        """
        if seq in self._timeline_cache:
            return self._timeline_cache[seq]
        out: dict | None = None
        view = self._views[seq]
        if isinstance(view, _EventTimeline):
            out = {"channels": view.channel_times(), "ticks": view.tick_times()}
        else:
            d = self._dirs.get(seq)
            if d is not None:
                import apairo

                ds = apairo.RawDataset(d, keys=self._keys) if self._keys else apairo.RawDataset(d)
                ts = getattr(ds, "timestamps", None)
                if ts:
                    out = {"channels": {k: np.asarray(v) for k, v in ts.items()}, "ticks": None}
        self._timeline_cache[seq] = out
        return out

    def trajectory(self, seq: str) -> np.ndarray | None:
        """Ego positions (N, 3) of the pose channel, cached per sequence.

        With a known sequence dir, reads a pose-only apairo dataset (one tiny file
        per event) instead of walking full frames; falls back to the view otherwise.
        """
        if seq in self._traj_cache:
            return self._traj_cache[seq]
        from ..core.poses import pose_to_matrix

        poses = [s.name for s in self._specs if s.kind is ChannelKind.POSE]
        out = None
        if poses:
            key = poses[0]
            d = self._dirs.get(seq)
            if d is not None:
                import apairo

                walk = apairo.RawDataset(str(d), keys=[key])
            else:
                walk = self._views[seq]
            pts = []
            for i in range(len(walk)):
                v = walk[i].data.get(key)
                if v is not None:
                    pts.append(pose_to_matrix(np.asarray(v))[:3, 3])
            out = np.asarray(pts) if pts else None
        self._traj_cache[seq] = out
        return out

    # ------------------------------------------------------------- classification
    def _classify(self) -> list[ChannelSpec]:
        """Discover each channel's kind/shape by scanning the first frames of the first
        sequence. In events mode channels appear one per sample, so we keep scanning
        until every requested key was seen (or the scan budget runs out).
        """
        ds = next(iter(self._views.values()))
        requested = list(self._keys) if self._keys else list(getattr(ds, "keys", []))
        found: dict[str, ChannelSpec] = {}
        samples: list[dict] = []
        budget = _SCAN_FRAMES if requested else min(len(ds), _SCAN_FRAMES)
        # Sparse-overlay views contribute one merged sample per sparse channel (its
        # first real event), so sparse labelings classify even when the scan window
        # never crosses one.
        samples.extend(getattr(ds, "extra_samples", list)())
        for data in samples:
            for k, v in data.items():
                if k in found or (requested and k not in requested):
                    continue
                arr = np.asarray(v)
                if arr.dtype == object or arr.ndim == 0:
                    continue
                found[k] = ChannelSpec(k, _kind_of(arr), arr.dtype, tuple(arr.shape))
        for i in range(min(len(ds), max(budget, _SCAN_FRAMES))):
            data = ds[i].data
            samples.append(data)
            for k, v in data.items():
                if k in found or (requested and k not in requested):
                    continue
                arr = np.asarray(v)
                if arr.dtype == object or arr.ndim == 0:
                    continue
                found[k] = ChannelSpec(k, _kind_of(arr), arr.dtype, tuple(arr.shape))
            if requested and len(found) == len(requested):
                break

        if requested:
            dropped = [k for k in requested if k not in found]
            if dropped:
                warnings.warn(
                    f"apairo: channels never seen in the first samples, skipped: {dropped}", stacklevel=2
                )
            order = [k for k in requested if k in found]
        else:
            order = sorted(found)
        specs = [found[k] for k in order]
        return self._detect_labels(specs, samples)

    @staticmethod
    def _detect_labels(specs: list[ChannelSpec], samples: list[dict]) -> list[ChannelSpec]:
        """Promote SCALAR channels to LABELS when they are integer `(N,)` arrays whose
        length matches exactly one cloud across the sampled frames."""
        clouds = [s.name for s in specs if s.kind is ChannelKind.POINTCLOUD]
        if not clouds:
            return specs
        out: list[ChannelSpec] = []
        for spec in specs:
            if (
                spec.kind is not ChannelKind.SCALAR
                or spec.dtype is None
                or not np.issubdtype(spec.dtype, np.integer)
            ):
                out.append(spec)
                continue
            # Candidate clouds: same length in every sampled frame where both appear.
            matches, values = set(clouds), []
            seen = 0
            for data in samples:
                if spec.name not in data:
                    continue
                arr = np.asarray(data[spec.name])
                if arr.ndim != 1:
                    matches.clear()
                    break
                seen += 1
                values.append(arr)
                matches = {c for c in matches if c in data and len(np.asarray(data[c])) == len(arr)}
            if not seen or not matches:
                out.append(spec)
                continue
            allv = np.concatenate(values)
            if len(np.unique(allv)) > _MAX_LABEL_IDS:
                out.append(spec)  # too many ids to be classes — a measure
                continue
            of = sorted(matches)[0]
            out.append(
                ChannelSpec(
                    spec.name,
                    ChannelKind.LABELS,
                    spec.dtype,
                    spec.shape,
                    of=of,
                    labelset=_auto_labelset(allv),
                )
            )
        return out

    # ------------------------------------------------------------- opening
    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        keys: list[str] | None = None,
        reference: str | None = None,
        method: str = "nearest",
        tolerance: float = 0.1,
        sequences: list[str] | None = None,
        every: int = 1,
        upright: bool = True,
    ) -> ApairoSource:
        """Open a dataset root (every sequence) or a lone sequence directory.

        `reference` synchronizes async sequences (full frames); without it an async
        sequence is served in events mode. `sequences` restricts to named ids;
        `every` keeps one frame out of N (windowing huge sequences).

        Sparse channels (few events vs the reference — e.g. a `ground_truth` labeled
        on a handful of frames) are excluded from `synchronize()` — apairo drops any
        reference tick a channel misses, so one sparse channel would collapse the
        whole timeline — and overlaid per tick instead: present in a frame only when
        an event falls within `tolerance` of it.
        """
        import apairo  # lazy import — install the `apairo` extra

        root = Path(path).expanduser()
        try:
            ids = list(apairo.RawDataset(str(root)).sequence_ids)
        except Exception:  # noqa: BLE001 — a lone sequence dir has no sequence_ids
            ids = []
        seq_dirs = {sid: root / sid for sid in ids} if ids else {root.name: root}
        if sequences:
            unknown = [s for s in sequences if s not in seq_dirs]
            if unknown:
                raise ValueError(f"unknown sequences {unknown}; available: {sorted(seq_dirs)}")
            seq_dirs = {sid: seq_dirs[sid] for sid in sequences}

        views: dict[str, object] = {}
        was_async = False
        for sid, d in seq_dirs.items():
            ds = apairo.RawDataset(str(d), keys=keys) if keys else apairo.RawDataset(str(d))
            was_async = was_async or not getattr(ds, "is_synchronous", True)
            if not getattr(ds, "is_synchronous", True) and reference is None:
                # Event timeline (the default for async rigs): no synchronization,
                # no dropped events — co-timestamped events grouped per tick.
                ds = cls._event_timeline(apairo, str(d), ds, every=every)
            else:
                if not getattr(ds, "is_synchronous", True):
                    ds = cls._synchronize_with_sparse(
                        apairo, str(d), ds, reference=reference, method=method, tolerance=tolerance
                    )
                if every > 1:
                    ds = ds.filter(list(range(0, len(ds), every)))
            views[sid] = ds
            mode = "sync" if getattr(ds, "is_synchronous", True) else "events"
            print(f"[projector] {sid}: {len(ds)} frames ({mode})")
        src = cls(views, keys=keys, dirs={sid: str(d) for sid, d in seq_dirs.items()})
        src.was_async = was_async  # the dataset can be resampled onto a reference clock
        if upright:
            # Lift tilted lidars upright with the static mount TF from the
            # calibration tree (apairo_rr behavior; pass upright=False / --raw to
            # keep the native sensor frame).
            first_dir = next(iter(seq_dirs.values()))
            try:
                src.mount = _auto_mount(apairo.RawDataset(str(first_dir)))
            except Exception:  # noqa: BLE001 — calibration is best-effort
                src.mount = None
            if src.mount is not None:
                print("[projector] mount TF applied (upright clouds); use --raw to disable")
        return src

    @staticmethod
    def _event_timeline(apairo, seq_dir: str, ds, *, every: int = 1):
        """One `_EventTimeline` over per-channel datasets (index = channel event index)."""
        channels = {
            k: (apairo.RawDataset(seq_dir, keys=[k]), np.asarray(v, dtype=np.float64))
            for k, v in ds.timestamps.items()
        }
        return _EventTimeline(channels, every=every)

    @staticmethod
    def _synchronize_with_sparse(apairo, seq_dir: str, ds, *, reference: str, method: str, tolerance: float):
        """Synchronize the dense channels; wrap sparse ones as per-tick overlays."""
        all_ts = {k: np.asarray(v) for k, v in ds.timestamps.items()}
        if reference not in all_ts:
            raise ValueError(f"unknown reference channel {reference!r}; have {sorted(all_ts)}")
        ref_n = len(all_ts[reference])
        sparse = [k for k, v in all_ts.items() if k != reference and len(v) < _SPARSE_RATIO * ref_n]
        dense = [k for k in all_ts if k not in sparse]

        dense_ds = apairo.RawDataset(seq_dir, keys=dense)
        sync = dense_ds.synchronize(reference=reference, method=method, tolerance=tolerance)
        if not sparse:
            return sync
        overlays = {k: (apairo.RawDataset(seq_dir, keys=[k]), all_ts[k]) for k in sparse}
        print(f"[projector] sparse channels overlaid per tick: {sparse}")
        return _SparseSync(sync, overlays, tolerance)


class _SparseSyncSample:
    """Duck-typed apairo sample: `.data`, `.timestamp`, plus per-channel timestamps
    and per-channel event indices (each sensor's own counter, apairo_rr-style)."""

    def __init__(self, data: dict, timestamp, channel_timestamps: dict, channel_indices: dict | None = None):
        self.data = data
        self.timestamp = timestamp
        self.channel_timestamps = channel_timestamps
        self.channel_indices = channel_indices or {}


class _EventTimeline:
    """Async events grouped by (quantized) timestamp — the rerun-style timeline.

    Index `i` is the i-th distinct event time of the sequence; the frame carries
    every channel that ticked at that instant. Nothing is synchronized and nothing
    is dropped: channels refresh at their own rate and the front keeps the last
    value of the others. Because a labeling derived from a scan shares the scan's
    timestamp (apairo's `timestamps_from`), it rides in the same frame as its cloud
    — so label coloring works without synchronizing.
    """

    is_synchronous = False
    _EPS = 1e-6  # equal-timestamp grouping tolerance (same clock, same file)
    _CARRY_MAX_BYTES = 4096  # small channels (pose, IMU) are carried between their ticks

    def __init__(self, channels: dict[str, tuple], every: int = 1) -> None:
        # channels: name → (single-channel dataset indexed by event, timestamps (N,))
        self._channels = channels
        self.keys = list(channels)
        events: list[tuple[float, str, int]] = []
        for name, (_ds, ts) in channels.items():
            events.extend((float(t), name, j) for j, t in enumerate(np.asarray(ts)))
        events.sort(key=lambda e: e[0])
        ticks: list[list[tuple[float, str, int]]] = []
        last_t: float | None = None
        for t, name, j in events:
            if last_t is None or t - last_t > self._EPS:
                ticks.append([])
                last_t = t
            ticks[-1].append((t, name, j))
        self._ticks = ticks[:: max(1, int(every))]
        # State channels: small payloads whose "latest value ≤ t" is the natural
        # semantics (pose, IMU). Carrying them makes scrubbing land on a frame that
        # still knows where the robot is; big channels (clouds, images, labelings)
        # stay strict — only shown on their real ticks.
        self._carry: list[str] = []
        for name, (ds, ts) in channels.items():
            if not len(ts):
                continue
            try:
                arr = np.asarray(ds[0].data[name])
            except Exception:  # noqa: BLE001 — unreadable first event: never carry
                continue
            if arr.nbytes <= self._CARRY_MAX_BYTES:
                self._carry.append(name)

    def __len__(self) -> int:
        return len(self._ticks)

    def channel_times(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(ts) for k, (_ds, ts) in self._channels.items()}

    def tick_times(self) -> np.ndarray:
        return np.asarray([g[0][0] for g in self._ticks])

    def __getitem__(self, index: int) -> _SparseSyncSample:
        group = self._ticks[index]
        data: dict = {}
        cts: dict[str, float] = {}
        cidx: dict[str, int] = {}
        for t, name, j in group:
            ds, _ts = self._channels[name]
            data[name] = ds[j].data[name]
            cts[name] = t
            cidx[name] = j
        t0 = group[0][0]
        for name in self._carry:
            if name in data:
                continue
            ds, ts = self._channels[name]
            j = int(np.searchsorted(ts, t0 + self._EPS)) - 1  # latest event ≤ t
            if j >= 0:
                data[name] = ds[j].data[name]
                cts[name] = float(ts[j])
                cidx[name] = j
        return _SparseSyncSample(data, t0, cts, cidx)


class _SparseSync:
    """A synchronized view plus sparse channels overlaid on their matching ticks.

    Index `i` is the i-th tick of the underlying synchronized view; each sparse
    channel appears in `data` only when one of its events falls within `tolerance`
    of the tick — no zero-order hold: a labeling is never shown on a scan it does
    not belong to.
    """

    is_synchronous = True

    def __init__(self, sync, overlays: dict[str, tuple], tolerance: float) -> None:
        self._sync = sync
        self._overlays = overlays
        self._tolerance = float(tolerance)
        self.keys = list(getattr(sync, "keys", [])) + list(overlays)

    def __len__(self) -> int:
        return len(self._sync)

    def _match(self, ts: np.ndarray, t: float) -> int | None:
        """Index of the event nearest to `t` within tolerance, or None."""
        j = int(np.searchsorted(ts, t))
        best, dist = None, self._tolerance
        for cand in (j - 1, j):
            if 0 <= cand < len(ts) and abs(float(ts[cand]) - t) <= dist:
                best, dist = cand, abs(float(ts[cand]) - t)
        return best

    def __getitem__(self, index: int) -> _SparseSyncSample:
        s = self._sync[index]
        data = dict(s.data)
        t = float(s.timestamp)
        channel_ts = {k: t for k in data}
        # Dense channels are resampled onto the tick clock: their counter is the
        # tick index. Sparse overlays keep their own event counter.
        channel_idx = {k: index for k in data}
        for name, (ch_ds, ch_ts) in self._overlays.items():
            j = self._match(ch_ts, t)
            if j is not None:
                data[name] = ch_ds[j].data[name]
                channel_ts[name] = float(ch_ts[j])
                channel_idx[name] = j
        return _SparseSyncSample(data, t, channel_ts, channel_idx)

    def extra_samples(self) -> list[dict]:
        """One merged sample per sparse channel, at its first event's tick — so
        classification sees every sparse channel next to its cloud."""
        out = []
        for _name, (_, ch_ts) in self._overlays.items():
            idx = self._nearest_tick(float(ch_ts[0]))
            if idx is not None:
                out.append(self[idx].data)
        return out

    def _nearest_tick(self, t: float) -> int | None:
        """Nearest synchronized tick to time `t`, by bisection on tick timestamps."""
        lo, hi = 0, len(self._sync) - 1
        if hi < 0:
            return None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if float(self._sync[mid].timestamp) < t:
                lo = mid
            else:
                hi = mid
        t_lo, t_hi = float(self._sync[lo].timestamp), float(self._sync[hi].timestamp)
        return lo if abs(t_lo - t) <= abs(t_hi - t) else hi
