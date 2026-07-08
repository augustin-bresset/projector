# Projector — Design

> Working name. Candidates in the toaster/splasher family: `projector`, `beamer`,
> `screener`, `viewfinder`. Rename the directory once decided — nothing else depends
> on the name yet.

Interactive web/desktop viewer for multi-sensor robotics sequences. Replays apairo
datasets (and anything else through a small adapter) with a multi-sequence timeline,
parallel 3D / 2D / BEV views, live display parameters, two-labeling comparison
(confusion), and an ego-following camera.

Sibling of [toaster](../toaster) and [splasher](../splasher) — same architecture DNA
(numpy core → headless engine → FastAPI → vanilla Three.js front, desktop = the web
front in a pywebview window). Successor of `apairo_visu` (Open3D viewer, to be
retired). Complement of `apairo_rr`, not a replacement: rerun stays the right tool
for "record an .rrd and inspect a dataset quickly"; projector owns the *interactive*
loop (compare labelings, tweak display live, follow the robot).

## Goals (v1)

1. **Multi-sequence timeline** — sequence picker, scrubbable timeline, play/pause,
   speed, frame stepping. Async rigs supported natively: each channel refreshes at
   its own rate; a per-channel frame-index panel shows what is on screen.
2. **Multi-channel, multi-view** — resizable panels, each bound to a channel:
   3D cloud view, 2D image view, BEV view. One channel can appear in several views
   with independent settings.
3. **Live display parameters** — per view: color mode, colormap bounds (vmin/vmax),
   point size, range filter, per-class visibility. All client-side: changing a
   slider never hits the server, so it works *during* playback.
4. **Labeling mixing / confusion** — several labelings on one cloud; color by one,
   overlay several, or compare two (binary confusion TP/TN/FP/FN) with live
   per-frame metrics. See "Coloring model".
5. **Ego-follow camera** — camera modes: free / follow (translation-locked to the
   robot pose, orientation stays free) / BEV-follow (top-down centered on ego).
   Trajectory overlay with the current position marked.
6. **Smooth playback** — binary websocket streaming with prefetch; REST only for
   control state.

V1 extras (cheap, high value):
- **Accumulation ±N frames** registered by poses (splasher `core/accumulate.py`).
- **Hover info** — point coordinates, intensity, label under the cursor.
- **A–B loop** — replay a timeline segment in a loop.
- **Bookmarks** — mark frames while exploring, clickable list.
- **Session sidecar** per dataset — layout, view settings, bookmarks, last frame.

## Non-goals (v1)

- **Annotation** — toaster and splasher own labeling; projector only displays.
- **Recording / file replay** (.rrd-style) — apairo_rr owns that.
- **Model pipelines** — v2 (see below). V1 visualizes channels that already exist
  on disk; it does not run inference.
- **Being a library for toaster/splasher** — projector *copies* from splasher, it
  does not become a dependency of anything. Extraction of a shared engine is a
  possible future, decided by the rule of three, not now.

## Positioning

| | apairo_rr (rerun) | apairo_visu (Open3D) | splasher | projector |
|---|---|---|---|---|
| Web + desktop | viewer only | no | yes | yes |
| Multi-sequence timeline | yes | no | no | yes |
| Async channels | yes | no | no (sync required) | yes |
| Live display params | limited | partial | per-frame | yes, during playback |
| Two-labeling confusion | via pipeline scripts | no | no | first-class |
| Ego-follow camera | manual | no | no | yes |
| Labeling | no | no | yes (BEV/points) | no |

## Architecture

Layers, copied/adapted from splasher (each depends only on the previous one):

```
projector/
  core/       numpy-only: source protocol, label sets, confusion, accumulation,
              poses/trajectory. Importable headless, zero UI deps.
  engine/     Player session: current sequence/frame/time, prefetch window,
              per-view state that must live server-side (accumulation radius).
              No drawing; exposes a semantic FrameState.
  server/     FastAPI: REST for control, binary websocket for frame streaming,
              serves web/. Desktop shell (pywebview) like splasher/server/desktop.py.
  adapters/   apairo_source.py — the ONLY module importing apairo.
              array_source.py — in-memory source for demo/tests.
  web/        vanilla JS, no build step, Three.js vendored (offline).
    src/views/  cloud.js (3D), image.js (2D), bev.js — seeded from splasher.
    src/        timeline.js, panels.js, resize.js, colors.js, api.js, ws.js
```

Dependencies: `numpy`, `fastapi`, `uvicorn`, `apairo` (confined to `adapters/`),
optional `pywebview` for the desktop window. No build toolchain.

Reused from splasher nearly as-is: `server/protocol.py` (numpy ↔ base64/TypedArray
codec, kept for REST), `web/src/{panels,resize,colors}.js`, the three view modules,
the pywebview shell, the file-browser pattern. Reused concepts from apairo_rr: the
Frames panel, sequence navigation, `confusion_class` and its palette, label configs.

## Data model

### Source protocol

Extends splasher's `Source` with time and sequences:

```python
class ChannelKind(Enum):
    POINTCLOUD = "pointcloud"   # (N, 3) or (N, 3+C)
    IMAGE      = "image"        # (H, W) or (H, W, C)
    POSE       = "pose"         # (4, 4) or (7,)
    LABELS     = "labels"       # (N,) int, aligned to a cloud channel
    SCALAR     = "scalar"       # anything else

@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: ChannelKind
    of: str | None = None            # LABELS: the cloud channel it is aligned to
    labelset: LabelSet | None = None # LABELS: id -> (name, color)
    placement: np.ndarray | None = None

@dataclass(frozen=True)
class SequenceSpec:
    id: str
    n_frames: int                    # reference-clock ticks

class Source(Protocol):
    def sequences(self) -> list[SequenceSpec]: ...
    def channels(self) -> list[ChannelSpec]: ...
    def frame(self, seq: str, index: int) -> Frame: ...
```

`Frame.channels: dict[str, np.ndarray]`, `Frame.timestamps: dict[str, float] | None`.
A synchronous dataset has `timestamps=None`; an async rig reports one timestamp per
channel and a channel may be absent from a frame (the front keeps its last value —
apairo_rr behavior). The apairo adapter maps `ds.sequence_ids` / `ds.sequence(id)`
and resolves the reference clock via `synchronize()` when asked.

### Labelings are channels

A labeling is a `(N,)` integer channel *aligned to* a cloud channel (`of="lidar"`).
One cloud can carry any number of them: `ground_truth`, `trav_traj`, a model's
prediction dumped by a preprocessor, a corridor label… The apairo adapter classifies
them by the sibling-channel convention (`<cloud>_<suffix>`) plus an explicit
`labels=` mapping for datasets that name them freely.

**Known trap** (the `trav_traj` 19398 ≠ 9701 bug): a labeling whose length does not
match its cloud on a given frame is *not* an error — the view drops that coloring
for the frame and shows a warning badge. Never crash, never color garbage.

## Coloring model — all client-side

The front receives raw arrays (cloud + its labelings ride along in the same frame
message) and computes colors itself, so every mode below works live during playback
with zero server round-trips.

Per cloud view, `color_by` is one of:

- `flat` — single color.
- `height` / `intensity` / `range` — scalar → colormap, with live vmin/vmax.
- `labels:<channel>` — one labeling through its LabelSet palette. Per-class
  visibility toggles apply.
- `overlay:[A, B, …]` — ordered labelings; later ones paint over earlier ones where
  they are ≠ ignore. This is the generic form of the hand-built `traj_gt` composite
  in `apairo_experiments/viz/run.py` (corridor + floating GT on one cloud).
- `confusion:(pred, ref)` — the v1 headline. Each side gets a **binarization**:
  `positive_ids` (classes counted as positive) and `valid_ids` (points scored at
  all; others → ignore). Defaults degrade gracefully: a 0/1 labeling needs no
  configuration; `ground_truth`-style 0=unlabeled/1=trav/2=obstacle sets
  `valid={1,2}, positive={1}` — the `_gt_binary` logic from viz/run.py, but as two
  checkbox groups in the panel instead of code. Output per point:
  `ignore / TP / TN / FP / FN`, fixed palette (apairo_rr `CONFUSION_CFG`:
  gray / green / dark gray / red / amber). The confusion classes behave like
  classes: toggling everything off except FP+FN shows *only where the model is
  wrong* — the single most useful debugging view.

Because confusion is just counting four categories, the view shows **live
per-frame metrics** (precision / recall / IoU) as a text overlay, plus a running
cumulative since playback start. Free to compute, huge to have.

## Camera

- **Free** — standard orbit (default).
- **Follow** — the camera keeps its user-set offset and orientation but translates
  with the ego pose: place yourself once, the robot stays framed for the whole
  sequence. Requires a POSE channel; the mode is grayed out without one.
- **BEV-follow** — top-down, centered on ego, robot-forward up.

No rigid mode (rotation locked to ego) in v1: pitch/roll on rough terrain makes it
unusable. Trajectory overlay drawn from the pose channel, current position marked;
optional ghost of past/future positions.

## Playback & transport

- **REST** (splasher-style, JSON + base64 codec) for control state: session info,
  view layout, camera mode, bookmarks, session save/load.
- **Binary websocket** for frames: one message = JSON header (seq, index, channel
  names, dtypes, shapes, timestamps) + concatenated raw buffers, decoded straight
  into TypedArrays. No base64, no per-frame REST.
- Server-side **prefetch** of the next K frames in playback direction; client LRU
  cache so scrubbing back is instant. Optional server-side point cap
  (`--max-points`, apairo_rr-style decimation) for huge clouds.

This is the one part deliberately *not* copied from splasher (frame-per-click REST
is fine for labeling, not for 10–20 Hz playback).

## V2 — designed for, not built

- **Parameterizable pipelines** — the `(pts, labels) -> (pts, labels)` contract
  shared by apairo_rr and apairo_visu, with introspectable parameters rendered as
  panel sliders; a change re-executes the current frame server-side. Brings model
  inference in; `viz/run.py` then rebases onto projector.
- **Scalar plots** — a SCALAR channel (IMU, speed, per-frame class counts) as a
  curve with a cursor synced to the timeline.
- **Lidar → camera projection** — points overlaid on the image view via the
  calibration tree (apairo_rr already resolves mount TFs).
- **Multi-class confusion** — per-frame confusion matrix view (binary confusion is
  v1; the matrix needs the plots panel).

## Milestones

1. **Skeleton** — repo, copied splasher plumbing, `ArraySource` demo, one static
   frame rendered in cloud + image + BEV views, resizable panels.
2. **Time** — sequence picker, timeline, websocket streaming, playback with
   prefetch, async channels + Frames panel. apairo adapter on a real dataset.
3. **Color** — display params panel (mode, vmin/vmax, size, range, class
   visibility), `labels:` mode, `overlay:`, `confusion:` with binarization UI and
   live metrics.
4. **Camera** — follow / BEV-follow modes, trajectory overlay, accumulation ±N.
5. **Comfort** — hover info, A–B loop, bookmarks, session sidecar, `--web` flag,
   desktop shell polish.

Acceptance test for the whole v1: replay the `viz/run.py` comparison workflow
(minus inference — using dumped prediction channels) entirely inside projector.
