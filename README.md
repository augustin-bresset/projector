# Projector

**Interactive viewer for multi-sensor robotics sequences** — timeline, parallel
3D / 2D / BEV views, live display parameters, label comparison (confusion).

Sibling of [Toaster](../toaster) and [Splasher](../splasher); replays
[apairo](../apairo) datasets (adapter arrives with milestone 2) and anything else
through a small `Source` protocol. See [DESIGN.md](DESIGN.md) for the full design
and roadmap.

> Status: event timeline by default (native rates, co-timestamped labelings ride
> with their cloud, small state channels carried between ticks); Time panel to
> resample onto a reference clock after opening; per-channel event tracks under
> the timeline (click to seek); automatic lidar mount TF (upright clouds, --raw
> to disable); mosaic drag-dock layout; per-view menu (hide / pause); hover info;
> A–B loop; bookmarks; per-dataset session state (layout, position, settings
> restored on reopen); fly navigation (WASD/QE + arrows on the hovered view);
> orbit or trackball controls per view; world/ego frame auto-detection; camera
> follow modes; past/future accumulation; confusion coloring with live P/R/IoU;
> Open… dialog with shell-style Tab completion.
> Plus user transforms: Load script in the rail imports any of your Python files,
> lists what it finds (functions, parameterized classes, apairo-style `process`
> objects), and runs your picks per frame — in memory only, never written to the
> dataset. Integer per-point output becomes a virtual labels channel (confusion /
> hover / class chips work on it immediately); cloud output replaces the displayed
> cloud. Transforms run top to bottom: put filters before labelers.
> Next: built-in torch (.pt) inference plugin, torchsparse extra, per-view
> accumulation variants, scalar plots.

## Quick start

```bash
uv venv && uv pip install -e ".[app]"   # or ".[api]" for browser-only

projector                  # empty viewer: Open… a dataset from the UI
projector demo             # native window on a synthetic dataset (zero external data)
projector demo --serve     # same app in the browser → http://127.0.0.1:8088

# apairo dataset (adapter auto-detected via .apairo); async rigs need a reference
projector /data/my_dataset --channels lidar,camera,pose,ground_truth \
    --reference lidar --tolerance 0.15
```

## Comparing two labelings (confusion)

Any cloud can carry several labelings (ground truth, model predictions, a driven
corridor…). Switch a 3D view to **Confusion**, pick `pred` and `ref`, and set each
class to `+` (positive) / `−` (negative) / `∅` (not scored, ref side) with the
chips. Points color as TP/TN/FP/FN, precision/recall/IoU update live during
playback, and hiding TP+TN shows *only where the model is wrong*. Everything is
client-side: no server round-trip when you change the binarization.

Sparse channels (a `ground_truth` labeled on a handful of frames) do not collapse
the synchronized timeline: they are overlaid only on the ticks where they exist,
and the 3D view shows a badge on the other frames instead of stale colors.

## Architecture

```
projector/
  core/      numpy-only: Source protocol (sequences, typed channels, labelings),
             LabelSet, poses
  engine/    headless Player (navigation state)
  server/    FastAPI (REST + static front), desktop shell (pywebview)
  adapters/  ArraySource (in-memory); apairo adapter arrives with milestone 2
  web/       vanilla JS front, Three.js vendored (offline), no build step
```

Labelings are first-class channels (`ChannelKind.LABELS`, aligned to a cloud via
`of=`), so a cloud can carry ground truth + predictions side by side — coloring,
overlay and confusion are computed client-side.
