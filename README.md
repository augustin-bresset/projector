---
title: Projector
emoji: 🎬
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Replay multi-sensor robotics sequences with label comparison
---

<!-- The YAML block above is Hugging Face Space metadata. It lives ONLY on this
     `hf` branch (pushed to the Space), so it never appears on GitHub's main. -->

> **Live demo** — this Hugging Face Space runs `projector demo --serve` (the synthetic
> source) inside a Docker container: parallel 3D / 2D / BEV views, live display, and
> confusion coloring are all interactive. The demo serves a **single shared session**,
> so concurrent visitors drive the same viewer. See the
> [GitHub repo](https://github.com/augustin-bresset/projector) to run it locally.

<div align="center">

<img src="projector/web/icon.svg" alt="Projector logo" width="120" />

# Projector

**Replay multi-sensor robotics sequences — parallel 3D / 2D / BEV views, live display, and side-by-side label comparison, frame by frame.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Live demo](https://img.shields.io/badge/demo-%F0%9F%A4%97%20Hugging%20Face-ffcc00)](https://huggingface.co/spaces/SmaugC137/projector)

[**▶ Live demo**](https://huggingface.co/spaces/SmaugC137/projector) · [**Sibling tool: Toaster**](https://github.com/augustin-bresset/toaster) · [**Sibling tool: Splasher**](https://github.com/augustin-bresset/splasher) · [Quick start](#quick-start) · [Design & roadmap](DESIGN.md)

</div>

---

**Projector** is a **viewer** for synchronized multi-sensor datasets. You give it a
sequence — at each tick, a **pack of named channels** (3D point cloud, camera image,
pose, labelings) — and it replays them: **scrub a timeline**, lay out **parallel 3D /
2D / BEV views**, tune display **live during playback**, and **compare two labelings**
point-for-point with live precision / recall / IoU.

Built on the same DNA as its siblings — a numpy-only core, a headless engine, a
FastAPI server, and a vanilla Three.js front that doubles as the desktop app — where
[Toaster](https://github.com/augustin-bresset/toaster) and
[Splasher](https://github.com/augustin-bresset/splasher) **label** data, **Projector
replays and inspects** it.

## The app

A **dark, warm "cinema" desktop app** (near-black, amber projector beam). It opens on a
timeline with a mosaic of views you drag-dock into place:

- **Event timeline by default** — every channel refreshes at its **native rate**;
  co-timestamped labelings ride along with their cloud, small state channels are
  carried between ticks. The **Time** panel resamples everything onto a reference
  clock after opening, if you want a fixed rate.
- **Parallel views, each bound to a channel** — a **3D cloud** view, a **2D image**
  view, a top-down **BEV** view. One channel can appear in several views with
  independent settings; per-view menu to **hide / pause**; hover info.
- **Live display parameters** — color mode, colormap bounds, point size, range
  filter, per-class visibility. All **client-side**: moving a slider never hits the
  server, so it works *during* playback.
- **Navigation** — fly (WASD/QE + arrows on the hovered view), **orbit or trackball**
  per view, automatic lidar mount TF (upright clouds; `--raw` to disable), world/ego
  frame auto-detection, and **ego-follow** camera modes with a trajectory overlay.
- **Playback aids** — play / pause, speed, frame stepping, **A–B loop**, bookmarks,
  and **past / future accumulation** registered by pose.
- **Per-dataset session state** — layout, position and settings are restored on
  reopen. Three.js is vendored (`projector/web/vendor`) → **works offline**.

## Quick start

```bash
git clone https://github.com/augustin-bresset/projector && cd projector
uv venv && uv pip install -e ".[app]"   # desktop app — or ".[api]" for browser-only
```

```bash
projector                  # empty viewer: Open… a dataset from the UI
projector demo             # native window on a synthetic dataset (zero external data)
projector demo --serve     # same app in the browser → http://127.0.0.1:8088
```

```bash
# apairo dataset (adapter auto-detected via .apairo); async rigs need a reference
projector /data/my_dataset --channels lidar,camera,pose,ground_truth \
    --reference lidar --tolerance 0.15
```

Launched **without a path**, the *Open…* dialog browses the filesystem with
shell-style **Tab** completion. Extras: `uv pip install -e .` (core + engine only,
numpy) · `.[api]` (FastAPI browser server) · `.[app]` (native desktop window) ·
`.[apairo]` (the apairo input adapter).

## Comparing two labelings (confusion)

Any cloud can carry several labelings (ground truth, model predictions, a driven
corridor…). Switch a 3D view to **Confusion**, pick `pred` and `ref`, and set each
class to `+` (positive) / `−` (negative) / `∅` (not scored, ref side) with the chips.
Points color as TP / TN / FP / FN, **precision / recall / IoU update live** during
playback, and hiding TP+TN shows *only where the model is wrong*. Everything is
client-side: no server round-trip when you change the binarization.

Sparse channels (a `ground_truth` labeled on a handful of frames) do not collapse the
synchronized timeline: they are overlaid only on the ticks where they exist, and the
3D view shows a badge on the other frames instead of stale colors.

## User transforms — run your own model per frame

*Load script* in the rail imports any of your Python files and lists what it finds
(functions, parameterized classes, apairo-style `process` objects); pick some and they
run **per frame, in memory only** — never written to the dataset. Integer per-point
output becomes a **virtual labels channel** (confusion / hover / class chips work on it
immediately); cloud output replaces the displayed cloud. Transforms run top to bottom,
so put filters before labelers.

## Architecture

```
projector/
  core/      numpy-only: Source protocol (sequences, typed channels, labelings),
             LabelSet, poses
  engine/    headless Player (navigation state)
  server/    FastAPI (REST + static front), desktop shell (pywebview)
  adapters/  ArraySource (in-memory); apairo adapter
  web/       vanilla JS front, Three.js vendored (offline), no build step
```

**One front, one engine.** The web front is served by the backend; the desktop app
(`projector` without `--serve`) opens that same front in a **native webview**. Frames
arrive over a **binary websocket** with client-side caching and prefetch; REST carries
only control state. Labelings are first-class channels (`ChannelKind.LABELS`, aligned
to a cloud via `of=`), so a cloud can carry ground truth + predictions side by side —
coloring, overlay and confusion are all computed **client-side**.

See [DESIGN.md](DESIGN.md) for the full design and roadmap.

## Sibling projects

Same house, same architecture DNA (numpy core → headless engine → FastAPI → vanilla
Three.js front, desktop = the web front in a window), different job:

- **[Toaster](https://github.com/augustin-bresset/toaster)** — annotate 3D lidar point
  clouds; plug in any model that *groups* points so **clicking one cluster labels the
  whole group**.
- **[Splasher](https://github.com/augustin-bresset/splasher)** — label synchronized
  multi-channel datasets into a top-down **BEV grid** or per-point labels.

Reach for **Toaster** or **Splasher** to *label* data; reach for **Projector** to
*replay, inspect, and compare* it.

## Development

```bash
uv sync --extra api --group dev   # core + engine + server + pytest/ruff
uv run pytest -q                  # run the suite
uv run ruff check . && uv run ruff format --check .
```

The core and engine import without any UI dependency; tests are pure-numpy + FastAPI's
`TestClient`.
