"""Synthetic multi-sequence source for demos/tests — no external data.

"Driving" scene: the ego moves forward in +x, the world is fixed (clouds are emitted
in the world frame so the ego pose is meaningful). Channels:

- `lidar`         : noisy ground + obstacles, (N, 4) [x, y, z, intensity]
- `camera_front`  : fake image with a moving obstacle
- `pose`          : (4, 4) ego pose per frame
- `gt`            : LABELS of `lidar` — 0 unlabeled / 1 traversable / 2 obstacle
- `pred`          : LABELS of `lidar` — `gt` with noise (a fake model, for confusion)

Two sequences with different obstacle layouts, so the sequence picker has something
to switch between.
"""

from __future__ import annotations

import numpy as np

from .adapters.array_source import ArraySource
from .core.labels import LabelClass, LabelSet
from .core.source import ChannelKind, ChannelSpec

GT_LABELS = LabelSet(
    [
        LabelClass(0, "unlabeled", (130, 130, 130)),
        LabelClass(1, "traversable", (50, 200, 80)),
        LabelClass(2, "obstacle", (200, 60, 60)),
    ],
    ignore_id=0,
)


def _make_image(h: int, w: int, t: int, n_frames: int) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    horizon = h // 2
    img[:horizon] = (90, 120, 200)
    img[horizon:] = (70, 110, 70)
    for r in range(horizon, h):
        frac = (r - horizon) / max(1, h - horizon)
        half = int((0.05 + 0.45 * frac) * w)
        c = w // 2
        img[r, max(0, c - half) : min(w, c + half)] = (60, 60, 64)
    prog = t / max(1, n_frames - 1)
    bx = int(prog * (w - 50))
    img[horizon - 30 : horizon + 10, bx : bx + 40] = (210, 80, 60)
    return img


def _make_sequence(n_frames: int, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    h, w = 200, 360
    speed = 1.2

    n_obs = 9
    obs_x = rng.uniform(8.0, 70.0, n_obs)
    obs_y = rng.uniform(-12.0, 12.0, n_obs)
    obs_r = rng.uniform(0.6, 1.8, n_obs)
    obs_h = rng.uniform(1.0, 3.0, n_obs)

    frames: list[dict[str, np.ndarray]] = []
    for t in range(n_frames):
        ex = speed * t

        ng = 14000
        gx = ex + rng.uniform(1.0, 40.0, ng)
        gy = rng.uniform(-20.0, 20.0, ng)
        gz = rng.normal(0.0, 0.03, ng)
        parts = [np.stack([gx, gy, gz, rng.uniform(0.1, 0.3, ng)], axis=1)]
        labels = [np.ones(ng, dtype=np.int32)]  # ground → traversable

        for k in range(n_obs):
            if 0.0 < obs_x[k] - ex < 40.0:
                m = 500
                px = obs_x[k] + rng.normal(0.0, obs_r[k], m)
                py = obs_y[k] + rng.normal(0.0, obs_r[k], m)
                pz = rng.uniform(0.0, obs_h[k], m)
                parts.append(np.stack([px, py, pz, rng.uniform(0.6, 1.0, m)], axis=1))
                labels.append(np.full(m, 2, dtype=np.int32))  # obstacle

        lidar = np.concatenate(parts, axis=0).astype(np.float32)
        gt = np.concatenate(labels, axis=0)
        gt[rng.random(len(gt)) < 0.05] = 0  # some unlabeled points

        # Fake model: the ground truth with ~12% of the labeled points flipped.
        pred = np.where(gt == 1, 1, 0).astype(np.int32)  # binary trav / not-trav
        flip = rng.random(len(pred)) < 0.12
        pred[flip] = 1 - pred[flip]

        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = ex

        frames.append(
            {
                "lidar": lidar,
                "camera_front": _make_image(h, w, t, n_frames),
                "pose": pose,
                "gt": gt,
                "pred": pred,
            }
        )
    return frames


def make_demo_source(seed: int = 0) -> ArraySource:
    h, w = 200, 360
    specs = [
        ChannelSpec(
            "lidar",
            ChannelKind.POINTCLOUD,
            np.dtype("float32"),
            (None, 4),
            placement=np.array([0.0, 0.0, 1.8], np.float32),
        ),
        ChannelSpec(
            "camera_front",
            ChannelKind.IMAGE,
            np.dtype("uint8"),
            (h, w, 3),
            placement=np.array([1.6, 0.0, 1.5], np.float32),
        ),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
        ChannelSpec("gt", ChannelKind.LABELS, np.dtype("int32"), (None,), of="lidar", labelset=GT_LABELS),
        ChannelSpec(
            "pred",
            ChannelKind.LABELS,
            np.dtype("int32"),
            (None,),
            of="lidar",
            labelset=LabelSet(
                [
                    LabelClass(0, "not traversable", (200, 60, 60)),
                    LabelClass(1, "traversable", (50, 200, 80)),
                ],
                ignore_id=-1,
            ),
        ),
    ]
    rng = np.random.default_rng(seed)
    return ArraySource(
        specs,
        {
            "seq_a": _make_sequence(40, rng),
            "seq_b": _make_sequence(28, rng),
        },
    )
