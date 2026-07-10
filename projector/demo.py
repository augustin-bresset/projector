"""Synthetic multi-sequence source for demos/tests — no external data.

"Natural terrain" scene: the ego drives forward over Perlin-noise hills scattered
with trees and patchy grass (`terrain.py` builds the world as a heightfield mesh);
`lidar` and `camera_front` are then *simulated* against that one consistent 3D
world — a real ray scan and a real ray-marched render — not hand-painted. Channels:

- `lidar`         : ray-cast scan of the terrain + trees, (N, 4) [x, y, z, intensity]
- `camera_front`  : ray-marched render of the same scene, (H, W, 3) uint8
- `pose`          : (4, 4) ego pose per frame — rides the terrain elevation
- `gt`            : LABELS of `lidar` — 0 unlabeled / 1 traversable (ground+grass) /
                     2 obstacle (tree trunk/canopy)
- `pred`          : LABELS of `lidar` — a noisy binary trav/not-trav "model",
                     for the confusion / precision-recall showcase

Two sequences with different terrain character (open meadow vs. denser treeline), so
the sequence picker has something to switch between.
"""

from __future__ import annotations

import numpy as np

from . import terrain
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

PRED_LABELS = LabelSet(
    [
        LabelClass(0, "not traversable", (200, 60, 60)),
        LabelClass(1, "traversable", (50, 200, 80)),
    ],
    ignore_id=-1,
)

CAM_H, CAM_W = 180, 320
LIDAR_MOUNT = 1.8  # meters above ground, matches the `lidar` ChannelSpec placement
CAM_PLACEMENT = np.array([1.6, 0.0, 1.5], np.float32)  # meters, ego-relative


def _make_sequence(
    n_frames: int, speed: float, seed: int, n_trees: int, hill_amplitude: float, rng: np.random.Generator
) -> list[dict[str, np.ndarray]]:
    span = speed * (n_frames - 1)
    scene = terrain.build_scene(
        seed=seed,
        x_range=(-8.0, span + 26.0),
        y_range=(-22.0, 22.0),
        n_trees=n_trees,
        hill_amplitude=hill_amplitude,
    )

    frames: list[dict[str, np.ndarray]] = []
    for t in range(n_frames):
        ex = speed * t
        ez = float(scene.field.height(np.array([ex]), np.array([0.0]))[0])

        lidar_pos = np.array([ex, 0.0, ez + LIDAR_MOUNT])
        cam_pos = np.array([ex, 0.0, ez]) + CAM_PLACEMENT

        points, materials = terrain.scan_lidar(scene, lidar_pos, rng)
        image = terrain.render_camera(scene, cam_pos, CAM_H, CAM_W, rng=rng)

        gt = np.where(materials <= terrain.GRASS, 1, 2).astype(np.int32)  # traversable / obstacle
        gt[rng.random(len(gt)) < 0.05] = 0  # a few unlabeled points, as a real labeling would have

        # Fake model: the ground truth with ~12% of the labeled points flipped.
        pred = np.where(gt == 1, 1, 0).astype(np.int32)
        flip = rng.random(len(pred)) < 0.12
        pred[flip] = 1 - pred[flip]

        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = ex
        pose[2, 3] = ez

        frames.append(
            {
                "lidar": points,
                "camera_front": image,
                "pose": pose,
                "gt": gt,
                "pred": pred,
            }
        )
    return frames


def make_demo_source(seed: int = 0) -> ArraySource:
    specs = [
        ChannelSpec(
            "lidar",
            ChannelKind.POINTCLOUD,
            np.dtype("float32"),
            (None, 4),
            placement=np.array([0.0, 0.0, LIDAR_MOUNT], np.float32),
        ),
        ChannelSpec(
            "camera_front",
            ChannelKind.IMAGE,
            np.dtype("uint8"),
            (CAM_H, CAM_W, 3),
            placement=CAM_PLACEMENT,
        ),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
        ChannelSpec("gt", ChannelKind.LABELS, np.dtype("int32"), (None,), of="lidar", labelset=GT_LABELS),
        ChannelSpec("pred", ChannelKind.LABELS, np.dtype("int32"), (None,), of="lidar", labelset=PRED_LABELS),
    ]
    rng = np.random.default_rng(seed)
    return ArraySource(
        specs,
        {
            # open meadow: gentle relief, a handful of scattered trees
            "seq_a": _make_sequence(40, 1.15, seed=seed + 1, n_trees=5, hill_amplitude=2.2, rng=rng),
            # treeline: hillier, denser trees
            "seq_b": _make_sequence(28, 1.35, seed=seed + 2, n_trees=9, hill_amplitude=4.2, rng=rng),
        },
    )
