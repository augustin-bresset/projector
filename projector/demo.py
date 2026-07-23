"""Synthetic multi-sequence source for demos/tests — no external data.

A "driving" scene built on the shared procedural world engine
(:mod:`projector.terrain`): the ego drives forward over Perlin-noise hills
scattered with trees and patchy grass, and every sensor is *simulated* against
that one 3D world — a real ray scan and a real ray-marched render, not
hand-painted. The lidar's own material ids become a traversability ground truth,
so the labelling / confusion features have something real to chew on. Channels:

- `lidar`         : ray-cast scan of the terrain + trees, (N, 4) [x, y, z, intensity], ego frame
- `camera_front`  : ray-marched render of the same scene, (H, W, 3) uint8
- `pose`          : (4, 4) ego pose per frame (rides the terrain elevation)
- `gt`            : LABELS of `lidar` — ground/grass → traversable, trunk/canopy → obstacle
- `pred`          : LABELS of `lidar` — `gt` with noise (a fake model, for confusion)

Two sequences over different worlds, so the sequence picker has something to
switch between.
"""

from __future__ import annotations

import numpy as np

from . import terrain
from .adapters.array_source import ArraySource
from .core.labels import LabelClass, LabelSet
from .core.source import ChannelKind, ChannelSpec

CAM_H, CAM_W = 200, 360
LIDAR_MOUNT = 1.8
CAM_FRONT_PLACEMENT = np.array([1.6, 0.0, 1.5], np.float32)

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


def _labels_from_materials(
    materials: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Scan material ids -> a traversability ground truth and a noisy fake prediction.

    ``ground``/``grass`` are traversable, ``trunk``/``canopy`` are obstacles; a few
    points are dropped to ``unlabeled`` and the prediction flips ~12 % of the labels.
    """
    gt = np.where(
        materials == terrain.MISS, 0, np.where(materials <= terrain.GRASS, 1, 2)
    ).astype(np.int32)
    gt[rng.random(len(gt)) < 0.05] = 0  # a few unlabeled points
    pred = (gt == 1).astype(np.int32)  # binary traversable / not
    flip = rng.random(len(pred)) < 0.12
    pred[flip] = 1 - pred[flip]
    return gt, pred


def _make_sequence(n_frames: int, seed: int, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    speed = 1.2
    span = speed * max(n_frames - 1, 1)
    scene = terrain.build_scene(
        seed=seed,
        x_range=(-8.0, span + 26.0),
        y_range=(-22.0, 22.0),
        n_trees=max(3, min(9, n_frames // 4 + 3)),
        hill_amplitude=2.8,
        grid_shape=(192, 192),
    )

    frames: list[dict[str, np.ndarray]] = []
    for t in range(n_frames):
        ex = speed * t
        ez = float(scene.field.height(np.array([ex]), np.array([0.0]))[0])
        ego = np.array([ex, 0.0, ez])

        points_world, materials = terrain.scan_lidar(
            scene, ego + np.array([0.0, 0.0, LIDAR_MOUNT]), rng, n_rings=26, n_az=480, march_steps=36
        )
        lidar = points_world.copy()
        lidar[:, :3] -= ego  # world -> ego frame
        gt, pred = _labels_from_materials(materials, rng)

        camera_front = terrain.render_camera(
            scene, ego + CAM_FRONT_PLACEMENT, CAM_H, CAM_W, march_steps=34, rng=rng
        )

        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = ex
        pose[2, 3] = ez

        frames.append(
            {
                "lidar": lidar.astype(np.float32),
                "camera_front": camera_front,
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
            placement=CAM_FRONT_PLACEMENT,
        ),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
        ChannelSpec("gt", ChannelKind.LABELS, np.dtype("int32"), (None,), of="lidar", labelset=GT_LABELS),
        ChannelSpec("pred", ChannelKind.LABELS, np.dtype("int32"), (None,), of="lidar", labelset=PRED_LABELS),
    ]
    rng = np.random.default_rng(seed)
    return ArraySource(
        specs,
        {
            "seq_a": _make_sequence(40, seed, rng),
            "seq_b": _make_sequence(28, seed + 1, rng),
        },
    )
