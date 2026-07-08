"""Example transforms for projector's Load-script flow.

Anything here that looks like a per-frame transform is offered in the UI:
functions taking a cloud, classes with numeric constructor params, apairo-style
objects with a `process` method. Nothing is ever written to disk.
"""

import numpy as np


def high_points(pts, threshold: float = 1.0):
    """Label points above `threshold` meters as class 1 (a fake obstacle detector)."""
    return (pts[:, 2] > threshold).astype(np.int32)


class RangeFilter:
    """Drop points farther than `max_range` meters from the origin."""

    def __init__(self, max_range: float = 15.0):
        self.max_range = max_range

    def __call__(self, pts):
        keep = np.linalg.norm(pts[:, :2], axis=1) <= self.max_range
        return pts[keep]
