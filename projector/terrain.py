"""Procedural natural terrain — the scene behind `splasher demo`.

One consistent 3D world (Perlin-noise hills, scattered trees, patchy grass) is built
once; the lidar and camera channels are then *derived* from it — a real ray scan and
a real ray-marched render — instead of hand-painted fake sensor data.

Pure numpy (the demo ships with the core install, no extra dependency). The
heightfield grid doubles as the terrain mesh: a regular grid IS a mesh (each cell is
two triangles), and bilinear sampling of the grid is the cheap, vectorized way to
query it — equivalent to nearest-triangle interpolation for a smoothly varying
surface, without the cost of an explicit triangle-soup raycaster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------ Perlin noise


def _perlin2d(shape: tuple[int, int], res: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Classic 2D gradient (Perlin) noise, vectorized. `shape` must be a multiple of
    `res` in both axes. Values roughly in [-1, 1]."""

    def fade(t: np.ndarray) -> np.ndarray:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : 1 / d[0], 0 : res[1] : 1 / d[1]].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    t = fade(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def _fbm(
    shape: tuple[int, int],
    res: tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: int = 2,
) -> np.ndarray:
    """Fractal sum of Perlin octaves — natural-looking rolling relief. Normalized to
    [-1, 1]."""
    noise = np.zeros(shape)
    amp, freq, total = 1.0, 1, 0.0
    for _ in range(octaves):
        noise += amp * _perlin2d(shape, (res[0] * freq, res[1] * freq), rng)
        total += amp
        amp *= persistence
        freq *= lacunarity
    noise /= total
    peak = np.abs(noise).max()
    return noise / peak if peak > 1e-9 else noise


def _bilinear(grid: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample `grid` (H, W) at fractional (row, col) = (v, u), clamped to the edges."""
    h, w = grid.shape
    u = np.clip(u, 0, w - 1.0 - 1e-6)
    v = np.clip(v, 0, h - 1.0 - 1e-6)
    u0, v0 = u.astype(np.int32), v.astype(np.int32)
    u1, v1 = u0 + 1, v0 + 1
    fu, fv = u - u0, v - v0
    a = grid[v0, u0] * (1 - fu) + grid[v0, u1] * fu
    b = grid[v1, u0] * (1 - fu) + grid[v1, u1] * fu
    return a * (1 - fv) + b * fv


# ------------------------------------------------------------------ scene


@dataclass
class Trees:
    pos: np.ndarray  # (N, 2) [x, y] world
    ground_z: np.ndarray  # (N,) terrain height at the base
    trunk_r: np.ndarray  # (N,)
    trunk_h: np.ndarray  # (N,)
    canopy_r: np.ndarray  # (N,)
    hue: np.ndarray  # (N,) small per-tree green-hue jitter, for render variety


@dataclass
class Heightfield:
    grid: np.ndarray  # (H, W) elevation, meters
    detail: np.ndarray  # (H, W) higher-freq noise in [-1, 1], for ground texture
    grass_mask: np.ndarray  # (H, W) in [0, 1], patchy grass coverage
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def _uv(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.grid.shape
        u = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (w - 1)
        v = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (h - 1)
        return u, v

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u, v = self._uv(x, y)
        return _bilinear(self.grid, u, v)

    def texture(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(detail noise, grass coverage) at (x, y), both bilinearly sampled."""
        u, v = self._uv(x, y)
        return _bilinear(self.detail, u, v), _bilinear(self.grass_mask, u, v)

    def normal(self, x: np.ndarray, y: np.ndarray, eps: float = 0.6) -> np.ndarray:
        """Unit surface normal via finite differences, shape (..., 3)."""
        hx0 = self.height(x - eps, y)
        hx1 = self.height(x + eps, y)
        hy0 = self.height(x, y - eps)
        hy1 = self.height(x, y + eps)
        dzdx = (hx1 - hx0) / (2 * eps)
        dzdy = (hy1 - hy0) / (2 * eps)
        n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


@dataclass
class Scene:
    field: Heightfield
    trees: Trees
    sun_dir: np.ndarray = field(default_factory=lambda: _unit(np.array([0.4, -0.5, 0.75])))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def build_scene(
    seed: int,
    x_range: tuple[float, float] = (-15.0, 130.0),
    y_range: tuple[float, float] = (-45.0, 45.0),
    n_trees: int = 7,
    hill_amplitude: float = 3.2,
    grid_shape: tuple[int, int] = (256, 256),
) -> Scene:
    rng = np.random.default_rng(seed)

    relief = _fbm(grid_shape, (4, 4), rng, octaves=5, persistence=0.55)
    grid = relief * hill_amplitude
    detail = _fbm(grid_shape, (4, 4), rng, octaves=3, persistence=0.6)
    grass_mask = np.clip(_fbm(grid_shape, (4, 4), rng, octaves=3, persistence=0.6) * 0.7 + 0.55, 0.0, 1.0)

    field_ = Heightfield(grid, detail, grass_mask, x_range, y_range)

    # Rejection-sampled tree positions: spread out, clear of the ego's start.
    pts: list[tuple[float, float]] = []
    attempts = 0
    min_dist = 11.0
    while len(pts) < n_trees and attempts < n_trees * 200:
        attempts += 1
        x = rng.uniform(x_range[0] + 12.0, x_range[1] - 5.0)
        y = rng.uniform(y_range[0] + 6.0, y_range[1] - 6.0)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for px, py in pts):
            pts.append((x, y))
    pos = np.array(pts, dtype=np.float64).reshape(-1, 2)
    ground_z = field_.height(pos[:, 0], pos[:, 1]) if len(pts) else np.zeros(0)
    n = len(pts)
    trees = Trees(
        pos=pos,
        ground_z=ground_z,
        trunk_r=rng.uniform(0.18, 0.32, n),
        trunk_h=rng.uniform(2.2, 3.4, n),
        canopy_r=rng.uniform(1.6, 2.6, n),
        hue=rng.uniform(-0.08, 0.08, n),
    )
    return Scene(field=field_, trees=trees)


# ------------------------------------------------------------------ ray/primitive tests

MISS, GROUND, GRASS, TRUNK, CANOPY = 0, 1, 2, 3, 4


def _ray_sphere(o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
    """Nearest positive hit distance of rays (o, d[K,3]) against one sphere. NaN = miss."""
    oc = o - c
    b = d @ oc
    disc = b * b - (oc @ oc - r * r)
    ok = disc >= 0
    s = np.full(d.shape[0], np.nan)
    root = -b[ok] - np.sqrt(disc[ok])
    s[ok] = np.where(root >= 0, root, np.nan)
    return s


def _ray_cylinder(
    o: np.ndarray, d: np.ndarray, cx: float, cy: float, r: float, z0: float, z1: float
) -> np.ndarray:
    """Nearest positive hit distance against a vertical, z-capped cylinder. NaN = miss."""
    ox, oy = o[0] - cx, o[1] - cy
    dx, dy = d[:, 0], d[:, 1]
    a = dx * dx + dy * dy
    b = 2 * (dx * ox + dy * oy)
    c = ox * ox + oy * oy - r * r
    s = np.full(d.shape[0], np.nan)
    valid = a > 1e-12
    disc = np.full(d.shape[0], -1.0)
    disc[valid] = b[valid] ** 2 - 4 * a[valid] * c
    hit = valid & (disc >= 0)
    root = (-b[hit] - np.sqrt(disc[hit])) / (2 * a[hit])
    z = o[2] + root * d[hit, 2]
    in_cap = (root >= 0) & (z >= z0) & (z <= z1)
    idx = np.where(hit)[0]
    s[idx[in_cap]] = root[in_cap]
    return s


def _nearest_tree_hit(
    o: np.ndarray, d: np.ndarray, trees: Trees
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest tree hit per ray. Returns (s, kind, tree_index); s = inf where none."""
    n = d.shape[0]
    best_s = np.full(n, np.inf)
    best_kind = np.zeros(n, dtype=np.int32)
    best_idx = np.full(n, -1, dtype=np.int32)
    for i in range(len(trees.pos)):
        cx, cy = trees.pos[i]
        base_z = trees.ground_z[i]
        canopy_c = np.array([cx, cy, base_z + trees.trunk_h[i] + trees.canopy_r[i] * 0.65])
        s_canopy = _ray_sphere(o, d, canopy_c, trees.canopy_r[i])
        s_trunk = _ray_cylinder(o, d, cx, cy, trees.trunk_r[i], base_z, base_z + trees.trunk_h[i])
        for s_cand, kind in ((s_canopy, CANOPY), (s_trunk, TRUNK)):
            ok = ~np.isnan(s_cand) & (s_cand < best_s)
            best_s[ok] = s_cand[ok]
            best_kind[ok] = kind
            best_idx[ok] = i
    return best_s, best_kind, best_idx


def _march_ground(o: np.ndarray, d: np.ndarray, hf: Heightfield, s_vals: np.ndarray) -> np.ndarray:
    """First distance along each ray where it crosses the heightfield. inf = no hit
    within `s_vals`'s range."""
    n = d.shape[0]
    hit_s = np.full(n, np.inf)
    found = np.zeros(n, dtype=bool)
    prev_diff = prev_s = None
    for s in s_vals:
        pos = o + s * d
        diff = pos[:, 2] - hf.height(pos[:, 0], pos[:, 1])
        if prev_diff is not None:
            crossed = (~found) & (prev_diff > 0) & (diff <= 0)
            if crossed.any():
                frac = prev_diff[crossed] / (prev_diff[crossed] - diff[crossed])
                hit_s[crossed] = prev_s + frac * (s - prev_s)
                found |= crossed
        prev_diff, prev_s = diff, s
    return hit_s


# ------------------------------------------------------------------ lidar


def scan_lidar(
    scene: Scene,
    sensor_pos: np.ndarray,
    rng: np.random.Generator,
    n_rings: int = 28,
    n_az: int = 560,
    elev_range: tuple[float, float] = (-22.0, 4.0),
    max_range: float = 42.0,
    march_steps: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a spinning multi-beam lidar against the scene. Returns points
    `(N, 4)` [x, y, z, intensity] float32 and material labels `(N,)` int32."""
    elev = np.radians(np.linspace(elev_range[0], elev_range[1], n_rings))
    az = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    el_g, az_g = np.meshgrid(elev, az, indexing="ij")
    el_g, az_g = el_g.ravel(), az_g.ravel()
    d = np.stack([np.cos(el_g) * np.cos(az_g), np.cos(el_g) * np.sin(az_g), np.sin(el_g)], axis=1)

    o = sensor_pos.astype(np.float64)
    tree_s, tree_kind, _ = _nearest_tree_hit(o, d, scene.trees)
    ground_s = _march_ground(o, d, scene.field, np.linspace(0.4, max_range, march_steps))

    final_s = np.minimum(ground_s, tree_s)
    hit = np.isfinite(final_s)
    kind = np.where(tree_s < ground_s, tree_kind, GROUND)

    s = final_s[hit] + rng.normal(0.0, 0.012, hit.sum())  # range noise
    pts = o[None, :] + s[:, None] * d[hit]
    labels = kind[hit].astype(np.int32)

    # Ground -> grass where the patchy coverage mask says so; blades sit a little
    # proud of bare earth, which is also what makes them register as returns.
    is_ground = labels == GROUND
    if is_ground.any():
        _, grass_cov = scene.field.texture(pts[is_ground, 0], pts[is_ground, 1])
        grassy = rng.random(is_ground.sum()) < grass_cov * 0.85
        idx = np.where(is_ground)[0]
        labels[idx[grassy]] = GRASS
        pts[idx[grassy], 2] += rng.uniform(0.03, 0.32, grassy.sum())

    intensity = np.empty(len(pts), dtype=np.float32)
    for lo, hi, mat in ((0.10, 0.30, GROUND), (0.15, 0.35, GRASS), (0.55, 0.85, TRUNK), (0.45, 0.75, CANOPY)):
        m = labels == mat
        intensity[m] = rng.uniform(lo, hi, m.sum())

    points = np.concatenate([pts, intensity[:, None]], axis=1).astype(np.float32)
    return points, labels


# ------------------------------------------------------------------ camera

SKY_HORIZON = np.array([0.80, 0.83, 0.86])
SKY_ZENITH = np.array([0.20, 0.42, 0.75])
SUN_COLOR = np.array([1.0, 0.92, 0.75])
GRASS_COLOR = np.array([0.24, 0.42, 0.16])
DIRT_COLOR = np.array([0.36, 0.32, 0.24])
TRUNK_COLOR = np.array([0.30, 0.20, 0.13])
CANOPY_COLOR = np.array([0.14, 0.32, 0.12])


def render_camera(
    scene: Scene,
    cam_pos: np.ndarray,
    h: int,
    w: int,
    fov_deg: float = 68.0,
    max_dist: float = 130.0,
    fog_dist: float = 85.0,
    march_steps: int = 40,
    yaw_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Ray-march a pinhole camera looking along +x (rotated `yaw_deg` about z — 180
    for a rear-facing mount) against the scene. Returns a `(h, w, 3)` uint8 image."""
    yaw = np.radians(yaw_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    forward = np.array([cy, sy, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.array([sy, -cy, 0.0])  # forward rotated -90° about z

    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    tan_y = np.tan(np.radians(fov_deg) / 2)
    tan_x = tan_y * (w / h)
    ndc_x = ((2 * (cols + 0.5) / w) - 1) * tan_x
    ndc_y = -((2 * (rows + 0.5) / h) - 1) * tan_y
    dirs = forward + ndc_x.reshape(-1, 1) * right + ndc_y.reshape(-1, 1) * up
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    o = cam_pos.astype(np.float64)
    n_px = dirs.shape[0]

    tree_s, tree_kind, tree_idx = _nearest_tree_hit(o, dirs, scene.trees)

    ground_s = np.full(n_px, np.inf)
    below = dirs[:, 2] < -1e-4
    if below.any():
        ratio = (max_dist / 1.0) ** (1.0 / (march_steps - 1))
        s_vals = 1.0 * ratio ** np.arange(march_steps)
        ground_s[below] = _march_ground(o, dirs[below], scene.field, s_vals)

    final_s = np.minimum(ground_s, tree_s)
    kind = np.where(tree_s < ground_s, tree_kind, np.where(np.isfinite(ground_s), GROUND, MISS))

    sun_dir = scene.sun_dir
    t_sky = np.clip(dirs[:, 2], 0.0, 1.0)
    sun_amt = np.clip(dirs @ sun_dir, 0.0, 1.0) ** 48
    color = SKY_HORIZON[None, :] * (1 - t_sky[:, None]) + SKY_ZENITH[None, :] * t_sky[:, None]
    color = color + sun_amt[:, None] * SUN_COLOR[None, :] * 0.85

    is_ground = kind == GROUND
    if is_ground.any():
        p = o[None, :] + final_s[is_ground, None] * dirs[is_ground]
        normal = scene.field.normal(p[:, 0], p[:, 1])
        ndotl = np.clip(normal @ sun_dir, 0.0, 1.0)
        detail, grass_cov = scene.field.texture(p[:, 0], p[:, 1])
        slope = np.clip((1 - normal[:, 2]) * 1.8 - 0.25, 0.0, 1.0)
        base = GRASS_COLOR[None, :] * (0.85 + 0.3 * detail[:, None])
        base = base * (0.5 + 0.5 * grass_cov[:, None]) + DIRT_COLOR[None, :] * (
            0.5 - 0.5 * grass_cov[:, None]
        )
        base = base * (1 - slope[:, None]) + DIRT_COLOR[None, :] * slope[:, None]
        shaded = base * (0.55 + 0.7 * ndotl[:, None])
        fog = np.clip((final_s[is_ground] / fog_dist) ** 1.3, 0.0, 1.0)[:, None]
        sky_at_hit = (
            SKY_HORIZON[None, :] * (1 - t_sky[is_ground, None]) + SKY_ZENITH[None, :] * t_sky[is_ground, None]
        )
        color[is_ground] = shaded * (1 - fog) + sky_at_hit * fog

    for mat, base_color in ((TRUNK, TRUNK_COLOR), (CANOPY, CANOPY_COLOR)):
        m = kind == mat
        if not m.any():
            continue
        p = o[None, :] + final_s[m, None] * dirs[m]
        idx = tree_idx[m]
        if mat == CANOPY:
            c = scene.trees.pos[idx]
            base_z = scene.trees.ground_z[idx] + scene.trees.trunk_h[idx] + scene.trees.canopy_r[idx] * 0.65
            center = np.stack([c[:, 0], c[:, 1], base_z], axis=1)
            normal = (p - center) / scene.trees.canopy_r[idx][:, None]
            hue = scene.trees.hue[idx][:, None]
        else:
            c = scene.trees.pos[idx]
            radial = p[:, :2] - c
            normal = np.concatenate([radial, np.zeros((len(idx), 1))], axis=1)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
            hue = 0.0
        ndotl = np.clip((normal * sun_dir[None, :]).sum(1), 0.0, 1.0)
        tint = base_color[None, :] * (1 + np.asarray(hue))
        shaded = tint * (0.5 + 0.7 * ndotl[:, None])
        fog = np.clip((final_s[m] / fog_dist) ** 1.3, 0.0, 1.0)[:, None]
        t_here = np.clip(dirs[m, 2], 0.0, 1.0)[:, None]
        sky_at_hit = SKY_HORIZON[None, :] * (1 - t_here) + SKY_ZENITH[None, :] * t_here
        color[m] = shaded * (1 - fog) + sky_at_hit * fog

    if rng is not None:
        color = color + rng.normal(0.0, 0.008, color.shape)  # sensor-noise dither, breaks flat banding
    img = np.clip(color, 0.0, 1.0).reshape(h, w, 3)
    return (img * 255).astype(np.uint8)
