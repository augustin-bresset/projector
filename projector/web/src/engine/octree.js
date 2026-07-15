// Client-side octree index: the per-frame LOD cut and the picking traversals
// over the node table built by octree-worker.js.
//
// The worker's output (see its header) is an *additive* octree: every node
// owns one contiguous slice of a global point-index permutation (`order`), a
// uniform random sample of its subtree; children refine on top. Two
// consequences this module exploits:
// - any set of accepted nodes is a valid, uniformly-thinning render of the
//   whole cloud — assembling a draw list is just concatenating slices;
// - every point lives in exactly one slice, so a traversal that visits all
//   non-pruned nodes sees every point exactly once — picking stays exact.

import * as THREE from "three";

// Bounding-sphere radius of a node = half-edge * sqrt(3).
const SQRT3 = Math.sqrt(3);

export class OctreeIndex {
  // `data` is the worker's message verbatim: { order, cx, cy, cz, half,
  // start, count, children, buildMs }.
  constructor(data) {
    this.data = data;
    this._frustum = new THREE.Frustum();
    this._projMat = new THREE.Matrix4();
    this._sphere = new THREE.Sphere();
    this._v = new THREE.Vector3();
    this._v2 = new THREE.Vector3();
  }

  get nodeCount() {
    return this.data.start.length;
  }

  // Breadth-first cut for the current camera: skip nodes outside the frustum,
  // emit every visited node's slice into `dst`, and descend only while a node
  // still covers more than `splitPx` on screen. BFS order means coarse
  // coverage lands before fine detail, so running out of `dst` capacity
  // degrades resolution, never coverage. Returns the number of indices used.
  cut(camera, viewportHeight, splitPx, dst) {
    const { cx, cy, cz, half, start, count, children, order } = this.data;
    // The cut runs before render(), so the camera's world matrix may not have
    // caught up with this frame's controls.update() yet.
    camera.updateMatrixWorld();
    this._projMat.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._projMat);
    const camPos = camera.position;
    const fovPx = viewportHeight / 2 / Math.tan((camera.fov * Math.PI) / 360);

    const capacity = dst.length;
    const queue = [0];
    let used = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const ni = queue[qi];
      const r = half[ni] * SQRT3;
      this._sphere.center.set(cx[ni], cy[ni], cz[ni]);
      this._sphere.radius = r;
      if (!this._frustum.intersectsSphere(this._sphere)) continue;
      const cnt = count[ni];
      if (used + cnt > capacity) break;
      dst.set(order.subarray(start[ni], start[ni] + cnt), used);
      used += cnt;
      const dist = camPos.distanceTo(this._sphere.center) - r;
      if (dist <= 0 || (2 * r * fovPx) / dist > splitPx) {
        const cb = ni * 8;
        for (let c = 0; c < 8; c++) if (children[cb + c] >= 0) queue.push(children[cb + c]);
      }
    }
    return used;
  }

  // Conservative screen-space bound of a node's bounding sphere: null when the
  // whole sphere is behind the near plane (safe to prune), the string
  // "straddle" when the sphere crosses the near plane / contains the camera
  // (projection untrustworthy — descend without pruning), else the projected
  // centre in canvas px plus a radius that can only over-estimate.
  _nodeCircle(ni, camera, w, h, fovPx) {
    const { cx, cy, cz, half } = this.data;
    const r = half[ni] * SQRT3;
    const v = this._v.set(cx[ni], cy[ni], cz[ni]);
    const zView = -this._v2.copy(v).applyMatrix4(camera.matrixWorldInverse).z;
    if (zView + r < camera.near) return null;
    if (zView - r < camera.near) return "straddle";
    v.project(camera);
    return {
      sx: (v.x * 0.5 + 0.5) * w,
      sy: (-v.y * 0.5 + 0.5) * h,
      rpx: (r / (zView - r)) * fovPx, // nearest possible depth → max apparent size
    };
  }

  // Depth-first walk that prunes subtrees whose screen bound fails
  // `hit(sx, sy, rpx)`. Every surviving node's slice is scanned point by
  // point — same projection and alpha rules as the viewer's brute-force scan —
  // and each visible on-screen point is handed to `perPoint(i, sx, sy)`.
  pickWalk(camera, w, h, hit, pos, alpha, perPoint) {
    const { order, start, count, children } = this.data;
    const fovPx = h / 2 / Math.tan((camera.fov * Math.PI) / 360);
    const v = this._v;
    const stack = [0];
    while (stack.length > 0) {
      const ni = stack.pop();
      const c = this._nodeCircle(ni, camera, w, h, fovPx);
      if (c === null) continue;
      if (c !== "straddle" && !hit(c.sx, c.sy, c.rpx)) continue;
      const end = start[ni] + count[ni];
      for (let k = start[ni]; k < end; k++) {
        const i = order[k];
        if (alpha[i] < 0.5) continue;
        v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(camera);
        if (v.z < -1 || v.z > 1) continue;
        perPoint(i, (v.x * 0.5 + 0.5) * w, (-v.y * 0.5 + 0.5) * h);
      }
      const cb = ni * 8;
      for (let k = 0; k < 8; k++) if (children[cb + k] >= 0) stack.push(children[cb + k]);
    }
  }
}
