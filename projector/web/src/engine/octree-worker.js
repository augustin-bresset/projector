// Octree builder — runs in a Web Worker so multi-million-point builds never
// block the UI thread.
//
// Potree-style *additive* octree: every point belongs to exactly one node —
// each internal node keeps a uniform random sample of its subtree (its "own"
// points), the rest flow down to its children. Drawing a node therefore adds
// detail on top of its ancestors, so any cut through the tree (coarse far
// away, fine near the camera) is a valid, uniformly-thinning render of the
// whole cloud.
//
// The output is designed for cheap per-frame use on the main thread:
// - `order`: a permutation of point indices where every node's own points are
//   one contiguous slice — a draw list for any set of accepted nodes is just a
//   concatenation of subarray copies.
// - flat node table (typed arrays, no objects): center/half for culling and
//   screen-size tests, start/count for the slice, children indices.
//
// Message in:  { xyz: Float32Array, n: int }
// Message out: { order, cx, cy, cz, half, start, count, children } (all
//               typed arrays, transferred)

// Own-sample size of an internal node. Smaller → sparser coarse levels but a
// deeper, more selective tree; bigger → denser far view, more points drawn
// per accepted node.
const NODE_CAP = 8192;
// Nodes with at most this many points keep them all and stop splitting.
const LEAF_CAP = 16384;

self.onmessage = (e) => {
  const { xyz, n } = e.data;
  const t0 = performance.now();
  const out = build(xyz, n);
  const ms = Math.round(performance.now() - t0);
  self.postMessage({ ...out, buildMs: ms }, [
    out.order.buffer,
    out.cx.buffer,
    out.cy.buffer,
    out.cz.buffer,
    out.half.buffer,
    out.start.buffer,
    out.count.buffer,
    out.children.buffer,
  ]);
};

function build(xyz, n) {
  // Bounding cube (single cube, not per-axis box, so child octants stay cubes
  // and the projected-size metric is isotropic).
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const rootCx = (minX + maxX) / 2, rootCy = (minY + maxY) / 2, rootCz = (minZ + maxZ) / 2;
  const rootHalf = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6) / 2;

  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;

  // Node table as growable plain arrays; converted to typed arrays at the end.
  const cx = [], cy = [], cz = [], half = [], start = [], count = [], children = [];

  // Scratch for the octant partition (reused across nodes, sized to the
  // largest range that can reach it).
  const scratch = new Uint32Array(n);

  // Explicit stack (a 10M-point cloud is ~10 levels deep, but a pathological
  // distribution shouldn't be able to blow the JS call stack).
  const stack = [{ lo: 0, hi: n, x: rootCx, y: rootCy, z: rootCz, h: rootHalf, slot: -1 }];
  while (stack.length > 0) {
    const { lo, hi, x, y, z, h, slot } = stack.pop();
    const idx = cx.length;
    if (slot >= 0) children[slot] = idx;
    cx.push(x); cy.push(y); cz.push(z); half.push(h);
    const total = hi - lo;
    const childBase = children.length + 8; // this node's 8 slots come first
    for (let c = 0; c < 8; c++) children.push(-1);

    if (total <= LEAF_CAP) {
      start.push(lo);
      count.push(total);
      continue;
    }

    // Partial Fisher–Yates: move NODE_CAP uniformly random points of the range
    // to its front — they become this node's own sample. Any prefix of a
    // uniformly shuffled range is itself a uniform sample.
    for (let i = lo; i < lo + NODE_CAP; i++) {
      const j = i + ((Math.random() * (hi - i)) | 0);
      const t = order[i]; order[i] = order[j]; order[j] = t;
    }
    start.push(lo);
    count.push(NODE_CAP);

    // Counting sort of the remainder into the 8 octants.
    const rl = lo + NODE_CAP;
    const counts = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = rl; i < hi; i++) {
      const p = order[i] * 3;
      counts[(xyz[p] > x ? 1 : 0) | (xyz[p + 1] > y ? 2 : 0) | (xyz[p + 2] > z ? 4 : 0)]++;
    }
    const offs = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let c = 1; c < 8; c++) offs[c] = offs[c - 1] + counts[c - 1];
    for (let i = rl; i < hi; i++) {
      const p = order[i] * 3;
      const oct = (xyz[p] > x ? 1 : 0) | (xyz[p + 1] > y ? 2 : 0) | (xyz[p + 2] > z ? 4 : 0);
      scratch[rl + offs[oct]++] = order[i];
    }
    order.set(scratch.subarray(rl, hi), rl);

    const qh = h / 2;
    let off = 0;
    for (let c = 0; c < 8; c++) {
      if (counts[c] === 0) continue;
      stack.push({
        lo: rl + off,
        hi: rl + off + counts[c],
        x: x + (c & 1 ? qh : -qh),
        y: y + (c & 2 ? qh : -qh),
        z: z + (c & 4 ? qh : -qh),
        h: qh,
        slot: childBase - 8 + c,
      });
      off += counts[c];
    }
  }

  return {
    order,
    cx: Float32Array.from(cx),
    cy: Float32Array.from(cy),
    cz: Float32Array.from(cz),
    half: Float32Array.from(half),
    start: Uint32Array.from(start),
    count: Uint32Array.from(count),
    children: Int32Array.from(children),
  };
}
