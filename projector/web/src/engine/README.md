# Point-cloud engine

A self-contained three.js point-cloud rendering engine. This directory is the
**reusable layer**: it knows nothing about Toaster's REST API, labels, classes,
segments, or DOM panels — those live one level up (`app.js`, `api.js`,
`colors.js`). Anything imported here besides `three` is a bug.

Designed to be lifted wholesale into sibling projects (projector, apairo
tooling): copy the directory, provide the `three` / `three/addons/` importmap
entries (vendor them — see `toaster/web/vendor/`), and drive the `Viewer`.

## Modules

- `viewer.js` — the `Viewer` class: renderer, free-tumble trackball camera,
  fly navigation (WASD/QE), shader point cloud with per-point colour and
  visibility alpha, motion LOD orchestration, screen-space picking entry
  points, render-on-demand loop (parks at idle — never leave a rAF loop
  spinning; embedded webviews leak per composited frame).
- `octree.js` — `OctreeIndex`: per-frame LOD cut (frustum-culled,
  screen-space-error driven) and exact picking traversals over the node table.
- `octree-worker.js` — builds that node table off-thread: a Potree-style
  additive octree encoded as one index permutation + flat typed arrays.
- `overlays.js` — scene furniture: `ReferenceGrid` (fading 1 m/10 m ground
  grid), `OrbitIndicator` (rerun-style orbit-pivot crosshair).

## Consumer contract

```js
const viewer = new Viewer(containerElement);
viewer.setCloud(xyzFloat32);            // (n*3,) — owns LOD/octree rebuild
viewer.setCloud(xyz, {octree: false, frame: false}); // streaming: no worker churn, camera untouched
viewer.buildOctree();                   // when a stream pauses — restores exact-fast picking
viewer.setColors(colors, alpha);        // full recolour: (n*3,) rgb + (n,) alpha in [0,1]
const { colors, alpha } = viewer.colorArrays(); // live buffers for in-place patching…
viewer.commitColors(touchedIndices);    // …then upload only the dirty ranges
viewer.pick(clientX, clientY);          // nearest visible point index, or -1
viewer.pickBox(x0, y0, x1, y1);         // visible point indices in a screen rect
viewer.frame();                         // reset camera to frame the cloud
viewer.setPointSize(size); viewer.setRound(bool); viewer.setBackground(hex);
viewer.setSizeAttenuation(bool);        // on: size is metres and shrinks with distance
viewer.setControlStyle("trackball"|"orbit"); // free tumble vs upright orbit
viewer.setBoxMode(bool);                // frees LEFT for a rubber band
viewer.setHighlight(indices, xyz); viewer.setVoxelGrid(centers, size);
viewer.rotateView("roll"|"pitch"|"yaw", radians); viewer.worldUp();
viewer.requestRender();                 // after mutating camera/scene directly
viewer.resize();                        // from a ResizeObserver, for non-window resizes
viewer.flyGate = () => bool;            // gate fly keys (multi-viewer hosts: hovered only)
new Viewer(el, {lodBudget: 350000});    // motion-LOD budget override
viewer.dispose();                       // detach window listeners, free GPU + worker
```

Points with `alpha < 0.5` are invisible AND unpickable — visibility policy is
the consumer's job (the engine just honours the mask). All picking is exact:
LOD decimation only ever affects what is drawn during camera motion, never
what `pick`/`pickBox`/`setColors` see.
