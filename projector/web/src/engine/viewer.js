// Three.js point-cloud viewport: shader points (per-point colour + visibility
// alpha), free-tumble (trackball) camera, screen-space picking and rubber-band
// box select. The octree LOD/picking maths lives in octree.js; the reference
// grid and orbit-pivot crosshair in overlays.js.

import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OctreeIndex } from "./octree.js";
import { ReferenceGrid, OrbitIndicator } from "./overlays.js";

// Above this many points, camera motion is rendered decimated; the full cloud
// is redrawn the moment the camera settles. Only the draw list changes —
// attributes (and CPU picking) always see every point, so nothing is ever
// decimated while actually labelling. The decimation itself is an octree cut
// (frustum-culled, dense near the camera, thinning with distance) built in a
// worker; a pre-shuffled random subset covers the seconds until it's ready.
const LOD_BUDGET = 1_000_000;
// How long after the last camera move the view still counts as "in motion".
const LOD_SETTLE_MS = 150;

// Build the octree for any cloud bigger than this: even below the motion-LOD
// threshold it makes picking O(nodes-near-cursor) instead of O(n).
const OCTREE_MIN_POINTS = 200_000;
// Keep descending into a node while its projected diameter exceeds this many
// pixels — below it, the node's own sample is already ~1 point per pixel.
const OCTREE_SPLIT_PX = 110;

// Sort a delta's indices and merge them into [start, count] runs, bridging
// gaps below 512 slots (uploading a few unchanged points is cheaper than
// another bufferSubData call). Returns null when the result is still too
// fragmented — the caller should full-upload instead.
function coalesceRuns(indices) {
  if (indices.length === 0) return [];
  const sorted = Uint32Array.from(indices).sort();
  const runs = [];
  let start = sorted[0], prev = sorted[0];
  for (let k = 1; k < sorted.length; k++) {
    const v = sorted[k];
    if (v - prev > 512) {
      runs.push([start, prev - start + 1]);
      start = v;
    }
    prev = v;
  }
  runs.push([start, prev - start + 1]);
  return runs.length > 64 ? null : runs;
}

const VERT = `
  attribute vec3 acolor;
  attribute float aalpha;
  // uSize is pixels when uAttenuate == 0, world units (metres) when 1.
  uniform float uSize;
  uniform float uAttenuate;
  // (drawing-buffer half-height in px) * projectionMatrix[1][1] — turns a
  // world size at view depth z into an on-screen pixel size.
  uniform float uProjScalePx;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vSizePx; // the point's actual on-screen size, for the outline
  void main() {
    vColor = acolor;
    vAlpha = aalpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uAttenuate > 0.5 ? uSize * uProjScalePx / max(0.0001, -mv.z) : uSize;
    vSizePx = gl_PointSize;
  }`;

const FRAG = `
  varying vec3 vColor;
  varying float vAlpha;
  varying float vSizePx;
  uniform float uRound;
  // Outline width as a fraction of the point's half-width/radius — relative,
  // not pixel-based, so it stays a thin sliver at any point size — but
  // clamped below to a fixed pixel budget so it doesn't grow into a fat,
  // ugly ring once the point is scaled way up.
  uniform float uOutline;
  // Hard cap on the rim's on-screen width, in pixels.
  uniform float uOutlineMaxPx;
  void main() {
    if (vAlpha < 0.5) discard;
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);   // 0 at centre, 0.5 at a round point's silhouette
    float rNorm = r * 2.0; // 0..1 across the radius

    // Centre-to-rim highlight — the same falloff re_renderer (rerun's point
    // shader) uses for its points/spheres — gives each point a soft 3D "bead"
    // look instead of a flat disc, which does most of the work of making
    // overlapping points readable.
    float shade = max(0.4, sqrt(max(0.0, 1.2 - rNorm)));
    vec3 shaded = vColor * shade;

    // uOutline is a fraction of the half-width (vSizePx / 2); re-derive the
    // fraction that a uOutlineMaxPx-wide rim would need at the current point
    // size, and never exceed it.
    float outline = vSizePx > 0.0 ? min(uOutline, uOutlineMaxPx * 2.0 / vSizePx) : uOutline;

    if (uRound > 0.5) {
      float aa = fwidth(r);
      float coverage = 1.0 - smoothstep(0.5 - aa, 0.5 + aa, r);
      if (coverage <= 0.0) discard;
      float rim = smoothstep(0.5 - outline - aa, 0.5 - outline + aa, r);
      gl_FragColor = vec4(mix(shaded, vec3(0.0), rim), coverage);
    } else {
      vec2 e = min(gl_PointCoord, 1.0 - gl_PointCoord);
      float rim = outline > 0.0 ? step(min(e.x, e.y), outline) : 0.0;
      gl_FragColor = vec4(mix(shaded, vec3(0.0), rim), 1.0);
    }
  }`;

export class Viewer {
  // opts:
  // - lodBudget: points drawn during camera motion (default 1M). Streaming
  //   hosts with tighter frame budgets pass a smaller one.
  constructor(container, { lodBudget = LOD_BUDGET } = {}) {
    this.container = container;
    this.lodBudget = lodBudget;
    // Hosts embedding several viewers set this to gate fly input to the
    // hovered one (e.g. () => hovered); null = fly is always armed.
    this.flyGate = null;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1f2430);
    this.camera = new THREE.PerspectiveCamera(55, this._aspect(), 0.01, 100000);
    this.camera.up.set(0, 0, 1); // Z-up, the natural convention for lidar
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);
    // Render on demand: the rAF loop only runs while something can still move
    // the view (a drag in progress, a fly key held, or a pending dirty frame)
    // and stops entirely otherwise. Merely returning early from a 60 fps rAF
    // callback is not enough: each scheduled frame keeps Chromium's BeginFrame
    // pipeline alive, and under QtWebEngine's Vulkan fallback that leaks Oilpan
    // memory until the renderer OOMs after ~15 minutes of idling.
    this._dirty = true;
    this._rafPending = false;
    this._interacting = false; // between the controls' "start" and "end" events
    // Fly navigation on physical WASD + QE (`e.code`, so layout-independent):
    // forward / back along the view axis, strafe left / right, and Q/E down /
    // up rerun-style along the screen's vertical. Keydown/keyup only track
    // which keys are held; the motion itself runs per-frame in _tick so it is
    // smooth and diagonals (e.g. forward+left) combine.
    this._flyKeys = new Set();
    this._shiftDown = false;
    this._radius = 10; // scene scale — refreshed by frame()
    this._clock = new THREE.Clock();

    // Window-level listeners are kept in one named map so dispose() can remove
    // them — a host that creates and destroys viewers must not leak handlers.
    // Blur matters beyond politeness: a drag can be interrupted without ever
    // delivering a "pointerup" to the canvas (alt-tab, a native dialog — the
    // common case in the desktop shell's embedded webview). Without it the
    // "end" event above never fires, _interacting stays stuck true, and the
    // rAF loop below spins forever instead of parking — which is exactly the
    // Vulkan-fallback Oilpan leak this render-on-demand scheme exists to avoid.
    // Shift toggles the box-mode button map; the rest is fly-key tracking.
    this._winHandlers = {
      keydown: (e) => {
        if (e.key === "Shift") this._applyMouseButtons(true);
        this._flyKey(e, true);
      },
      keyup: (e) => {
        if (e.key === "Shift") this._applyMouseButtons(false);
        this._flyKey(e, false);
      },
      blur: () => {
        this._interacting = false;
        this._flyKeys.clear();
        this._requestRender();
      },
      resize: () => this._resize(),
    };
    for (const [type, fn] of Object.entries(this._winHandlers)) window.addEventListener(type, fn);
    // "change" fires on every actual camera move (drag, wheel zoom, fly — the
    // controls detect external position changes in update() — and the arrow-key
    // rotations). That is the motion signal for the LOD: a plain click fires
    // "start" but no "change", so selecting never blinks the subset in.
    this._motionUntil = 0;
    this._motionLod = false; // motion frames render decimated (n > LOD_BUDGET)
    this._lodIndex = null; // random-subset fallback until the octree is built
    this._lodOn = false; // the frame currently on screen was drawn decimated
    this._octree = null; // OctreeIndex over the worker-built node table
    this._octreeWorker = null;
    this._drawAttr = null; // reusable index attribute the octree cut writes into

    // Box mode uses LEFT for the rubber band, so the camera moves on RIGHT
    // (orbit) / MIDDLE (pan). Holding Shift turns a RIGHT-drag into a pan too —
    // the button maps have no modifier support, so we follow the Shift key
    // through the window handlers above.
    this._boxMode = false;

    this.geom = null;
    this.points = null;
    this.highlight = null;
    this.grid = new ReferenceGrid(this.scene);
    this.orbitIndicator = new OrbitIndicator(this.scene);
    // Camera controls last: their event handlers poke the orbit indicator.
    this.controls = null;
    this._controlStyle = null;
    this.setControlStyle("trackball");
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uSize: { value: 2.0 },
        uAttenuate: { value: 0.0 },
        uProjScalePx: { value: 1.0 },
        uRound: { value: 0.0 },
        uOutline: { value: 0.06 },
        uOutlineMaxPx: { value: 1.25 },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
    });
    this._updateProjScale();
    this._requestRender();
  }

  // Switch the camera control style. "trackball" (default) tumbles freely
  // with no locked up-axis, so a scan that isn't gravity-aligned (e.g. a
  // forest-path lidar frame) can be turned to any angle. "orbit" keeps the
  // world's up upright — steadier for gravity-aligned, streamed scenes.
  // The current camera pose and pivot survive the swap.
  setControlStyle(style) {
    if (style === this._controlStyle) return;
    const target = this.controls ? this.controls.target.clone() : null;
    if (this.controls) this.controls.dispose();
    if (style === "orbit") {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = false; // no inertia drift — crisp, predictable
    } else {
      style = "trackball";
      this.controls = new TrackballControls(this.camera, this.renderer.domElement);
      this.controls.staticMoving = true; // no inertia drift — crisp, predictable
      // TrackballControls ships hidden modifier keys (A/S/D force a drag to
      // rotate/zoom/pan). Those are our fly keys: flying with S or D while
      // orbiting turned the drag into abrupt zooms/pans. Neutralize them.
      this.controls.keys = ["", "", ""];
      this.controls.rotateSpeed = 3.0;
      this.controls.zoomSpeed = 1.2;
      this.controls.panSpeed = 0.8;
    }
    this._controlStyle = style;
    if (target) this.controls.target.copy(target);
    this.controls.addEventListener("start", () => {
      this._interacting = true;
      this._schedule();
    });
    this.controls.addEventListener("end", () => {
      this._interacting = false;
      this._requestRender();
    });
    // "change" fires on every actual camera move (drag, wheel zoom, fly — the
    // controls detect external position changes in update() — and the arrow-key
    // rotations). That is the motion signal for the LOD: a plain click fires
    // "start" but no "change", so selecting never blinks the subset in.
    this.controls.addEventListener("change", () => {
      const now = performance.now();
      this._motionUntil = now + LOD_SETTLE_MS;
      this.orbitIndicator.poke(now);
      this._requestRender();
    });
    this._applyMouseButtons(false);
    this.controls.update();
    this._requestRender();
  }

  // Public: hosts that move the camera or mutate the scene directly (follow
  // modes, extra objects) call this to get a frame drawn.
  requestRender() {
    this._requestRender();
  }

  // Mark the view dirty and make sure a frame is scheduled to draw it.
  _requestRender() {
    this._dirty = true;
    this._schedule();
  }

  _schedule() {
    if (this._rafPending) return;
    this._rafPending = true;
    requestAnimationFrame(() => this._tick());
  }

  // Load a cloud. Options for streaming consumers (a new frame every tick):
  // - octree: false skips the worker build — motion LOD falls back to the
  //   random subset. Call buildOctree() when playback pauses instead; a
  //   per-frame octree build would just churn the worker.
  // - frame: false keeps the current camera instead of reframing — a stream
  //   must not yank the camera on every frame.
  setCloud(xyz, { octree = true, frame = true } = {}) {
    if (this.points) {
      this.scene.remove(this.points);
      this.geom.dispose();
    }
    const n = xyz.length / 3;
    this.geom = new THREE.BufferGeometry();
    this.geom.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    this.geom.setAttribute("acolor", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.geom.setAttribute("aalpha", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    this._lodOn = false;
    this._lodIndex = null;
    this._octree = null;
    this._drawAttr = null;
    this._motionLod = n > this.lodBudget;
    if (this._octreeWorker) {
      this._octreeWorker.terminate();
      this._octreeWorker = null;
    }
    if (this._motionLod) {
      // Partial Fisher-Yates: the first LOD_BUDGET slots become a uniform random
      // sample, so drawing them alone still shows the whole scene, just sparser.
      // This is only the stopgap for the second or two the octree takes to build.
      const idx = new Uint32Array(n);
      for (let i = 0; i < n; i++) idx[i] = i;
      for (let i = 0; i < this.lodBudget; i++) {
        const j = i + Math.floor(Math.random() * (n - i));
        const t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
      }
      this._lodIndex = new THREE.BufferAttribute(idx.slice(0, this.lodBudget), 1);
    }
    if (octree && n > OCTREE_MIN_POINTS) this._buildOctree(xyz, n);
    this.points = new THREE.Points(this.geom, this.material);
    this.scene.add(this.points);
    if (frame) {
      this.frame();
    } else {
      // Still track the scene scale (fly speed, indicator sizing) — just
      // don't touch the camera.
      this.geom.computeBoundingSphere();
      const s = this.geom.boundingSphere;
      if (s && Number.isFinite(s.radius) && s.radius > 0) this._radius = s.radius;
      this._requestRender();
    }
    // The grid is scale-derived; rebuilding it for every streamed frame whose
    // extent wobbles a few percent would be pure churn.
    if (!this._gridRadius || Math.abs(this._radius - this._gridRadius) > this._gridRadius * 0.25) {
      this.grid.rebuild(this._radius);
      this._gridRadius = this._radius;
    }
  }

  // Build (or rebuild) the octree for the current cloud — for streaming
  // consumers that load frames with {octree: false} and want exact-fast
  // picking and the octree cut once playback pauses.
  buildOctree() {
    if (!this.geom) return;
    const pos = this.geom.getAttribute("position");
    if (pos.count > OCTREE_MIN_POINTS) this._buildOctree(pos.array, pos.count);
  }

  setColors(colors, alpha) {
    if (!this.geom) return;
    const ca = this.geom.getAttribute("acolor");
    ca.array.set(colors);
    ca.needsUpdate = true;
    const aa = this.geom.getAttribute("aalpha");
    aa.array.set(alpha);
    aa.needsUpdate = true;
    this._requestRender();
  }

  // The live GPU-side arrays, for in-place patching by the delta recolour.
  // Write into them, then call commitColors with the touched indices.
  colorArrays() {
    if (!this.geom) return null;
    return {
      colors: this.geom.getAttribute("acolor").array,
      alpha: this.geom.getAttribute("aalpha").array,
    };
  }

  // Upload only what a label delta touched: the indices are coalesced into a
  // few contiguous ranges (lidar points arrive in scan order, so a labelled
  // cluster is usually index-local too). Falls back to a full upload when the
  // edit is too scattered for ranged updates to win.
  commitColors(indices) {
    if (!this.geom) return;
    const ca = this.geom.getAttribute("acolor");
    const aa = this.geom.getAttribute("aalpha");
    const runs = coalesceRuns(indices);
    if (runs && ca.addUpdateRange) {
      ca.clearUpdateRanges();
      aa.clearUpdateRanges();
      for (const [start, count] of runs) {
        ca.addUpdateRange(start * 3, count * 3);
        aa.addUpdateRange(start, count);
      }
    }
    ca.needsUpdate = true;
    aa.needsUpdate = true;
    this._requestRender();
  }

  setHighlight(indices, xyz) {
    if (this.highlight) {
      this.scene.remove(this.highlight);
      this.highlight.geometry.dispose();
      this.highlight.material.dispose(); // each call makes a new material — leaks its GL program otherwise
      this.highlight = null;
    }
    if (!indices || indices.length === 0) return;
    const pos = new Float32Array(indices.length * 3);
    for (let i = 0; i < indices.length; i++) {
      const p = indices[i] * 3;
      pos[i * 3] = xyz[p];
      pos[i * 3 + 1] = xyz[p + 1];
      pos[i * 3 + 2] = xyz[p + 2];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const m = new THREE.PointsMaterial({
      color: 0xf4d35e,
      size: this.material.uniforms.uSize.value + 4,
      sizeAttenuation: false,
      depthTest: false,
    });
    this.highlight = new THREE.Points(g, m);
    this.scene.add(this.highlight);
    this._requestRender();
  }

  setPointSize(s) {
    this.material.uniforms.uSize.value = s;
    this._requestRender();
  }
  // With attenuation on, setPointSize's value is a WORLD size (metres) and
  // points shrink with distance; off (default), it's a fixed pixel size.
  setSizeAttenuation(on) {
    this.material.uniforms.uAttenuate.value = on ? 1 : 0;
    this._updateProjScale();
    this._requestRender();
  }
  // world-size → on-screen-px factor; depends on the drawing-buffer height
  // and the projection, so refresh after any resize or projection change.
  _updateProjScale() {
    this.material.uniforms.uProjScalePx.value =
      (this.renderer.domElement.height / 2) * this.camera.projectionMatrix.elements[5];
  }
  setRound(on) {
    this.material.uniforms.uRound.value = on ? 1 : 0;
    this._requestRender();
  }
  setBackground(hex) {
    this.scene.background = new THREE.Color(hex);
    this._requestRender();
  }
  setControlsEnabled(on) {
    this.controls.enabled = on;
  }

  // Box mode frees the LEFT button for the rubber band, so the camera moves on
  // RIGHT (orbit), MIDDLE (pan) and the wheel (zoom). Holding Shift turns a
  // RIGHT-drag into a pan as well — the same lateral move a plain right-drag
  // gives in point/voxel mode.
  setBoxMode(on) {
    this._boxMode = on;
    this.controls.enabled = true;
    this._applyMouseButtons(false);
  }

  _flyKey(e, down) {
    this._shiftDown = e.shiftKey;
    if (!["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"].includes(e.code)) return;
    if (down) {
      if (this.flyGate && !this.flyGate()) return; // e.g. another panel is hovered
      // Chorded shortcuts are not fly input: on AZERTY, Ctrl+Z (undo) is the
      // physical KeyW — without this guard every undo lurched the camera forward.
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (!this._rafPending) this._clock.getDelta(); // discard idle time — no jump on the first step
      this._flyKeys.add(e.code);
      this._schedule(); // the loop is parked while idle — restart it for the fly motion
    } else {
      // Always release, even if focus moved to an input mid-hold — otherwise
      // the key would stick and the camera would drift forever.
      this._flyKeys.delete(e.code);
    }
  }

  // Move the camera by the fly keys currently held. The orbit target shifts by
  // the same vector, so the pivot stays in front of the camera and the next
  // mouse-drag orbits around where the user is now looking. Speed scales with
  // the scene radius (like near/far in frame()); Shift boosts it.
  _fly(dt) {
    if (this._flyKeys.size === 0) return;
    const has = (c) => this._flyKeys.has(c);
    const fwd = this.camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion); // screen up
    const dir = new THREE.Vector3();
    if (has("KeyW")) dir.add(fwd);
    if (has("KeyS")) dir.sub(fwd);
    if (has("KeyD")) dir.add(right);
    if (has("KeyA")) dir.sub(right);
    if (has("KeyE")) dir.add(up); // rerun-style: E up, Q down
    if (has("KeyQ")) dir.sub(up);
    if (dir.lengthSq() === 0) return; // opposite keys cancel out
    const speed = this._radius * (this._shiftDown ? 1.5 : 0.5);
    // Cap dt: after the tab was backgrounded the first delta can be huge, and
    // one giant step would teleport the camera out of the scene.
    dir.normalize().multiplyScalar(speed * Math.min(dt, 0.1));
    this.camera.position.add(dir);
    this.controls.target.add(dir);
    this._dirty = true;
  }

  // The two control classes map buttons in opposite directions. Trackball:
  // ACTION (LEFT→rotate, MIDDLE→zoom, RIGHT→pan) to the BUTTON INDEX that
  // triggers it (0=left, 1=middle, 2=right; -1 disables the action). Orbit:
  // BUTTON to the ACTION enum (null disables the button).
  _applyMouseButtons(shift) {
    if (this._controlStyle === "orbit") {
      const M = THREE.MOUSE;
      if (!this._boxMode) {
        this.controls.mouseButtons = { LEFT: M.ROTATE, MIDDLE: M.DOLLY, RIGHT: M.PAN };
      } else if (shift) {
        this.controls.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: M.PAN };
      } else {
        this.controls.mouseButtons = { LEFT: null, MIDDLE: M.PAN, RIGHT: M.ROTATE };
      }
      return;
    }
    if (!this._boxMode) {
      this.controls.mouseButtons = { LEFT: 0, MIDDLE: 1, RIGHT: 2 }; // rotate / zoom / pan
    } else if (shift) {
      this.controls.mouseButtons = { LEFT: -1, MIDDLE: -1, RIGHT: 2 }; // shift+right → pan
    } else {
      this.controls.mouseButtons = { LEFT: 2, MIDDLE: -1, RIGHT: 1 }; // right → orbit, middle → pan
    }
  }

  // Draw a translucent wireframe cube around each voxel centre (one merged
  // LineSegments — cheap even for tens of thousands of voxels).
  setVoxelGrid(centers, size) {
    this.clearVoxelGrid();
    const n = centers.length / 3;
    if (n === 0) return;
    const h = size / 2;
    const corner = [
      [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
      [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
      [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    const pos = new Float32Array(n * edges.length * 2 * 3);
    let k = 0;
    for (let i = 0; i < n; i++) {
      const cx = centers[i * 3], cy = centers[i * 3 + 1], cz = centers[i * 3 + 2];
      for (const [a, b] of edges) {
        pos[k++] = cx + corner[a][0]; pos[k++] = cy + corner[a][1]; pos[k++] = cz + corner[a][2];
        pos[k++] = cx + corner[b][0]; pos[k++] = cy + corner[b][1]; pos[k++] = cz + corner[b][2];
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const m = new THREE.LineBasicMaterial({ color: 0xe10600, transparent: true, opacity: 0.25 });
    this.voxelGrid = new THREE.LineSegments(g, m);
    this.scene.add(this.voxelGrid);
    this._requestRender();
  }

  clearVoxelGrid() {
    if (this.voxelGrid) {
      this.scene.remove(this.voxelGrid);
      this.voxelGrid.geometry.dispose();
      this.voxelGrid.material.dispose();
      this.voxelGrid = null;
      this._requestRender();
    }
  }

  // Hand the cloud to the octree worker; adopt the result when it lands.
  // The xyz copy's buffer is transferred, so the only real cost here is one
  // memcpy — the build itself happens off-thread.
  _buildOctree(xyz, n) {
    const worker = new Worker(new URL("./octree-worker.js", import.meta.url), { type: "module" });
    this._octreeWorker = worker;
    worker.onmessage = (e) => {
      worker.terminate();
      if (worker !== this._octreeWorker) return; // a newer cloud superseded this build
      this._octreeWorker = null;
      this._octree = new OctreeIndex(e.data);
      // The octree replaces the random fallback outright; free its 4 MB/M.
      this._lodIndex = null;
      // The cut writes into one preallocated index attribute. Capacity: the
      // budget plus one worst-case node (traversal stops *after* the node that
      // crosses the budget), clamped to the cloud itself.
      const capacity = Math.min(n, this.lodBudget + 16384);
      this._drawAttr = new THREE.BufferAttribute(new Uint32Array(capacity), 1);
      this._drawAttr.setUsage(THREE.DynamicDrawUsage);
      console.log(`octree: ${this._octree.nodeCount} nodes over ${n} points in ${e.data.buildMs} ms`);
    };
    const copy = xyz.slice();
    worker.postMessage({ xyz: copy, n }, [copy.buffer]);
  }

  // Swap the draw list between the full cloud and a decimated one — the octree
  // cut when it's ready, the pre-shuffled random subset until then.
  _applyLod(on) {
    if (!this.geom) return;
    if (on && this._octree && this._drawAttr) {
      const used = this._octree.cut(
        this.camera, this.container.clientHeight, OCTREE_SPLIT_PX, this._drawAttr.array,
      );
      if (this._drawAttr.addUpdateRange) {
        this._drawAttr.clearUpdateRanges();
        this._drawAttr.addUpdateRange(0, used);
      }
      this._drawAttr.needsUpdate = true;
      if (this.geom.index !== this._drawAttr) this.geom.setIndex(this._drawAttr);
      this.geom.setDrawRange(0, used);
      this._lodOn = true;
      return;
    }
    const want = on ? this._lodIndex : null;
    if (this.geom.index === want && this._lodOn === (want !== null)) return;
    this.geom.setIndex(want);
    this.geom.setDrawRange(0, Infinity);
    this._lodOn = want !== null;
  }

  // Nearest *visible* point to a screen position, or -1. Robust (no raycaster
  // threshold tuning). With the octree: only nodes whose screen bound overlaps
  // the cursor are visited; without it: projects every point once per click.
  pick(clientX, clientY) {
    if (!this.geom) return -1;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mx = clientX - rect.left, my = clientY - rect.top;
    const w = rect.width, h = rect.height;
    const pos = this.geom.getAttribute("position").array;
    const alpha = this.geom.getAttribute("aalpha").array;
    let best = -1, bestD = 14 * 14;
    const perPoint = (i, sx, sy) => {
      const dx = sx - mx, dy = sy - my;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    };
    if (this._octree) {
      this._octree.pickWalk(
        this.camera, w, h,
        (sx, sy, rpx) => {
          const dx = sx - mx, dy = sy - my;
          const reach = rpx + 14;
          return dx * dx + dy * dy <= reach * reach;
        },
        pos, alpha, perPoint,
      );
      return best;
    }
    const v = new THREE.Vector3();
    for (let i = 0; i < alpha.length; i++) {
      if (alpha[i] < 0.5) continue;
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(this.camera);
      if (v.z < -1 || v.z > 1) continue;
      perPoint(i, (v.x * 0.5 + 0.5) * w, (-v.y * 0.5 + 0.5) * h);
    }
    return best;
  }

  // Indices of visible points inside a screen rectangle (CSS px relative to viewport).
  pickBox(x0, y0, x1, y1) {
    if (!this.geom) return [];
    const rect = this.renderer.domElement.getBoundingClientRect();
    const lo = [Math.min(x0, x1), Math.min(y0, y1)];
    const hi = [Math.max(x0, x1), Math.max(y0, y1)];
    const w = rect.width, h = rect.height;
    const pos = this.geom.getAttribute("position").array;
    const alpha = this.geom.getAttribute("aalpha").array;
    const out = [];
    const perPoint = (i, sx, sy) => {
      if (sx >= lo[0] && sx <= hi[0] && sy >= lo[1] && sy <= hi[1]) out.push(i);
    };
    if (this._octree) {
      this._octree.pickWalk(
        this.camera, w, h,
        (sx, sy, rpx) => {
          // Distance from the node's screen circle to the rect: prune only
          // when even the circle's closest approach misses the rect.
          const dx = Math.max(lo[0] - sx, 0, sx - hi[0]);
          const dy = Math.max(lo[1] - sy, 0, sy - hi[1]);
          return dx * dx + dy * dy <= rpx * rpx;
        },
        pos, alpha, perPoint,
      );
      return out;
    }
    const v = new THREE.Vector3();
    for (let i = 0; i < alpha.length; i++) {
      if (alpha[i] < 0.5) continue;
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(this.camera);
      if (v.z < -1 || v.z > 1) continue;
      perPoint(i, (v.x * 0.5 + 0.5) * w, (-v.y * 0.5 + 0.5) * h);
    }
    return out;
  }

  frame() {
    if (!this.geom) return;
    this.geom.computeBoundingSphere();
    const s = this.geom.boundingSphere;
    if (!s || !Number.isFinite(s.radius) || s.radius <= 0) return; // empty cloud
    this._radius = s.radius;
    this.controls.target.copy(s.center);
    // 3/4 aerial view for a Z-up scene (above, and to the side). Tumbling and
    // the arrow-key rotations change camera.up — restore it too, or a "reset"
    // after rolling the scene would keep the roll.
    this.camera.up.set(0, 0, 1);
    const off = new THREE.Vector3(1.3, -1.3, 0.9).multiplyScalar(s.radius);
    this.camera.position.copy(s.center.clone().add(off));
    this.camera.near = Math.max(s.radius / 1000, 0.001);
    this.camera.far = s.radius * 50;
    this.camera.updateProjectionMatrix();
    this._updateProjScale();
    this.controls.update();
    this._requestRender();
  }

  // The world-space "up" the current view shows (the camera's local +Y) — i.e.
  // which way is up on screen right now. Lets a ground filter know where gravity
  // is once a not-level scan has been turned the right way round.
  worldUp() {
    const v = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    return [v.x, v.y, v.z];
  }

  // Step-rotate the view around the target by `angle` (radians): "roll" spins
  // about the view axis (camera stays put), "pitch" tips over the screen's
  // horizontal axis, "yaw" turns about the screen's vertical axis.
  rotateView(kind, angle) {
    const cam = this.camera;
    const target = this.controls.target;
    let axis;
    if (kind === "roll") axis = new THREE.Vector3().subVectors(target, cam.position).normalize();
    else if (kind === "pitch") axis = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion);
    else axis = new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion); // yaw
    const q = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    const offset = cam.position.clone().sub(target).applyQuaternion(q);
    cam.position.copy(target).add(offset);
    cam.up.applyQuaternion(q);
    cam.lookAt(target);
    this.controls.update();
    this._requestRender();
  }

  // Tear down everything the viewer attached outside its own object graph:
  // window listeners, the octree worker, GPU resources, the canvas. For hosts
  // that create and destroy viewers (the engine embedded in another app) —
  // the single-viewer toaster page never needs it.
  dispose() {
    for (const [type, fn] of Object.entries(this._winHandlers)) window.removeEventListener(type, fn);
    if (this._octreeWorker) {
      this._octreeWorker.terminate();
      this._octreeWorker = null;
    }
    this.controls.dispose();
    if (this.points) this.scene.remove(this.points);
    if (this.geom) this.geom.dispose();
    this.material.dispose();
    this.setHighlight(null);
    this.clearVoxelGrid();
    this.grid.dispose();
    this.orbitIndicator.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  _aspect() {
    return this.container.clientWidth / Math.max(1, this.container.clientHeight);
  }
  // Public: hosts whose container resizes without a window resize (panel
  // splitters, workspace layouts) call this from their own ResizeObserver.
  resize() {
    this._resize();
  }
  _resize() {
    this.camera.aspect = this._aspect();
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    // TrackballControls caches the canvas rect for its math; Orbit doesn't.
    if (this.controls.handleResize) this.controls.handleResize();
    this._updateProjScale();
    this._requestRender();
  }

  _tick() {
    this._rafPending = false;
    this._fly(this._clock.getDelta());
    this.controls.update(); // fires "change" (→ dirty + motion window) when the camera moved
    const now = performance.now();
    const moving = this._motionLod && (this._flyKeys.size > 0 || now < this._motionUntil);
    const indicator = this.orbitIndicator.tick(now, this.camera, this.controls.target);
    if (indicator.dirty) this._dirty = true;
    if (this._dirty) {
      this._dirty = false;
      this._applyLod(moving);
      this.renderer.render(this.scene, this.camera);
    } else if (this._lodOn && !moving) {
      // The camera settled with a decimated frame on screen — redraw it full.
      this._applyLod(false);
      this.renderer.render(this.scene, this.camera);
    }
    // Keep looping while something can still move the camera, while a
    // decimated frame is showing (its full-res refine is still owed), or while
    // the orbit-pivot crosshair is still lingering/fading; otherwise the loop
    // dies here and _requestRender()/_schedule() restarts it.
    if (this._interacting || this._flyKeys.size > 0 || this._dirty || this._lodOn || indicator.active)
      this._schedule();
  }
}
