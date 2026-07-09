// 3D point cloud view (Three.js): free navigation (orbit/zoom/pan), z up.
// Bound to ONE point-cloud channel. Color modes (all client-side, live during
// playback):
//   height / intensity      scalar → viridis, with optional vmin/vmax bounds
//   labels:<name>           one labeling through its palette, per-class visibility
//   overlay                 every labeling stacked; later ones paint over earlier
//   confusion               two labelings binarized → ignore/TN/TP/FP/FN + metrics

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { viridis } from "../colors.js";
import { invertRigid, poseToMatrix, translationOf } from "../pose.js";

// Confusion categories: index = category id (used by visibility toggles too).
export const CONFUSION = [
  { key: "ignore", name: "ignored", color: [55, 55, 60] },
  { key: "tn", name: "TN", color: [125, 125, 130] },
  { key: "tp", name: "TP", color: [50, 200, 80] },
  { key: "fp", name: "FP", color: [220, 60, 50] },
  { key: "fn", name: "FN", color: [240, 170, 40] },
];

const FALLBACK = [70, 70, 75];

const LOD_BUDGET = 350000;   // points drawn while the camera moves (full cloud at rest)

function gcd(a, b) {
  while (b) [a, b] = [b, a % b];
  return a;
}

export class CloudView {
  constructor(container) {
    this.container = container;
    this.channel = null;            // cloud channel name
    this.poseChannel = null;        // POSE channel name (ego marker), or null
    this.colorBy = "height";
    this.frame = null;              // last decoded frame
    this.labelings = [];            // [{name, of, labelset}] for this cloud (set by panels)
    this.luts = new Map();          // labeling name → {colors: Map, ignore}
    this.lastWarning = null;
    this.lastMetrics = null;        // confusion mode: {tp, fp, fn, tn, precision, recall, iou}
    // Display parameters (see panels.js options strip).
    this.cfg = {
      vmin: null, vmax: null,       // scalar modes; null = auto
      hidden: new Set(),            // labels mode: hidden class ids
      confusion: null,              // {pred, ref, predPos:Set, refPos:Set, refOff:Set, hiddenCats:Set}
    };
    this._rev = 0;                  // bumped on every cfg change (invalidates the sig)
    this._seq = null;               // refit the camera when the sequence changes
    this._framed = false;
    this._sig = null;
    this._raf = null;               // pending on-demand render (idle = zero GPU work)
    this._disposed = false;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c0a08);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-30, -30, 25);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(this.renderer.domElement);

    // Motion LOD ("poor man's LOD"): buffers are built in a low-discrepancy
    // order, so shrinking the draw range to a budget while the camera moves
    // draws a uniform subset; the full cloud returns the moment it stops.
    this.lodBudget = LOD_BUDGET;
    this._moving = false;
    this._count = 0;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;   // no inertia: the camera stops as soon as you do
    this._hookControls();
    this.controlStyle = "orbit";
    this._tbRaf = null;

    const grid = new THREE.GridHelper(200, 40, 0x2e2a24, 0x1b1815);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    this.geom = new THREE.BufferGeometry();
    const mat = new THREE.PointsMaterial({ size: 0.18, vertexColors: true, sizeAttenuation: true });
    this.points = new THREE.Points(this.geom, mat);
    this.points.frustumCulled = false;   // never cull the whole cloud
    this.scene.add(this.points);

    // GPU picking: the same geometry rendered with per-point ids into a 1×1
    // target (camera view-offset at the cursor), one pixel read back — O(1)
    // per hover instead of the raycaster's O(N) walk.
    this._pickTarget = new THREE.WebGLRenderTarget(1, 1);
    this._pickBuf = new Uint8Array(4);
    this._pickMat = new THREE.ShaderMaterial({
      uniforms: { size: { value: mat.size }, scale: { value: 300 } },
      vertexShader: `
        attribute vec3 pid;
        uniform float size, scale;
        varying vec3 vPid;
        void main() {
          vPid = pid;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (scale / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        varying vec3 vPid;
        void main() { gl_FragColor = vec4(vPid, 1.0); }`,
    });
    this._pickScene = new THREE.Scene();
    const pickPoints = new THREE.Points(this.geom, this._pickMat);
    pickPoints.frustumCulled = false;
    this._pickScene.add(pickPoints);

    // Ego marker: X red (forward), Y green (left), Z blue (up); follows the pose channel.
    this.ego = new THREE.AxesHelper(2.2);
    this.scene.add(this.ego);
    this.cameraMode = "free";       // "free" | "follow" (translation-locked) | "bev" (top-down)
    this._lastEgo = null;           // ego position at the previous frame (follow delta)

    this.trajLine = null;           // THREE.Line of the sequence trajectory, or null

    // Cloud frame: some datasets emit scans in the WORLD frame (pose ≈ where the
    // scan sits), others in the EGO/sensor frame (scan around the origin, pose in
    // a map frame elsewhere) — trajectory and ego must be drawn in the cloud's
    // frame or nothing lines up. "auto" guesses per sequence from the data.
    this.frameMode = "auto";        // "auto" | "world" | "ego"
    this._frameGuess = null;        // resolved auto guess for the current sequence

    // Fly navigation (toaster-style), active on the hovered view: physical WASD
    // move, Q/E down/up, arrows orbit, Shift boosts, R refits. Runs in its own
    // rAF loop so held keys and simultaneous mouse-orbit combine smoothly.
    this._flyKeys = new Set();
    this._shift = false;
    this._hover = false;
    this._flyRaf = null;
    this._flyLast = 0;
    container.addEventListener("mouseenter", () => { this._hover = true; });
    container.addEventListener("mouseleave", () => { this._hover = false; });
    this._onKeyDown = (e) => this._flyKey(e, true);
    this._onKeyUp = (e) => this._flyKey(e, false);
    this._onBlur = () => this._flyKeys.clear();
    window.addEventListener("keydown", this._onKeyDown, true);
    window.addEventListener("keyup", this._onKeyUp, true);
    window.addEventListener("blur", this._onBlur);

    // Hover info: the point under the cursor (throttled GPU pick).
    this._drawIdx = null;           // drawn index k → original point index i
    this._tip = document.createElement("div");
    this._tip.className = "hover-tip";
    this._tip.hidden = true;
    container.appendChild(this._tip);
    this._hoverPending = false;
    container.addEventListener("mousemove", (e) => this._queueHover(e));
    container.addEventListener("mouseleave", () => { this._tip.hidden = true; });

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(container);
    this._resize();
    this._invalidate();
  }

  _queueHover(e) {
    if (this._hoverPending || this._disposed) return;
    this._hoverPending = true;
    requestAnimationFrame(() => {
      this._hoverPending = false;
      this._hoverInfo(e);
    });
  }

  _hoverInfo(e) {
    const p = this.frame && this.channel ? this.frame.channels[this.channel] : null;
    if (!p || !this._drawIdx || this.geom.drawRange.count === 0) { this._tip.hidden = true; return; }
    const r = this.renderer.domElement.getBoundingClientRect();
    const i = this._pickAt(e.clientX, e.clientY);
    if (i === null) { this._tip.hidden = true; return; }
    const stride = p.shape[1];
    const at = (c) => p.data[i * stride + c];
    const parts = [`x ${at(0).toFixed(2)}  y ${at(1).toFixed(2)}  z ${at(2).toFixed(2)}`];
    if (stride >= 4) parts.push(`intensity ${at(3).toFixed(3)}`);
    for (const l of this.labelings) {
      const arr = this.frame.channels[l.name];
      if (!arr || arr.shape[0] !== p.shape[0]) continue;
      const id = arr.data[i];
      const cls = l.labelset && l.labelset.classes.find((c) => c.id === id);
      parts.push(`${l.name}: ${cls ? cls.name : id}`);
    }
    this._tip.textContent = parts.join("  ·  ");
    this._tip.style.left = `${e.clientX - r.left + 12}px`;
    this._tip.style.top = `${e.clientY - r.top + 12}px`;
    this._tip.hidden = false;
  }

  // One pixel of the id render under the cursor → original point index, or null.
  _pickAt(clientX, clientY) {
    const el = this.renderer.domElement;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    const dpr = this.renderer.getPixelRatio();
    const w = Math.floor(r.width * dpr), h = Math.floor(r.height * dpr);
    const x = Math.min(w - 1, Math.max(0, Math.floor((clientX - r.left) * dpr)));
    const y = Math.min(h - 1, Math.max(0, Math.floor((clientY - r.top) * dpr)));
    this.camera.setViewOffset(w, h, x, y, 1, 1);
    const prevColor = new THREE.Color();
    this.renderer.getClearColor(prevColor);
    const prevAlpha = this.renderer.getClearAlpha();
    this.renderer.setRenderTarget(this._pickTarget);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear();
    this.renderer.render(this._pickScene, this.camera);
    this.renderer.readRenderTargetPixels(this._pickTarget, 0, 0, 1, 1, this._pickBuf);
    this.renderer.setRenderTarget(null);
    this.renderer.setClearColor(prevColor, prevAlpha);
    this.camera.clearViewOffset();
    const b = this._pickBuf;
    if (b[3] === 0) return null;                     // background pixel
    const slot = b[0] + (b[1] << 8) + (b[2] << 16);
    return slot < this._count ? this._drawIdx[slot] : null;
  }

  // -------------------------------------------------------- motion LOD
  _hookControls() {
    this.controls.addEventListener("change", () => this._invalidate());
    this.controls.addEventListener("start", () => this._setMoving(true));
    this.controls.addEventListener("end", () => this._setMoving(false));
  }

  _setMoving(on) {
    if (this._moving === on) return;
    this._moving = on;
    this._applyDrawRange();
    this._invalidate();
  }

  _applyDrawRange() {
    this.geom.setDrawRange(0, this._moving ? Math.min(this._count, this.lodBudget) : this._count);
  }

  setChannel(name) { this.channel = name; this._framed = false; this._rebuild(); }
  setPoseChannel(name) { this.poseChannel = name; this._rebuild(); }
  setColorBy(mode) { this.colorBy = mode; this._rev++; this._rebuild(); }
  setLabelings(labelings, luts) { this.labelings = labelings; this.luts = luts; }
  setPointSize(s) {
    this.points.material.size = s;
    this._pickMat.uniforms.size.value = s;
    this._invalidate();
  }

  // "orbit" (stable, z-up — no roll) | "trackball" (free tumbling: straighten a
  // cloud recorded without a mount TF). Trackball applies input inside update(),
  // so it gets its own light rAF loop while active.
  setControlStyle(style) {
    if (style === this.controlStyle) return;
    const target = this.controls.target.clone();
    this.controls.dispose();
    if (style === "trackball") {
      this.controls = new TrackballControls(this.camera, this.renderer.domElement);
      this.controls.rotateSpeed = 2.4;
      this.controls.zoomSpeed = 1.2;
      this.controls.panSpeed = 0.8;
      this.controls.staticMoving = true;   // no inertia, like the orbit config
    } else {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = false;
      this.camera.up.set(0, 0, 1);         // restore the world-up convention
    }
    this.controls.target.copy(target);
    this._hookControls();
    this.controls.update();
    this.controlStyle = style;
    if (style === "trackball") this._trackballLoop();
    this._invalidate();
  }

  _trackballLoop() {
    if (this._tbRaf) return;
    const loop = () => {
      if (this._disposed || this.controlStyle !== "trackball") { this._tbRaf = null; return; }
      this.controls.update();              // applies pending pointer input, fires "change"
      this._tbRaf = requestAnimationFrame(loop);
    };
    this._tbRaf = requestAnimationFrame(loop);
  }

  // "free" (orbit) / "follow" (camera translates with the ego, orientation stays
  // yours) / "bev" (top-down, centered on the ego).
  setCameraMode(mode) {
    this.cameraMode = mode;
    const ego = this.ego.position;
    if (mode === "bev") {
      const h = Math.max(10, this.camera.position.distanceTo(this.controls.target));
      this.controls.target.copy(ego);
      this.camera.position.set(ego.x, ego.y, ego.z + h);
      this.controls.update();
    } else if (mode === "follow" && this._lastEgo === null) {
      this._lastEgo = ego.clone();
    }
    this._invalidate();
  }

  // Sequence trajectory (Float32 (N,3) decoded array, or null). Drawn in world
  // coordinates; in ego-frame mode its matrix carries inv(pose) per frame.
  setTrajectory(arr) {
    if (this.trajLine) {
      this.trajLine.geometry.dispose();
      this.trajLine.material.dispose();
      this.scene.remove(this.trajLine);
      this.trajLine = null;
    }
    if (arr && arr.shape[0] > 1) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(arr.data), 3));
      const mat = new THREE.LineBasicMaterial({ color: 0xffb000, transparent: true, opacity: 0.7 });
      this.trajLine = new THREE.Line(geom, mat);
      this.trajLine.matrixAutoUpdate = false;
      this.scene.add(this.trajLine);
    }
    this._updateEgo();
    this._invalidate();
  }

  setFrameMode(mode) {
    this.frameMode = mode;
    this._updateEgo();
    this._invalidate();
  }

  // Effective cloud frame ("world" | "ego"): explicit mode, else the auto guess.
  effectiveFrame() {
    return this.frameMode === "auto" ? (this._frameGuess || "world") : this.frameMode;
  }

  // Auto guess: a scan is centered on the robot — if its centroid sits near the
  // origin while the pose is far away, the cloud lives in the ego frame. Only
  // decided once the pose has moved enough to disambiguate.
  _guessFrame(p, n, stride, pm) {
    if (this._frameGuess !== null || this.frameMode !== "auto" || !pm) return;
    const [px, py] = translationOf(pm);
    if (Math.hypot(px, py) < 5) return;            // pose still near origin: ambiguous
    let cx = 0, cy = 0, k = 0;
    const step = Math.max(1, Math.floor(n / 500));
    for (let i = 0; i < n; i += step) {
      const x = p.data[i * stride], y = p.data[i * stride + 1];
      if (Number.isFinite(x) && Number.isFinite(y)) { cx += x; cy += y; k++; }
    }
    if (!k) return;
    cx /= k; cy /= k;
    this._frameGuess = Math.hypot(cx, cy) < Math.hypot(cx - px, cy - py) ? "ego" : "world";
  }

  // Merge display parameters (vmin/vmax/hidden/confusion) and recolor.
  set(patch) {
    Object.assign(this.cfg, patch);
    this._rev++;
    this._rebuild();
  }

  setFrame(frame) {
    if (frame && frame.seq !== this._seq) {
      this._seq = frame.seq;
      this._framed = false;
      this._lastEgo = null;         // no follow delta across sequences
      this._frameGuess = null;      // re-guess the cloud frame per sequence
    }
    this.frame = frame;
    this._rebuild();
  }

  refit() { this._framed = false; this._rebuild(); }

  // Arrays the active color mode reads beyond the cloud itself (fast-path signature).
  _deps() {
    const mode = this.colorBy, ch = this.frame.channels;
    if (mode.startsWith("labels:")) return [ch[mode.slice(7)]];
    if (mode === "overlay") return this.labelings.map((l) => ch[l.name]);
    if (mode === "confusion" && this.cfg.confusion) {
      return [ch[this.cfg.confusion.pred], ch[this.cfg.confusion.ref]];
    }
    return [];
  }

  // ------------------------------------------------------------- color contexts
  _labArr(name, n) {
    // A labeling array valid for the current cloud, or a warning string.
    const arr = this.frame.channels[name];
    if (!arr) return `${name}: not on this frame`;
    if (arr.shape[0] !== n) return `${name}: ${arr.shape[0]} labels ≠ ${n} points`;
    return arr.data;
  }

  _prepare(p, n, stride) {
    // Build {color(i) → [r,g,b]|null, counts?} for the active mode; null = hidden.
    const mode = this.colorBy;

    if (mode.startsWith("labels:")) {
      const name = mode.slice(7);
      const data = this._labArr(name, n);
      if (typeof data === "string") { this.lastWarning = data; return this._scalarCtx(p, n, stride, 2); }
      const lut = this.luts.get(name) || { colors: new Map(), ignore: 0 };
      const hidden = this.cfg.hidden;
      return { color: (i) => (hidden.has(data[i]) ? null : lut.colors.get(data[i]) || FALLBACK) };
    }

    if (mode === "overlay") {
      const layers = [];
      for (const l of this.labelings) {
        const data = this._labArr(l.name, n);
        if (typeof data === "string") continue;         // absent layers just don't paint
        layers.push({ data, lut: this.luts.get(l.name) });
      }
      if (!layers.length) { this.lastWarning = "overlay: no labeling on this frame"; return this._scalarCtx(p, n, stride, 2); }
      return {
        color: (i) => {
          let rgb = FALLBACK;
          for (const L of layers) {                     // later layers override where ≠ ignore
            const v = L.data[i];
            if (v !== L.lut.ignore && L.lut.colors.has(v)) rgb = L.lut.colors.get(v);
          }
          return rgb;
        },
      };
    }

    if (mode === "confusion") {
      const c = this.cfg.confusion;
      if (!c || !c.pred || !c.ref) { this.lastWarning = "confusion: pick pred & ref"; return this._scalarCtx(p, n, stride, 2); }
      const pred = this._labArr(c.pred, n);
      const ref = this._labArr(c.ref, n);
      if (typeof pred === "string") { this.lastWarning = pred; return this._scalarCtx(p, n, stride, 2); }
      if (typeof ref === "string") { this.lastWarning = ref; return this._scalarCtx(p, n, stride, 2); }
      const counts = [0, 0, 0, 0, 0];
      return {
        counts,
        color: (i) => {
          let cat;                                       // 0 ignore / 1 TN / 2 TP / 3 FP / 4 FN
          if (c.refOff.has(ref[i])) cat = 0;
          else {
            const rp = c.refPos.has(ref[i]), pp = c.predPos.has(pred[i]);
            cat = pp ? (rp ? 2 : 3) : (rp ? 4 : 1);
          }
          counts[cat]++;
          return c.hiddenCats.has(cat) ? null : CONFUSION[cat].color;
        },
      };
    }

    const sCol = mode === "intensity" && stride >= 4 ? 3 : 2;
    return this._scalarCtx(p, n, stride, sCol);
  }

  _scalarCtx(p, n, stride, sCol) {
    const isFin = Number.isFinite;
    let lo = this.cfg.vmin, hi = this.cfg.vmax;
    if (lo === null || hi === null) {
      let dlo = Infinity, dhi = -Infinity;
      for (let i = 0; i < n; i++) {
        const s = p.data[i * stride + sCol];
        if (isFin(s)) { if (s < dlo) dlo = s; if (s > dhi) dhi = s; }
      }
      if (lo === null) lo = dlo;
      if (hi === null) hi = dhi;
    }
    if (!(hi > lo)) hi = lo + 1;
    return { color: (i) => { const s = p.data[i * stride + sCol]; return viridis(isFin(s) ? (s - lo) / (hi - lo) : 0); } };
  }

  // ------------------------------------------------------------- rebuild
  _rebuild() {
    const p = this.frame && this.channel ? this.frame.channels[this.channel] : null;
    if (!p || p.shape[0] === 0) {
      this.lastWarning = null;
      this.lastMetrics = null;
      this._count = 0;
      this.geom.setDrawRange(0, 0);
      this._invalidate();
      return;
    }

    // Async playback: a frame where neither the cloud nor any array the active mode
    // reads ticked (e.g. a pose event) must not pay a full geometry rebuild — and
    // keeps its warning/metrics.
    const sig = { cloud: p, rev: this._rev, colorBy: this.colorBy, deps: this._deps() };
    if (this._sig && this._sig.cloud === sig.cloud && this._sig.rev === sig.rev
        && this._sig.colorBy === sig.colorBy
        && this._sig.deps.length === sig.deps.length
        && this._sig.deps.every((d, i) => d === sig.deps[i])) {
      this._updateEgo();
      this._invalidate();
      return;
    }
    this._sig = sig;
    this.lastWarning = null;
    this.lastMetrics = null;

    const [n, stride] = p.shape;
    this._guessFrame(p, n, stride,
                     poseToMatrix(this.poseChannel ? this.frame.channels[this.poseChannel] : null));
    const isFin = Number.isFinite;
    const ok = (i) => isFin(p.data[i * stride]) && isFin(p.data[i * stride + 1])
                      && isFin(p.data[i * stride + 2]);
    const ctx = this._prepare(p, n, stride);

    const pos = new Float32Array(n * 3);
    const col = new Uint8Array(n * 3);                   // normalized in the shader
    const pid = new Uint8Array(n * 3);                   // drawn index as a color (GPU pick)
    const drawIdx = new Uint32Array(n);                  // drawn k → original i (hover)
    // Low-discrepancy build order (golden-ratio stride, coprime with n): any
    // prefix of the buffers is a uniform spatial subset, so the motion LOD can
    // shrink the draw range with no per-frame shuffle.
    let s = Math.max(1, Math.round(n * 0.618033988749895) % n);
    while (gcd(s, n) !== 1) s++;
    let k = 0;
    for (let j = 0; j < n; j++) {
      const i = (j * s) % n;
      if (!ok(i)) continue;
      const rgb = ctx.color(i);
      if (rgb === null) continue;                        // hidden class/category
      pos[k * 3] = p.data[i * stride]; pos[k * 3 + 1] = p.data[i * stride + 1]; pos[k * 3 + 2] = p.data[i * stride + 2];
      col[k * 3] = rgb[0]; col[k * 3 + 1] = rgb[1]; col[k * 3 + 2] = rgb[2];
      pid[k * 3] = k & 255; pid[k * 3 + 1] = (k >> 8) & 255; pid[k * 3 + 2] = (k >> 16) & 255;
      drawIdx[k] = i;
      k++;
    }
    this._drawIdx = drawIdx;

    if (ctx.counts) {
      const [, tn, tp, fp, fn] = ctx.counts;
      const div = (a, b) => (b > 0 ? a / b : null);
      this.lastMetrics = {
        tp, fp, fn, tn,
        precision: div(tp, tp + fp),
        recall: div(tp, tp + fn),
        iou: div(tp, tp + fp + fn),
      };
    }

    // Release the previous attributes' GPU buffers (the renderer frees them on geometry
    // `dispose`; replacing attributes without it leaks VRAM per rebuild).
    this.geom.dispose();
    this.geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.geom.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
    this.geom.setAttribute("pid", new THREE.BufferAttribute(pid, 3, true));
    this._count = k;
    this._applyDrawRange();
    this.geom.computeBoundingSphere();
    if (!this._framed && k > 0) { this._framed = true; this._fit(this.geom.boundingSphere); }

    this._updateEgo();
    this._invalidate();
  }

  _updateEgo() {
    const pose = this.poseChannel && this.frame ? this.frame.channels[this.poseChannel] : null;
    const pm = poseToMatrix(pose);
    if (!pm) return;
    if (this.effectiveFrame() === "ego") {
      // Cloud in the sensor frame: the robot IS the origin; the world-frame
      // trajectory is brought into the scan's frame by inv(pose).
      this.ego.position.set(0, 0, 0);
      this.ego.quaternion.identity();
      if (this.trajLine) {
        this.trajLine.matrix.set(...invertRigid(pm));
      }
    } else {
      const m = new THREE.Matrix4().set(...pm);
      const posv = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
      m.decompose(posv, quat, scl);
      this.ego.position.copy(posv);
      this.ego.quaternion.copy(quat);
      if (this.trajLine) this.trajLine.matrix.identity();
    }
    this._followEgo();
  }

  // Follow modes: the camera keeps its user-chosen offset/orientation and translates
  // with the ego ("follow"), or stays straight above it ("bev").
  _followEgo() {
    const ego = this.ego.position;
    if (this.cameraMode === "follow") {
      if (this._lastEgo !== null) {
        const delta = ego.clone().sub(this._lastEgo);
        this.camera.position.add(delta);
        this.controls.target.add(delta);
        this.controls.update();
      }
    } else if (this.cameraMode === "bev") {
      const h = Math.max(10, this.camera.position.z - ego.z);
      this.controls.target.copy(ego);
      this.camera.position.set(ego.x, ego.y, ego.z + h);
      this.controls.update();
    }
    this._lastEgo = ego.clone();
  }

  // Frame the camera on the cloud once, so points are visible wherever they sit in space.
  _fit(sphere) {
    if (!sphere || !Number.isFinite(sphere.radius) || sphere.radius <= 0) return;
    const c = sphere.center, r = sphere.radius;
    this.controls.target.copy(c);
    const d = r * 2.2 + 1;
    this.camera.position.set(c.x - d * 0.7, c.y - d * 0.7, c.z + d * 0.6);
    this.camera.near = Math.max(0.05, r / 100);
    this.camera.far = r * 20 + 100;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._pickMat.uniforms.scale.value = h * 0.5;   // sizeAttenuation, like PointsMaterial
    this._invalidate();
  }

  // ------------------------------------------------------------- fly navigation
  static FLY_CODES = ["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE",
                      "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"];

  _flyKey(e, down) {
    if (e.key === "Shift") { this._shift = down; return; }
    if (!CloudView.FLY_CODES.includes(e.code) && e.code !== "KeyR") return;
    if (down) {
      if (!this._hover || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target;
      if (t && ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName)) return;
      if (e.code === "KeyR") { this.refit(); return; }
      // The hovered 3D view owns these keys (arrows would step the timeline).
      e.preventDefault();
      e.stopPropagation();
      this._flyKeys.add(e.code);
      this._startFly();
    } else {
      // Always release, even if focus moved mid-hold — a stuck key drifts forever.
      this._flyKeys.delete(e.code);
    }
  }

  _startFly() {
    if (this._flyRaf !== null) return;
    this._flyLast = performance.now();
    this._setMoving(true);
    const loop = () => {
      if (this._disposed || this._flyKeys.size === 0) {
        this._flyRaf = null;
        this._setMoving(false);
        return;
      }
      const now = performance.now();
      // Cap dt: after a backgrounded tab the first delta can be huge.
      const dt = Math.min((now - this._flyLast) / 1000, 0.1);
      this._flyLast = now;
      this._fly(dt);
      this.renderer.render(this.scene, this.camera);
      this._flyRaf = requestAnimationFrame(loop);
    };
    this._flyRaf = requestAnimationFrame(loop);
  }

  // One fly step: WASD/QE translate camera + orbit target together (the pivot
  // stays in front, so a simultaneous mouse-drag orbits where you now look);
  // arrows orbit around the target. Speed scales with the scene radius.
  _fly(dt) {
    const has = (c) => this._flyKeys.has(c);
    const radius = (this.geom.boundingSphere && this.geom.boundingSphere.radius) || 20;

    const fwd = this.camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    const dir = new THREE.Vector3();
    if (has("KeyW")) dir.add(fwd);
    if (has("KeyS")) dir.sub(fwd);
    if (has("KeyD")) dir.add(right);
    if (has("KeyA")) dir.sub(right);
    if (has("KeyE")) dir.add(up);
    if (has("KeyQ")) dir.sub(up);
    if (dir.lengthSq() > 0) {
      dir.normalize().multiplyScalar(radius * (this._shift ? 1.5 : 0.5) * dt);
      this.camera.position.add(dir);
      this.controls.target.add(dir);
    }

    let dAz = 0, dPol = 0;
    if (has("ArrowLeft")) dAz += 1;
    if (has("ArrowRight")) dAz -= 1;
    if (has("ArrowUp")) dPol += 1;
    if (has("ArrowDown")) dPol -= 1;
    if (dAz || dPol) {
      const rate = 1.7 * dt;
      const off = this.camera.position.clone().sub(this.controls.target);
      if (dAz) off.applyAxisAngle(new THREE.Vector3(0, 0, 1), dAz * rate);
      if (dPol) {
        const axis = right.clone().setZ(0).normalize();
        const cand = off.clone().applyAxisAngle(axis, dPol * rate);
        const polar = cand.angleTo(new THREE.Vector3(0, 0, 1));
        if (polar > 0.05 && polar < Math.PI - 0.05) off.copy(cand);   // no pole flip
      }
      this.camera.position.copy(this.controls.target).add(off);
    }
    this.controls.update();
  }

  // On-demand rendering: draw once per dirty mark instead of a continuous loop.
  _invalidate() {
    if (this._raf !== null || this._disposed) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = null;
      this.renderer.render(this.scene, this.camera);
    });
  }

  dispose() {
    this._disposed = true;
    if (this._raf !== null) { cancelAnimationFrame(this._raf); this._raf = null; }
    if (this._flyRaf !== null) { cancelAnimationFrame(this._flyRaf); this._flyRaf = null; }
    if (this._tbRaf !== null) { cancelAnimationFrame(this._tbRaf); this._tbRaf = null; }
    window.removeEventListener("keydown", this._onKeyDown, true);
    window.removeEventListener("keyup", this._onKeyUp, true);
    window.removeEventListener("blur", this._onBlur);
    this._ro.disconnect();
    this.controls.dispose();
    this._pickTarget.dispose();
    this._pickMat.dispose();
    this.scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (o.material.map) o.material.map.dispose();
        o.material.dispose();
      }
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
