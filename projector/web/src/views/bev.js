// Top-down (BEV) view — 2D canvas, world frame (y upwards, locked aspect).
// Bound to ONE point-cloud channel; points colored by height (viridis). Drag = pan,
// wheel = zoom. The ego marker follows the pose channel.

import { finiteRange, viridis } from "../colors.js";
import { applyRigid, invertRigid, poseToMatrix, translationOf } from "../pose.js";

const MAX_POINTS = 30000;

export class BevView {
  constructor(stage) {
    this.stage = stage;
    this.canvas = document.createElement("canvas");
    stage.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.channel = null;
    this.poseChannel = null;
    this.frame = null;
    this.trajectory = null;
    this._seq = null;               // refit when the sequence changes
    this._fitted = false;
    this._frameGuess = null;        // world/ego cloud frame (see _guessFrame)
    this.scale = 10; this.cx = 0; this.cy = 0;
    this.dot = 1.8;                 // point size in px (view settings)
    this._drag = null;

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(stage);
    this._bindMouse();
    this._resize();
  }

  setChannel(name) { this.channel = name; this._fitted = false; this.render(); }
  setPoseChannel(name) { this.poseChannel = name; this.render(); }
  setTrajectory(arr) { this.trajectory = arr; this.render(); }   // (N,3) or null
  setDot(px) { this.dot = px; this.render(); }
  refit() { this._fitted = false; this.render(); }               // re-frame the cloud

  setFrame(frame) {
    if (frame && frame.seq !== this._seq) {
      this._seq = frame.seq;
      this._fitted = false;
      this._frameGuess = null;      // re-guess the cloud frame per sequence
    }
    this.frame = frame;
    this.render();
  }

  // Same world/ego cloud-frame guess as the 3D view (see cloud.js): trajectory and
  // ego marker must be drawn in the cloud's frame or nothing lines up.
  _guessFrame(p) {
    if (this._frameGuess !== null) return;
    const pm = poseToMatrix(this.poseChannel && this.frame ? this.frame.channels[this.poseChannel] : null);
    if (!pm) return;
    const [px, py] = translationOf(pm);
    if (Math.hypot(px, py) < 5) return;            // pose still near origin: ambiguous
    const [n, stride] = p.shape;
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

  get W() { return this.stage.clientWidth; }
  get H() { return this.stage.clientHeight; }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, this.W * dpr);
    this.canvas.height = Math.max(1, this.H * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  }

  _points() {
    return this.frame && this.channel ? this.frame.channels[this.channel] : null;
  }

  _fit(p) {
    const [n, stride] = p.shape;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = p.data[i * stride], y = p.data[i * stride + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    if (!(x1 > x0) || !(y1 > y0)) return;
    this.scale = Math.min(this.W / (x1 - x0), this.H / (y1 - y0)) * 0.92 || 10;
    this.cx = (x0 + x1) / 2;
    this.cy = (y0 + y1) / 2;
    this._fitted = true;
  }

  _toScreen(x, y) {
    return [this.W / 2 + (x - this.cx) * this.scale, this.H / 2 - (y - this.cy) * this.scale];
  }
  _toWorld(px, py) {
    return [this.cx + (px - this.W / 2) / this.scale, this.cy - (py - this.H / 2) / this.scale];
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);
    const p = this._points();
    if (!p || p.shape[0] === 0) return;
    if (!this._fitted) this._fit(p);
    this._guessFrame(p);

    const [n, stride] = p.shape;
    // Height gradient over the z column (subsampled bounds are fine here).
    const zs = new Float32Array(Math.ceil(n / Math.max(1, Math.ceil(n / 4000))));
    for (let i = 0, k = 0; i < n && k < zs.length; i += Math.max(1, Math.ceil(n / 4000)), k++) {
      zs[k] = p.data[i * stride + 2];
    }
    const [zlo, zhi] = finiteRange(zs);

    const step = Math.max(1, Math.ceil(n / MAX_POINTS));
    for (let i = 0; i < n; i += step) {
      const x = p.data[i * stride], y = p.data[i * stride + 1], z = p.data[i * stride + 2];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const [sx, sy] = this._toScreen(x, y);
      if (sx < -2 || sy < -2 || sx > this.W + 2 || sy > this.H + 2) continue;
      const [r, g, b] = viridis((z - zlo) / (zhi - zlo));
      this.ctx.fillStyle = `rgba(${r | 0},${g | 0},${b | 0},0.85)`;
      this.ctx.fillRect(sx, sy, this.dot, this.dot);
    }
    this._drawTrajectory();
    this._drawEgo();
  }

  _drawTrajectory() {
    const t = this.trajectory;
    if (!t || t.shape[0] < 2) return;
    // In ego-frame mode the world trajectory is brought into the scan's frame.
    let inv = null;
    if (this._frameGuess === "ego") {
      const pm = poseToMatrix(this.poseChannel && this.frame ? this.frame.channels[this.poseChannel] : null);
      if (pm) inv = invertRigid(pm);
    }
    const ctx = this.ctx;
    ctx.strokeStyle = "rgba(255,176,0,0.65)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < t.shape[0]; i++) {
      let x = t.data[i * 3], y = t.data[i * 3 + 1];
      if (inv) [x, y] = applyRigid(inv, x, y, t.data[i * 3 + 2]);
      const [sx, sy] = this._toScreen(x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
  }

  // Ego marker at the pose translation, heading from the rotation's first column
  // — at the origin, forward +x, when the cloud lives in the ego frame.
  _drawEgo() {
    const pose = this.poseChannel && this.frame ? this.frame.channels[this.poseChannel] : null;
    let ex = 0, ey = 0, hx = 1, hy = 0;
    if (this._frameGuess !== "ego" && pose && pose.shape.length === 2 && pose.shape[0] === 4) {
      ex = pose.data[3]; ey = pose.data[7];         // row-major (4,4): t = col 3
      hx = pose.data[0]; hy = pose.data[4];         // R[:,0] = forward
    } else if (this._frameGuess !== "ego" && pose && pose.shape.length === 1 && pose.shape[0] === 7) {
      const [x, y, , qx, qy, qz, qw] = pose.data;   // [x,y,z, qx,qy,qz,qw]
      ex = x; ey = y;
      hx = 1 - 2 * (qy * qy + qz * qz);             // R[:,0] from the quaternion
      hy = 2 * (qx * qy + qz * qw);
    }
    const [ox, oy] = this._toScreen(ex, ey);
    const a = Math.atan2(-hy, hx);                  // screen y is flipped
    const L = 26;
    const ctx = this.ctx;
    ctx.strokeStyle = "#ff6b4a"; ctx.fillStyle = "#ff6b4a"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(ox, oy);
    ctx.lineTo(ox + L * Math.cos(a), oy + L * Math.sin(a)); ctx.stroke();
    ctx.beginPath();
    ctx.arc(ox, oy, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#e7eaf0";
    ctx.beginPath(); ctx.arc(ox, oy, 2, 0, Math.PI * 2); ctx.fill();
  }

  _bindMouse() {
    const el = this.canvas;
    const pos = (e) => { const r = el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

    el.addEventListener("mousedown", (e) => {
      e.preventDefault();
      const [x, y] = pos(e);
      this._drag = { x0: x, y0: y };
    });
    window.addEventListener("mousemove", (e) => {
      if (!this._drag) return;
      const [x, y] = pos(e);
      this.cx -= (x - this._drag.x0) / this.scale;
      this.cy += (y - this._drag.y0) / this.scale;
      this._drag.x0 = x; this._drag.y0 = y;
      this.render();
    });
    window.addEventListener("mouseup", () => { this._drag = null; });

    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [x, y] = pos(e);
      const [wx, wy] = this._toWorld(x, y);
      this.scale *= Math.exp(-e.deltaY * 0.0015);
      this.cx = wx - (x - this.W / 2) / this.scale;
      this.cy = wy + (y - this.H / 2) / this.scale;
      this.render();
    }, { passive: false });
  }

  dispose() {
    this._ro.disconnect();
    this.canvas.remove();
  }
}
