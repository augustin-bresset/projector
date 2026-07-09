// View manager: addable/closable/resizable panels, each bound to a channel.
// Kinds: "3d" (CloudView + options strip), "img" (canvas), "bev" (BevView).

import { CloudView, CONFUSION } from "./views/cloud.js";
import { BevView } from "./views/bev.js";
import { drawImage } from "./views/image.js";
import { growGutter } from "./resize.js";
import { buildLut, pretty, rgbCss } from "./colors.js";

export class PanelManager {
  constructor(stackEl, session, menuEl = null) {
    this.stack = stackEl;
    this.session = session;
    this.menuEl = menuEl;     // rail "Views" menu: one row per panel (hide / pause)
    this.panels = [];
    // Layout: a split tree — {t:"leaf", id} | {t:"split", dir:"row"|"col", kids}.
    // Docking left/right of a view splits THAT cell, not the whole workspace.
    this.tree = null;
    this.frame = null;
    this._seq = 0;
    this._hintEl = null;      // drop-zone highlight while dragging a panel

    this.cloudKeys = session.channels.filter((c) => c.kind === "pointcloud").map((c) => c.name);
    this.imageKeys = session.channels.filter((c) => c.kind === "image").map((c) => c.name);
    const poses = session.channels.filter((c) => c.kind === "pose").map((c) => c.name);
    this.poseKey = poses.length ? poses[0] : null;

    // Labelings and their LUTs, by cloud channel.
    this.labelings = session.channels.filter((c) => c.kind === "labels");
    this.luts = new Map();
    for (const c of this.labelings) {
      this.luts.set(c.name, buildLut(c.labelset || { ignore_id: -1, classes: [] }));
    }
  }

  labelingsOf(cloud) {
    return this.labelings.filter((c) => c.of === cloud);
  }

  // --------------------------------------------- session persistence (layout)
  serialize() {
    const enc = (node) => {
      if (!node) return null;
      if (node.t === "leaf") {
        const p = this._byId(node.id);
        const out = { kind: p.kind, channel: p.channel, hidden: p.hiddenP, paused: p.paused };
        if (p.kind === "3d") {
          out.colorBy = p.view.colorBy;
          out.cameraMode = p.view.cameraMode;
          out.frameMode = p.view.frameMode;
          out.size = p.view.points.material.size;
          out.controls = p.view.controlStyle;
        } else if (p.kind === "bev") {
          out.dot = p.view.dot;
        } else {
          out.imgFit = p.imgCfg.fit;
          out.imgSmooth = p.imgCfg.smooth;
          out.imgOrder = p.imgCfg.order;
        }
        return out;
      }
      return { dir: node.dir, kids: node.kids.map(enc).filter(Boolean) };
    };
    return enc(this.tree);
  }

  _applySpec(p, spec) {
    if (p.kind === "3d") {
      if (spec.colorBy && [...p.colorSel.options].some((o) => o.value === spec.colorBy)) {
        p.colorSel.value = spec.colorBy;
        p.view.setColorBy(spec.colorBy);
        this._buildOpts(p, spec.colorBy);
      }
      if (spec.cameraMode && !p.camSel.disabled) {
        p.camSel.value = spec.cameraMode;
        p.view.setCameraMode(spec.cameraMode);
      }
      if (spec.frameMode) p.view.setFrameMode(spec.frameMode);
      if (spec.size) p.view.setPointSize(spec.size);
      if (spec.controls) p.view.setControlStyle(spec.controls);
    }
    if (p.kind === "bev" && spec.dot) {
      p.view.setDot(spec.dot);
      this._buildOpts(p);              // controls reflect the restored value
    }
    if (p.kind === "img" && (spec.imgFit || spec.imgSmooth === false || spec.imgOrder)) {
      if (spec.imgFit) p.imgCfg.fit = spec.imgFit;
      if (spec.imgSmooth === false) p.imgCfg.smooth = false;
      if (spec.imgOrder) p.imgCfg.order = spec.imgOrder;
      this._buildOpts(p);              // rebuild controls + apply the cfg
    }
    if (spec.hidden) p.hiddenP = true;
    if (spec.paused) p.paused = true;
  }

  // Rebuild panels from a serialized layout; false when nothing applied
  // (e.g. the channels changed since the state was saved). Accepts the split
  // tree ({dir, kids}) and the legacy array-of-columns format.
  restore(saved) {
    const valid = new Set([...this.cloudKeys, ...this.imageKeys]);
    const specTree = Array.isArray(saved)
      ? { dir: "row", kids: (saved || []).map((col) => ({ dir: "col", kids: col || [] })) }
      : saved;
    let added = 0;
    const build = (node) => {
      if (!node) return null;
      if (node.dir) {
        const kids = (node.kids || []).map(build).filter(Boolean);
        if (!kids.length) return null;
        if (kids.length === 1) return kids[0];
        return { t: "split", dir: node.dir === "col" ? "col" : "row", kids };
      }
      if (!valid.has(node.channel)) return null;
      const p = this._createPanel(node.kind, node.channel);
      added++;
      this._applySpec(p, node);
      return { t: "leaf", id: p.id };
    };
    const tree = build(specTree);
    if (tree && added) {
      this.tree = tree;
      this._relayout();
      this._renderMenu();
    }
    return added > 0;
  }

  // Sequence trajectory ((N,3) decoded array or null) → every 3D and BEV view.
  setTrajectory(arr) {
    this.trajectory = arr;
    for (const p of this.panels) {
      if (p.view && p.view.setTrajectory) p.view.setTrajectory(arr);
    }
  }

  // Options for the "add view" select: one entry per (kind, channel).
  addOptions() {
    const opts = [];
    for (const k of this.cloudKeys) opts.push([`3d:${k}`, `3D — ${pretty(k)}`]);
    for (const k of this.imageKeys) opts.push([`img:${k}`, `Image — ${pretty(k)}`]);
    for (const k of this.cloudKeys) opts.push([`bev:${k}`, `BEV — ${pretty(k)}`]);
    return opts;
  }

  add(kind, channel, col = null) {
    const panel = this._createPanel(kind, channel);
    this._place(panel.id, col);
    this._relayout();
    this._renderMenu();
    if (this.frame) this._render(panel);
    return panel;
  }

  // Build the panel (DOM, view, settings) without touching the layout tree.
  _createPanel(kind, channel) {
    const id = ++this._seq;
    const el = document.createElement("section");
    el.className = "vpanel";
    el.style.flex = "1 1 0";

    const head = document.createElement("div");
    head.className = "vpanel-head";
    const tag = document.createElement("span");
    tag.className = "vpanel-title";
    tag.textContent = { "3d": "3D", img: "IMG", bev: "BEV" }[kind];
    const name = document.createElement("span");
    name.className = "vpanel-name";
    name.textContent = pretty(channel);
    const idx = document.createElement("span");
    idx.className = "vpanel-idx";
    idx.title = "Frame index shown by this view (its channel's own counter on async rigs)";
    const warn = document.createElement("span");
    warn.className = "vpanel-warn";
    const close = document.createElement("button");
    close.className = "tbtn close";
    close.textContent = "✕";
    close.title = "Close view";

    const body = document.createElement("div");
    body.className = "vpanel-body " + (kind === "img" ? "img-body" : "canvas-body");

    const panel = { id, kind, channel, el, body, warn, idxEl: idx, view: null, canvas: null };

    // Every view gets a settings strip behind the gear; its content is per-kind.
    const gear = document.createElement("button");
    gear.className = "tbtn gear";
    gear.textContent = "⚙";
    gear.title = "View settings";
    const optsEl = document.createElement("div");
    optsEl.className = "vopts";
    optsEl.hidden = true;
    panel.optsEl = optsEl;
    gear.onclick = () => { optsEl.hidden = !optsEl.hidden; };

    if (kind === "3d") {
      const labelings = this.labelingsOf(channel);
      const spec = this.session.channels.find((c) => c.name === channel);
      const colorSel = document.createElement("select");
      colorSel.className = "color-sel";
      colorSel.title = "Color by";
      const opts = [["height", "Height"]];
      if (spec && spec.shape && spec.shape.length > 1 && spec.shape[1] >= 4) {
        opts.push(["intensity", "Intensity"]);
      }
      for (const c of labelings) opts.push([`labels:${c.name}`, `Labels — ${pretty(c.name)}`]);
      // Labelings of OTHER clouds stay selectable (e.g. same-sensor variants);
      // the per-frame length check badges them when they don't fit this cloud.
      for (const c of this.labelings) {
        if (c.of !== channel) opts.push([`labels:${c.name}`, `Labels — ${pretty(c.name)} (${pretty(c.of)})`]);
      }
      if (labelings.length >= 1) opts.push(["overlay", "Overlay (all labelings)"]);
      if (labelings.length >= 2) opts.push(["confusion", "Confusion (pred vs ref)"]);
      fillSelect(colorSel, opts);

      const camSel = document.createElement("select");
      camSel.className = "cam-sel";
      camSel.title = "Camera mode";
      fillSelect(camSel, [["free", "Free cam"], ["follow", "Follow ego"], ["bev", "Top-down ego"]]);
      camSel.disabled = !this.poseKey;
      camSel.onchange = () => panel.view.setCameraMode(camSel.value);

      head.append(tag, name, idx, warn, colorSel, camSel, gear, close);
      panel.labelings = labelings;

      panel.view = new CloudView(body);
      panel.view.setLabelings(labelings, this.luts);
      panel.view.setPoseChannel(this.poseKey);
      panel.view.setChannel(channel);
      if (this.trajectory) panel.view.setTrajectory(this.trajectory);

      panel.colorSel = colorSel;
      panel.camSel = camSel;
      colorSel.onchange = () => {
        panel.view.setColorBy(colorSel.value);
        this._buildOpts(panel, colorSel.value);
        optsEl.hidden = false;              // mode change: show what it configures
        this._afterRender(panel);
        if (this.onChanged) this.onChanged();
      };
      camSel.onchange = () => {
        panel.view.setCameraMode(camSel.value);
        if (this.onChanged) this.onChanged();
      };
      this._buildOpts(panel, "height");
    } else if (kind === "bev") {
      head.append(tag, name, idx, warn, gear, close);
      panel.view = new BevView(body);
      panel.view.setPoseChannel(this.poseKey);
      panel.view.setChannel(channel);
      if (this.trajectory) panel.view.setTrajectory(this.trajectory);
      this._buildOpts(panel);
    } else {
      head.append(tag, name, idx, warn, gear, close);
      panel.canvas = document.createElement("canvas");
      body.appendChild(panel.canvas);
      panel.imgCfg = { fit: "contain", smooth: true, order: "rgb" };
      this._buildOpts(panel);
      this._bindImgZoom(panel);
    }
    el.append(head, optsEl, body);

    close.onclick = () => this.remove(id);
    this._bindDrag(panel, head);
    panel.hiddenP = false;
    panel.paused = false;
    this.panels.push(panel);
    return panel;
  }

  // ------------------------------------------------- rail menu (hide / pause)
  _renderMenu() {
    if (!this.menuEl) return;
    this.menuEl.replaceChildren();
    for (const p of this.panels) {
      const row = document.createElement("div");
      row.className = "view-row";
      const name = document.createElement("span");
      name.className = "view-name";
      name.textContent = `${{ "3d": "3D", img: "IMG", bev: "BEV" }[p.kind]} ${pretty(p.channel)}`;
      const hide = document.createElement("button");
      hide.className = "chip";
      const paintHide = () => {
        hide.dataset.state = p.hiddenP ? "off" : "pos";
        hide.textContent = p.hiddenP ? "hidden" : "shown";
      };
      hide.onclick = () => {
        p.hiddenP = !p.hiddenP;
        paintHide();
        this._relayout();
        if (!p.hiddenP && this.frame) this._render(p);
      };
      const pause = document.createElement("button");
      pause.className = "chip";
      const paintPause = () => {
        pause.dataset.state = p.paused ? "off" : "pos";
        pause.textContent = p.paused ? "paused" : "live";
      };
      pause.onclick = () => {
        p.paused = !p.paused;
        paintPause();
        if (!p.paused && this.frame) this._render(p);   // catch up on resume
      };
      paintHide();
      paintPause();
      row.append(name, hide, pause);
      this.menuEl.appendChild(row);
    }
    if (this.onChanged) this.onChanged();
  }

  // ------------------------------------------------- split-tree layout
  // `col` keeps the historic column semantics: root is a row of columns, and
  // col=null stacks into the last one (what "+ Add view" always did).
  _place(id, col) {
    const leaf = { t: "leaf", id };
    if (!this.tree) { this.tree = leaf; return; }
    if (this.tree.t !== "split" || this.tree.dir !== "row") {
      this.tree = { t: "split", dir: "row", kids: [this.tree] };
    }
    const kids = this.tree.kids;
    if (col === null) col = kids.length - 1;
    if (col >= kids.length) { kids.push(leaf); return; }
    const target = kids[col];
    if (target.t === "split" && target.dir === "col") target.kids.push(leaf);
    else kids[col] = { t: "split", dir: "col", kids: [target, leaf] };
  }

  _dropFromLayout(id) {
    const prune = (node) => {
      if (!node) return null;
      if (node.t === "leaf") return node.id === id ? null : node;
      node.kids = node.kids.map(prune).filter(Boolean);
      if (!node.kids.length) return null;
      if (node.kids.length === 1) return node.kids[0];
      return node;
    };
    this.tree = prune(this.tree);
  }

  _byId(id) {
    return this.panels.find((p) => p.id === id);
  }

  // Drag a panel by its header onto another panel: left/right quarter = new column
  // beside it, top/bottom half = stack above/below it.
  _bindDrag(panel, head) {
    head.style.cursor = "grab";
    head.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || e.target.closest("select, button, input")) return;
      e.preventDefault();
      const start = [e.clientX, e.clientY];
      let dragging = false;
      const move = (ev) => {
        if (!dragging && Math.hypot(ev.clientX - start[0], ev.clientY - start[1]) < 6) return;
        if (!dragging) { dragging = true; panel.el.classList.add("dragging"); }
        this._showHint(ev, panel);
      };
      const up = (ev) => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        this._clearHint();
        if (!dragging) return;
        panel.el.classList.remove("dragging");
        const t = this._dropTarget(ev, panel);
        if (t) this._dock(panel.id, t.panel.id, t.zone);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
  }

  _dropTarget(ev, dragPanel) {
    const el = document.elementFromPoint(ev.clientX, ev.clientY);
    const host = el && el.closest(".vpanel");
    if (!host) return null;
    const p = this.panels.find((q) => q.el === host);
    if (!p || p.id === dragPanel.id) return null;
    const r = host.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width;
    const fy = (ev.clientY - r.top) / r.height;
    const zone = fx < 0.25 ? "left" : fx > 0.75 ? "right" : fy < 0.5 ? "top" : "bottom";
    return { panel: p, zone };
  }

  _showHint(ev, dragPanel) {
    const t = this._dropTarget(ev, dragPanel);
    this._clearHint();
    if (!t) return;
    if (!this._hintEl) {
      this._hintEl = document.createElement("div");
      this._hintEl.className = "drop-hint";
    }
    this._hintEl.dataset.zone = t.zone;
    t.panel.el.appendChild(this._hintEl);
  }

  _clearHint() {
    if (this._hintEl) this._hintEl.remove();
  }

  // Dock `dragId` against `targetId`: left/right splits the target's CELL into
  // a row, top/bottom into a column. When the target's parent already splits in
  // that direction, the dragged view slides in beside it instead of nesting.
  _dock(dragId, targetId, zone) {
    if (dragId === targetId) return;
    this._dropFromLayout(dragId);
    const dir = zone === "left" || zone === "right" ? "row" : "col";
    const first = zone === "left" || zone === "top";
    const drag = { t: "leaf", id: dragId };
    const place = (node, parent) => {
      if (node.t === "leaf") {
        if (node.id !== targetId) return false;
        if (parent && parent.dir === dir) {
          const i = parent.kids.indexOf(node);
          parent.kids.splice(first ? i : i + 1, 0, drag);
        } else {
          const split = { t: "split", dir, kids: first ? [drag, node] : [node, drag] };
          if (parent) parent.kids[parent.kids.indexOf(node)] = split;
          else this.tree = split;
        }
        return true;
      }
      return node.kids.some((k) => place(k, node));
    };
    if (!this.tree) this.tree = drag;
    else if (!place(this.tree, null)) {
      this.tree = { t: "split", dir: "row", kids: [this.tree, drag] };  // lost target: new column
    }
    this._relayout();
    if (this.onChanged) this.onChanged();
  }

  // ------------------------------------------------- per-view settings strip
  _buildOpts(panel, mode) {
    if (panel.kind === "3d") this._buildOpts3d(panel, mode);
    else if (panel.kind === "bev") this._buildOptsBev(panel);
    else this._buildOptsImg(panel);
  }

  _buildOptsBev(panel) {
    const box = panel.optsEl;
    box.replaceChildren();
    const view = panel.view;

    const sizeLab = document.createElement("label");
    sizeLab.className = "opt";
    sizeLab.textContent = "size";
    const size = document.createElement("input");
    size.type = "range"; size.min = "1"; size.max = "5"; size.step = "0.2";
    size.value = String(view.dot);
    size.oninput = () => view.setDot(+size.value);
    size.onchange = () => { if (this.onChanged) this.onChanged(); };
    sizeLab.appendChild(size);

    const refit = document.createElement("button");
    refit.className = "tbtn";
    refit.textContent = "Re-fit";
    refit.title = "Re-frame the view on the current cloud";
    refit.onclick = () => view.refit();
    box.append(sizeLab, refit);
  }

  _buildOptsImg(panel) {
    const box = panel.optsEl;
    box.replaceChildren();

    const fitLab = document.createElement("label");
    fitLab.className = "opt";
    fitLab.textContent = "fit";
    const fit = document.createElement("select");
    fillSelect(fit, [["contain", "Fit"], ["fill", "Fill"], ["actual", "1:1 pixels"]]);
    fit.value = panel.imgCfg.fit;
    fit.onchange = () => {
      panel.imgCfg.fit = fit.value;
      this._applyImgCfg(panel);
      if (this.onChanged) this.onChanged();
    };
    fitLab.appendChild(fit);

    const smoothLab = document.createElement("label");
    smoothLab.className = "opt";
    const smooth = document.createElement("input");
    smooth.type = "checkbox";
    smooth.checked = panel.imgCfg.smooth;
    smooth.onchange = () => {
      panel.imgCfg.smooth = smooth.checked;
      this._applyImgCfg(panel);
      if (this.onChanged) this.onChanged();
    };
    smoothLab.append(smooth, document.createTextNode(" smooth"));

    // Channel order: rosbag extractions are OpenCV bgr8 with no metadata left,
    // so a red sky means "flip me" — the choice is per view and persisted.
    const orderLab = document.createElement("label");
    orderLab.className = "opt";
    orderLab.textContent = "channels";
    const order = document.createElement("select");
    fillSelect(order, [["rgb", "RGB"], ["bgr", "BGR (OpenCV)"]]);
    order.value = panel.imgCfg.order;
    order.title = "Red sky? The source stores BGR (OpenCV/rosbag) — flip it here.";
    order.onchange = () => {
      panel.imgCfg.order = order.value;
      panel._lastImg = null;                 // force a redraw with the new order
      if (this.frame) this._render(panel);
      if (this.onChanged) this.onChanged();
    };
    orderLab.appendChild(order);

    box.append(fitLab, smoothLab, orderLab);
    this._applyImgCfg(panel);
  }

  _applyImgCfg(panel) {
    panel.body.dataset.fit = panel.imgCfg.fit;
    panel.body.dataset.smooth = panel.imgCfg.smooth ? "on" : "off";
  }

  // Wheel = zoom on the cursor, drag = pan (once zoomed), double-click = reset.
  // A CSS transform over the canvas, so it costs nothing per frame.
  _bindImgZoom(panel) {
    const z = { k: 1, x: 0, y: 0 };
    panel.zoom = z;
    const apply = () => {
      panel.canvas.style.transform = z.k === 1 ? "" : `translate(${z.x}px, ${z.y}px) scale(${z.k})`;
      panel.canvas.style.transformOrigin = "center";
      panel.body.classList.toggle("zoomed", z.k !== 1);
    };
    // Cursor offset from the canvas's UNtransformed center (layout box).
    const center = () => {
      const br = panel.body.getBoundingClientRect();
      return [
        br.left + panel.canvas.offsetLeft + panel.canvas.offsetWidth / 2,
        br.top + panel.canvas.offsetTop + panel.canvas.offsetHeight / 2,
      ];
    };
    panel.body.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [cx, cy] = center();
      const mx = e.clientX - cx, my = e.clientY - cy;
      const k2 = Math.min(24, Math.max(1, z.k * Math.exp(-e.deltaY * 0.0018)));
      const f = k2 / z.k;
      z.x = mx - f * (mx - z.x);   // keep the texel under the cursor fixed
      z.y = my - f * (my - z.y);
      z.k = k2;
      if (z.k === 1) { z.x = 0; z.y = 0; }
      apply();
    }, { passive: false });
    panel.body.addEventListener("mousedown", (e) => {
      if (z.k === 1 || e.button !== 0) return;
      e.preventDefault();
      let [px, py] = [e.clientX, e.clientY];
      const move = (ev) => {
        z.x += ev.clientX - px; z.y += ev.clientY - py;
        px = ev.clientX; py = ev.clientY;
        apply();
      };
      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
    panel.body.addEventListener("dblclick", () => { z.k = 1; z.x = 0; z.y = 0; apply(); });
  }

  _buildOpts3d(panel, mode) {
    const box = panel.optsEl;
    box.replaceChildren();
    const view = panel.view;

    // Point size — always available.
    const sizeLab = document.createElement("label");
    sizeLab.className = "opt";
    sizeLab.textContent = "size";
    const size = document.createElement("input");
    size.type = "range"; size.min = "0.04"; size.max = "0.6"; size.step = "0.02";
    size.value = String(view.points.material.size);
    size.oninput = () => view.setPointSize(+size.value);
    sizeLab.appendChild(size);
    box.appendChild(sizeLab);

    // Controls style: stable z-up orbit, or free trackball (tumble a cloud
    // recorded without a mount TF upright).
    {
      const lab = document.createElement("label");
      lab.className = "opt";
      lab.textContent = "controls";
      const sel = document.createElement("select");
      fillSelect(sel, [["orbit", "Orbit"], ["trackball", "Trackball"]]);
      sel.value = view.controlStyle;
      sel.onchange = () => {
        view.setControlStyle(sel.value);
        if (this.onChanged) this.onChanged();
      };
      lab.appendChild(sel);
      box.appendChild(lab);
    }

    // Cloud frame (needs a pose): auto-guessed, overridable when the guess is off.
    if (this.poseKey) {
      const lab = document.createElement("label");
      lab.className = "opt";
      lab.textContent = "frame";
      const sel = document.createElement("select");
      fillSelect(sel, [["auto", "Auto"], ["world", "World"], ["ego", "Ego"]]);
      sel.value = view.frameMode;
      sel.title = "Frame the cloud lives in — aligns trajectory & ego marker";
      sel.onchange = () => view.setFrameMode(sel.value);
      lab.appendChild(sel);
      box.appendChild(lab);
    }

    if (mode === "height" || mode === "intensity") {
      for (const key of ["vmin", "vmax"]) {
        const lab = document.createElement("label");
        lab.className = "opt";
        lab.textContent = key;
        const inp = document.createElement("input");
        inp.type = "number"; inp.step = "any"; inp.placeholder = "auto";
        inp.className = "opt-num";
        if (view.cfg[key] !== null) inp.value = String(view.cfg[key]);
        inp.onchange = () => view.set({ [key]: inp.value === "" ? null : +inp.value });
        lab.appendChild(inp);
        box.appendChild(lab);
      }
      return;
    }

    if (mode.startsWith("labels:")) {
      const name = mode.slice(7);
      const spec = panel.labelings.find((c) => c.name === name);
      if (!spec || !spec.labelset) return;
      view.set({ hidden: new Set() });                  // reset visibility on (re)entry
      const row = document.createElement("span");
      row.className = "chips";
      for (const cls of spec.labelset.classes) {
        const chip = this._chip(cls.name, cls.color, true, (on) => {
          const hidden = new Set(view.cfg.hidden);
          if (on) hidden.delete(cls.id); else hidden.add(cls.id);
          view.set({ hidden });
        });
        row.appendChild(chip);
      }
      box.appendChild(row);
      return;
    }

    if (mode === "confusion") {
      const labs = panel.labelings;
      const current = view.cfg.confusion || {};
      const cfg = {
        pred: current.pred || (labs[1] ? labs[1].name : labs[0].name),
        ref: current.ref || labs[0].name,
        ...this._binDefaults(labs, current.pred || (labs[1] ? labs[1].name : labs[0].name),
                             current.ref || labs[0].name),
        hiddenCats: current.hiddenCats || new Set(),
      };
      view.set({ confusion: cfg });

      const mkSel = (label, key) => {
        const lab = document.createElement("label");
        lab.className = "opt";
        lab.textContent = label;
        const sel = document.createElement("select");
        fillSelect(sel, labs.map((c) => [c.name, pretty(c.name)]));
        sel.value = cfg[key];
        sel.onchange = () => {
          cfg[key] = sel.value;
          Object.assign(cfg, this._binDefaults(labs, cfg.pred, cfg.ref));
          view.set({ confusion: cfg });
          this._buildOpts(panel, "confusion");          // re-render the chips
          this._afterRender(panel);
        };
        lab.appendChild(sel);
        return lab;
      };
      box.append(mkSel("pred", "pred"), mkSel("ref", "ref"));

      // Binarization chips. pred: pos/neg. ref: pos/neg/off (off = not scored).
      box.appendChild(this._binChips(panel, cfg, "pred"));
      box.appendChild(this._binChips(panel, cfg, "ref"));

      // Category visibility (e.g. keep only FP+FN = "show me where the model is wrong").
      const cats = document.createElement("span");
      cats.className = "chips";
      CONFUSION.forEach((cat, ci) => {
        const chip = this._chip(cat.name, cat.color, !cfg.hiddenCats.has(ci), (on) => {
          if (on) cfg.hiddenCats.delete(ci); else cfg.hiddenCats.add(ci);
          view.set({ confusion: cfg });
        });
        cats.appendChild(chip);
      });
      box.appendChild(cats);

      const metrics = document.createElement("span");
      metrics.className = "metrics";
      panel.metricsEl = metrics;
      box.appendChild(metrics);
    }
  }

  // Default binarization per labeling: ignore_id → off (ref only), 1 (or the max id)
  // → positive, the rest → negative. The chips are there because no default is right
  // for every labeling.
  _binDefaults(labs, predName, refName) {
    const idsOf = (name) => {
      const spec = labs.find((c) => c.name === name);
      return spec && spec.labelset ? spec.labelset : { classes: [], ignore_id: null };
    };
    const pick = (ls) => {
      const ids = ls.classes.map((c) => c.id);
      return ids.includes(1) ? 1 : Math.max(...ids);
    };
    const pred = idsOf(predName), ref = idsOf(refName);
    return {
      predPos: new Set([pick(pred)]),
      refPos: new Set([pick(ref)]),
      refOff: new Set(ref.classes.filter((c) => c.id === ref.ignore_id).map((c) => c.id)),
    };
  }

  _binChips(panel, cfg, side) {
    // side = "pred" (pos/neg cycle) | "ref" (pos/neg/off cycle)
    const spec = panel.labelings.find((c) => c.name === cfg[side]);
    const row = document.createElement("span");
    row.className = "chips";
    const head = document.createElement("span");
    head.className = "chips-head";
    head.textContent = side === "pred" ? "pred:" : "ref:";
    row.appendChild(head);
    if (!spec || !spec.labelset) return row;

    const stateOf = (id) => {
      if (side === "ref" && cfg.refOff.has(id)) return "off";
      return (side === "pred" ? cfg.predPos : cfg.refPos).has(id) ? "pos" : "neg";
    };
    for (const cls of spec.labelset.classes) {
      const chip = document.createElement("button");
      chip.className = "chip";
      const paint = () => {
        const st = stateOf(cls.id);
        chip.dataset.state = st;
        chip.textContent = `${pretty(cls.name)} ${st === "pos" ? "+" : st === "neg" ? "−" : "∅"}`;
        chip.style.setProperty("--chip", rgbCss(cls.color));
      };
      chip.title = side === "pred"
        ? "Cycle: positive + / negative −"
        : "Cycle: positive + / negative − / ∅ not scored";
      chip.onclick = () => {
        const pos = side === "pred" ? cfg.predPos : cfg.refPos;
        const st = stateOf(cls.id);
        if (side === "ref") {
          if (st === "pos") { pos.delete(cls.id); }
          else if (st === "neg") { cfg.refOff.add(cls.id); }
          else { cfg.refOff.delete(cls.id); pos.add(cls.id); }
        } else {
          if (st === "pos") pos.delete(cls.id); else pos.add(cls.id);
        }
        paint();
        panel.view.set({ confusion: cfg });
        this._afterRender(panel);
      };
      paint();
      row.appendChild(chip);
    }
    return row;
  }

  _chip(text, color, on, onToggle) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = pretty(text);
    chip.style.setProperty("--chip", rgbCss(color));
    chip.dataset.state = on ? "pos" : "off";
    chip.onclick = () => {
      const now = chip.dataset.state !== "pos";
      chip.dataset.state = now ? "pos" : "off";
      onToggle(now);
    };
    return chip;
  }

  // ------------------------------------------------------------- rendering
  remove(id) {
    const i = this.panels.findIndex((p) => p.id === id);
    if (i < 0) return;
    const [p] = this.panels.splice(i, 1);
    if (p.view) p.view.dispose();
    this._dropFromLayout(id);
    this._relayout();
    this._renderMenu();
  }

  dispose() {
    for (const p of this.panels) if (p.view) p.view.dispose();
    this.panels = [];
    this.tree = null;
    this.stack.replaceChildren();
    if (this.menuEl) this.menuEl.replaceChildren();
  }

  // Merge each frame over the previous one (same sequence): on an async source a
  // frame carries only the channels that ticked; the views keep the last value of
  // the others. A sequence switch resets the merge.
  //
  // Exception: a labeling is dropped when its cloud ticked without it — stale
  // labels on a fresh scan would color it wrong (silently so on fixed-size
  // sensors, where the length check cannot catch it).
  update(frame) {
    // Per-channel display index, recorded BEFORE the merge: a channel that did
    // not tick in this frame keeps its previous counter (apairo_rr semantics).
    if (!this.frame || this.frame.seq !== frame.seq) this.chanIndex = {};
    this.chanIndex = this.chanIndex || {};
    for (const k of Object.keys(frame.channels)) {
      this.chanIndex[k] = frame.indices && k in frame.indices ? frame.indices[k] : frame.index;
    }
    if (!this.frame || this.frame.seq !== frame.seq) {
      this.frame = { ...frame, channels: { ...frame.channels } };
    } else {
      const merged = Object.assign(this.frame.channels, frame.channels);
      for (const c of this.labelings) {
        if (c.of in frame.channels && !(c.name in frame.channels)) {
          delete merged[c.name];
        }
      }
      this.frame = { ...frame, channels: merged };
    }
    for (const p of this.panels) this._render(p);
  }

  _render(panel) {
    if (panel.hiddenP || panel.paused) return;   // frozen views keep their last frame
    this._paintIdx(panel);
    if (panel.kind === "img") {
      const img = this.frame.channels[panel.channel];
      if (img && img !== panel._lastImg) {
        drawImage(panel.canvas, img, panel.imgCfg.order === "bgr");
        panel._lastImg = img;
      }
      return;
    }
    panel.view.setFrame(this.frame);
    this._afterRender(panel);
  }

  // Frame counter of the view's channel — its own event index on async rigs,
  // the global frame index otherwise. Paused views freeze it with their frame.
  _paintIdx(panel) {
    const i = this.chanIndex && panel.channel in this.chanIndex
      ? this.chanIndex[panel.channel] : this.frame.index;
    panel.idxEl.textContent = `#${i}`;
    const ts = this.frame.timestamps && this.frame.timestamps[panel.channel];
    if (ts !== undefined && ts !== null) panel.idxEl.title = `frame #${i} · t=${ts.toFixed(3)}s`;
  }

  _afterRender(panel) {
    if (panel.kind !== "3d") return;
    panel.warn.textContent = panel.view.lastWarning || "";
    if (panel.metricsEl) {
      const m = panel.view.lastMetrics;
      const fmt = (v) => (v === null ? "—" : v.toFixed(3));
      panel.metricsEl.textContent = m
        ? `P ${fmt(m.precision)} · R ${fmt(m.recall)} · IoU ${fmt(m.iou)}`
        : "";
    }
  }

  // Rebuild the mosaic from the split tree, re-inserting the persistent panel
  // and split elements so flex sizes survive a relayout. Hidden panels (and
  // splits they empty) are left out; a split with one visible kid dissolves.
  _relayout() {
    const render = (node) => {
      if (node.t === "leaf") {
        const p = this._byId(node.id);
        return !p || p.hiddenP ? null : p.el;
      }
      const parts = node.kids.map(render).filter(Boolean);
      if (!parts.length) return null;
      if (parts.length === 1) return parts[0];
      if (!node.el) {
        node.el = document.createElement("div");
        node.el.style.flex = "1 1 0";
      }
      node.el.className = `vsplit vsplit-${node.dir}`;
      node.el.replaceChildren();
      parts.forEach((el, i) => {
        if (i > 0) {
          const g = document.createElement("div");
          g.className = node.dir === "row" ? "gutter gutter-v" : "gutter gutter-h";
          node.el.appendChild(g);
          growGutter(g, parts[i - 1], el, node.dir === "row" ? "x" : "y", node.dir === "row" ? 160 : 90);
        }
        node.el.appendChild(el);
      });
      return node.el;
    };
    this.stack.replaceChildren();
    if (!this.tree) return;
    const rootEl = render(this.tree);
    if (rootEl) this.stack.appendChild(rootEl);
  }
}

export function fillSelect(sel, pairs) {
  sel.replaceChildren();
  for (const [value, label] of pairs) {
    const o = document.createElement("option");
    o.value = value; o.textContent = label;
    sel.appendChild(o);
  }
}
