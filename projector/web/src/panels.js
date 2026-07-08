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
    this.columns = [];        // mosaic layout: [{el, panels: [id, ...]}, ...]
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
    return this.columns.map((c) =>
      c.panels.map((id) => {
        const p = this._byId(id);
        const out = { kind: p.kind, channel: p.channel, hidden: p.hiddenP, paused: p.paused };
        if (p.kind === "3d") {
          out.colorBy = p.view.colorBy;
          out.cameraMode = p.view.cameraMode;
          out.frameMode = p.view.frameMode;
          out.size = p.view.points.material.size;
          out.controls = p.view.controlStyle;
        }
        return out;
      }),
    );
  }

  // Rebuild panels from a serialized layout; false when nothing applied
  // (e.g. the channels changed since the state was saved).
  restore(cols) {
    const valid = new Set([...this.cloudKeys, ...this.imageKeys]);
    let added = 0;
    (cols || []).forEach((col, ci) => {
      for (const spec of col || []) {
        if (!valid.has(spec.channel)) continue;
        const p = this.add(spec.kind, spec.channel, ci);
        added++;
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
        if (spec.hidden) p.hiddenP = true;
        if (spec.paused) p.paused = true;
      }
    });
    if (added) {
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
    const warn = document.createElement("span");
    warn.className = "vpanel-warn";
    const close = document.createElement("button");
    close.className = "tbtn close";
    close.textContent = "✕";
    close.title = "Close view";

    const body = document.createElement("div");
    body.className = "vpanel-body " + (kind === "img" ? "img-body" : "canvas-body");

    const panel = { id, kind, channel, el, body, warn, view: null, canvas: null };

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

      const gear = document.createElement("button");
      gear.className = "tbtn gear";
      gear.textContent = "⚙";
      gear.title = "Display options";
      head.append(tag, name, warn, colorSel, camSel, gear, close);

      const optsEl = document.createElement("div");
      optsEl.className = "vopts";
      optsEl.hidden = true;
      panel.optsEl = optsEl;
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
      gear.onclick = () => { optsEl.hidden = !optsEl.hidden; };
      this._buildOpts(panel, "height");

      el.append(head, optsEl, body);
    } else if (kind === "bev") {
      head.append(tag, name, warn, close);
      panel.view = new BevView(body);
      panel.view.setPoseChannel(this.poseKey);
      panel.view.setChannel(channel);
      if (this.trajectory) panel.view.setTrajectory(this.trajectory);
      el.append(head, body);
    } else {
      head.append(tag, name, warn, close);
      panel.canvas = document.createElement("canvas");
      body.appendChild(panel.canvas);
      el.append(head, body);
    }

    close.onclick = () => this.remove(id);
    this._bindDrag(panel, head);
    panel.hiddenP = false;
    panel.paused = false;
    this.panels.push(panel);
    this._place(id, col);
    this._relayout();
    this._renderMenu();
    if (this.frame) this._render(panel);
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

  // ------------------------------------------------- mosaic layout (columns of rows)
  _newCol() {
    const el = document.createElement("div");
    el.className = "vcol";
    el.style.flex = "1 1 0";
    return el;
  }

  _place(id, col) {
    if (col === null) col = Math.max(0, this.columns.length - 1);
    while (this.columns.length <= col) this.columns.push({ el: this._newCol(), panels: [] });
    this.columns[col].panels.push(id);
  }

  _dropFromLayout(id) {
    for (const c of this.columns) {
      const i = c.panels.indexOf(id);
      if (i >= 0) { c.panels.splice(i, 1); break; }
    }
    this.columns = this.columns.filter((c) => c.panels.length);
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

  _dock(dragId, targetId, zone) {
    this._dropFromLayout(dragId);
    let ci = -1, pi = -1;
    this.columns.forEach((c, i) => {
      const j = c.panels.indexOf(targetId);
      if (j >= 0) { ci = i; pi = j; }
    });
    if (ci < 0) {
      this.columns.push({ el: this._newCol(), panels: [dragId] });
    } else if (zone === "top" || zone === "bottom") {
      this.columns[ci].panels.splice(zone === "top" ? pi : pi + 1, 0, dragId);
    } else {
      const col = { el: this._newCol(), panels: [dragId] };
      this.columns.splice(zone === "left" ? ci : ci + 1, 0, col);
    }
    this._relayout();
    if (this.onChanged) this.onChanged();
  }

  // ------------------------------------------------- options strip (3D panels)
  _buildOpts(panel, mode) {
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
    this.columns = [];
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
    if (panel.kind === "img") {
      const img = this.frame.channels[panel.channel];
      if (img && img !== panel._lastImg) { drawImage(panel.canvas, img); panel._lastImg = img; }
      return;
    }
    panel.view.setFrame(this.frame);
    this._afterRender(panel);
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

  // Rebuild the mosaic (columns, panels, resize handles) by re-inserting the
  // existing nodes — panel and column elements persist, so flex sizes survive.
  // Hidden panels (and columns left empty by them) are simply left out.
  _relayout() {
    this.stack.replaceChildren();
    let prevCol = null;
    for (const c of this.columns) {
      const visible = c.panels.map((id) => this._byId(id)).filter((p) => !p.hiddenP);
      if (!visible.length) continue;
      if (prevCol) {
        const g = document.createElement("div");
        g.className = "gutter gutter-v";
        this.stack.appendChild(g);
        growGutter(g, prevCol.el, c.el, "x", 160);
      }
      c.el.replaceChildren();
      visible.forEach((p, pi) => {
        if (pi > 0) {
          const g = document.createElement("div");
          g.className = "gutter gutter-h";
          c.el.appendChild(g);
          growGutter(g, visible[pi - 1].el, p.el, "y", 90);
        }
        c.el.appendChild(p.el);
      });
      this.stack.appendChild(c.el);
      prevCol = c;
    }
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
