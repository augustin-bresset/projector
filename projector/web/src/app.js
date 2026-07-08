// Projector front — bootstrap, rail (sequences / channels / views / time /
// bookmarks / accumulate / frames), timeline with per-channel tracks, A-B loop,
// Open-dataset modal with shell-style Tab completion, per-dataset session state.
//
// Frames arrive over the binary websocket (src/ws.js) with client-side caching and
// prefetch; the REST /api/frame endpoint remains for debugging only.

import { api } from "./api.js";
import { FrameStream } from "./ws.js";
import { PanelManager, fillSelect } from "./panels.js";
import { pretty } from "./colors.js";
import { basisGutter } from "./resize.js";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const BASE_FPS = 10;
const PREFETCH = 4;

const stream = new FrameStream();
let session = null;
let panels = null;
let seq = null;          // current sequence id
let nFrames = 1;
let index = 0;
let reqId = 0;           // only the latest in-flight show() is applied
let playing = false;
let speed = 1;
let accum = { back: 0, fwd: 0 };   // pose-registered accumulation (server-side)
let tsOrigin = null;     // first timestamp seen in the sequence (Frames panel display)
let tsSeen = {};         // channel → last event timestamp
let loopA = null;        // A-B loop bounds (frame indices in the current sequence)
let loopB = null;
let bookmarks = [];      // [{seq, index}]
let timeline = null;     // {channels: {name: Float64Array}, ticks: Float64Array|null}
let stateTimer = null;   // debounced session-state save

function applyFrame(frame) {
  nFrames = frame.nFrames;
  panels.update(frame);
  updateFramesPanel(frame);
  syncFooter();
  drawTracks(frame);
}

async function show(i) {
  if (!seq) return;
  index = Math.max(0, Math.min(i, nFrames - 1));
  const id = ++reqId;
  const frame = await stream.request(seq, index, accum.back, accum.fwd);
  if (id !== reqId) return;                      // superseded while scrubbing
  applyFrame(frame);
  stream.prefetch(seq, index, 2, nFrames, accum.back, accum.fwd);
  saveStateSoon();
}

function syncFooter() {
  $("frame").max = nFrames - 1;
  $("frame").value = index;
  $("frame-label").textContent = `frame ${index + 1} / ${nFrames}`;
  $("status").textContent = seq ? `${seq} · frame ${index + 1}/${nFrames}` : "";
}

// ------------------------------------------------------- Frames panel (async rigs)
function updateFramesPanel(frame) {
  const ts = frame.timestamps;
  const panel = $("frames-panel");
  if (!ts || !Object.keys(ts).length) { panel.hidden = true; return; }
  panel.hidden = false;
  for (const [ch, t] of Object.entries(ts)) {
    if (tsOrigin === null || t < tsOrigin) tsOrigin = t;
    tsSeen[ch] = t;
  }
  const box = $("frames");
  box.replaceChildren();
  for (const [ch, t] of Object.entries(tsSeen).sort()) {
    const row = document.createElement("div");
    row.className = "frame-row";
    const name = document.createElement("span");
    name.textContent = pretty(ch);
    const val = document.createElement("span");
    val.className = "frame-ts";
    val.textContent = `+${(t - tsOrigin).toFixed(3)}s`;
    row.append(name, val);
    box.appendChild(row);
  }
}

// --------------------------------------------------- per-channel tracks (footer)
function loadTimeline(id) {
  timeline = null;
  $("tracks").hidden = true;
  api.timeline(id).then((tl) => {
    if (seq !== id || !tl || !tl.channels) return;
    timeline = tl;
    $("tracks").hidden = !$("btn-tracks").classList.contains("active");
    drawTracks(null);
  }).catch(() => {});
}

function currentTime(frame) {
  if (timeline && timeline.ticks) return timeline.ticks.data[index];
  if (frame && frame.timestamps) {
    const vals = Object.values(frame.timestamps);
    if (vals.length) return Math.max(...vals);
  }
  return null;
}

function drawTracks(frame) {
  const strip = $("tracks");
  if (strip.hidden || !timeline) return;
  const names = Object.keys(timeline.channels).sort();
  const canvas = $("tracks-canvas");
  const dpr = window.devicePixelRatio || 1;
  const W = strip.clientWidth - 110, H = names.length * 16 + 4;
  canvas.style.height = `${H}px`;
  canvas.width = Math.max(1, W * dpr);
  canvas.height = Math.max(1, H * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  let t0 = Infinity, t1 = -Infinity;
  for (const n of names) {
    const a = timeline.channels[n].data;
    if (a.length) { t0 = Math.min(t0, a[0]); t1 = Math.max(t1, a[a.length - 1]); }
  }
  if (!(t1 > t0)) return;
  const x = (t) => ((t - t0) / (t1 - t0)) * W;

  const labels = $("tracks-labels");
  labels.replaceChildren();
  names.forEach((n, r) => {
    const lab = document.createElement("div");
    lab.textContent = pretty(n);
    labels.appendChild(lab);
    const a = timeline.channels[n].data;
    ctx.strokeStyle = "rgba(255,176,0,0.55)";
    ctx.beginPath();
    const y0 = r * 16 + 3, y1 = r * 16 + 13;
    const step = Math.max(1, Math.floor(a.length / W));   // ≤ ~1 tick per px
    for (let i = 0; i < a.length; i += step) {
      const px = x(a[i]);
      ctx.moveTo(px, y0);
      ctx.lineTo(px, y1);
    }
    ctx.stroke();
  });

  const t = currentTime(frame);
  if (t !== null) {
    ctx.strokeStyle = "#ece7dd";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x(t), 0);
    ctx.lineTo(x(t), H);
    ctx.stroke();
  }
}

function tracksSeek(e) {
  if (!timeline || !timeline.ticks) return;      // index↔time only 1:1 on event timelines
  const canvas = $("tracks-canvas");
  const r = canvas.getBoundingClientRect();
  const names = Object.keys(timeline.channels);
  let t0 = Infinity, t1 = -Infinity;
  for (const n of names) {
    const a = timeline.channels[n].data;
    if (a.length) { t0 = Math.min(t0, a[0]); t1 = Math.max(t1, a[a.length - 1]); }
  }
  const t = t0 + ((e.clientX - r.left) / r.width) * (t1 - t0);
  const ticks = timeline.ticks.data;
  let lo = 0, hi = ticks.length - 1;             // nearest tick by bisection
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ticks[mid] < t) lo = mid; else hi = mid;
  }
  setPlaying(false);
  show(Math.abs(ticks[lo] - t) <= Math.abs(ticks[hi] - t) ? lo : hi);
}

// -------------------------------------------------------------- A-B loop
function setLoopBound(which) {
  if (which === "a") loopA = loopA === index ? null : index;
  else loopB = loopB === index ? null : index;
  if (loopA !== null && loopB !== null && loopB < loopA) [loopA, loopB] = [loopB, loopA];
  paintLoop();
  saveStateSoon();
}

function paintLoop() {
  $("btn-loop-a").classList.toggle("active", loopA !== null);
  $("btn-loop-b").classList.toggle("active", loopB !== null);
  const f = $("frame");
  if (loopA !== null && loopB !== null && nFrames > 1) {
    const a = (loopA / (nFrames - 1)) * 100, b = (loopB / (nFrames - 1)) * 100;
    f.style.background = `linear-gradient(90deg, transparent ${a}%,
      color-mix(in srgb, var(--accent) 30%, transparent) ${a}%,
      color-mix(in srgb, var(--accent) 30%, transparent) ${b}%, transparent ${b}%)`;
  } else {
    f.style.background = "";
  }
}

// -------------------------------------------------------------- bookmarks
function renderBookmarks() {
  const box = $("bookmarks");
  const panel = $("bookmarks-panel");
  panel.hidden = bookmarks.length === 0;
  box.replaceChildren();
  for (const bm of bookmarks) {
    const row = document.createElement("div");
    row.className = "bookmark-row";
    const name = document.createElement("span");
    name.className = "bookmark-name";
    name.textContent = `${pretty(bm.seq)} · ${bm.index + 1}`;
    name.onclick = () => {
      setPlaying(false);
      if (bm.seq !== seq) setSequence(bm.seq, bm.index);
      else show(bm.index);
    };
    const del = document.createElement("button");
    del.className = "icon-btn";
    del.textContent = "x";
    del.onclick = () => {
      bookmarks = bookmarks.filter((b) => b !== bm);
      renderBookmarks();
      saveStateSoon();
    };
    row.append(name, del);
    box.appendChild(row);
  }
}

function toggleBookmark() {
  const at = bookmarks.findIndex((b) => b.seq === seq && b.index === index);
  if (at >= 0) bookmarks.splice(at, 1);
  else bookmarks.push({ seq, index });
  renderBookmarks();
  saveStateSoon();
}

// -------------------------------------------------------------- session state
function saveStateSoon() {
  clearTimeout(stateTimer);
  stateTimer = setTimeout(saveState, 1200);
}

async function saveState() {
  if (!session || !session.sequences.length) return;
  try {
    await api.saveState({
      position: { seq, index },
      accum, speed,
      loop: { a: loopA, b: loopB },
      bookmarks,
      views: panels.serialize(),
      plugins: {
        scripts: [...pluginScripts],
        active: pluginActive.map((a) => ({ id: a.id, cloud: a.cloud, params: a.params })),
      },
    });
  } catch { /* stateless server or no dataset key — fine */ }
}

// -------------------------------------------------------------- sequences / rail
function setSequence(id, at = 0) {
  seq = id;
  nFrames = session.sequences.find((s) => s.id === id).n_frames;
  tsOrigin = null;
  tsSeen = {};
  loopA = loopB = null;
  paintLoop();
  for (const b of document.querySelectorAll("#sequences .seq-btn")) {
    b.classList.toggle("active", b.dataset.seq === id);
  }
  setPlaying(false);
  show(at);
  loadTrajectory(id);
  loadTimeline(id);
}

async function loadTrajectory(id) {
  panels.setTrajectory(null);
  try {
    const traj = await api.trajectory(id);
    if (seq === id) panels.setTrajectory(traj);   // ignore a stale (slow) reply
  } catch (e) {
    console.warn("trajectory:", e);
  }
}

function buildRail() {
  const seqBox = $("sequences");
  seqBox.replaceChildren();
  for (const s of session.sequences) {
    const b = document.createElement("button");
    b.className = "seq-btn tbtn";
    b.dataset.seq = s.id;
    b.textContent = `${pretty(s.id)} · ${s.n_frames}`;
    b.onclick = () => setSequence(s.id);
    seqBox.appendChild(b);
  }

  const chBox = $("channels");
  chBox.replaceChildren();
  for (const c of session.channels) {
    const row = document.createElement("div");
    row.className = "channel-item";
    const kind = document.createElement("span");
    kind.className = "kind-badge";
    kind.textContent = c.kind === "pointcloud" ? "3D" :
                       c.kind === "image" ? "IMG" :
                       c.kind === "labels" ? "LBL" :
                       c.kind === "pose" ? "POSE" : "SCL";
    const name = document.createElement("span");
    name.textContent = pretty(c.name) + (c.of ? ` → ${pretty(c.of)}` : "");
    row.append(kind, name);
    chBox.appendChild(row);
  }

  // Time model panel: event timeline vs resampled onto a reference clock.
  const syncPanel = $("sync-panel");
  syncPanel.hidden = !session.sync;
  if (session.sync) {
    const sel = $("sync-ref");
    fillSelect(sel, [["", "event timeline"],
                     ...session.channels.map((c) => [c.name, `resample on ${pretty(c.name)}`])]);
    sel.value = session.sync.reference || "";
    $("sync-tol").value = String(session.sync.tolerance);
    $("sync-tol-row").hidden = !sel.value;
    sel.onchange = () => { $("sync-tol-row").hidden = !sel.value; };
  }
}

async function applySync() {
  const btn = $("sync-apply");
  btn.disabled = true;
  $("status").textContent = "resampling…";
  try {
    const s = await api.sync({
      reference: $("sync-ref").value || null,
      tolerance: +$("sync-tol").value || 0.1,
    });
    initSession(s);
  } catch (e) {
    $("status").textContent = String(e.message || e);
  } finally {
    btn.disabled = false;
  }
}

// -------------------------------------------------------------- playback
function setPlaying(on) {
  if (playing === on) return;
  playing = on;
  $("btn-play").textContent = on ? "⏸" : "▶";
  if (on) playLoop();
}

async function playLoop() {
  while (playing) {
    const t0 = performance.now();
    let next = index + 1;
    if (loopA !== null && loopB !== null && (next > loopB || next < loopA)) {
      next = loopA;                                // A-B loop wraps
    } else if (index >= nFrames - 1) {
      setPlaying(false);
      return;
    }
    const frame = await stream.request(seq, next, accum.back, accum.fwd);
    if (!playing) return;
    index = next;
    reqId++;                                       // playback owns the position now
    applyFrame(frame);
    stream.prefetch(seq, index, PREFETCH, nFrames, accum.back, accum.fwd);
    const budget = 1000 / (BASE_FPS * speed);
    const dt = performance.now() - t0;
    await sleep(Math.max(0, budget - dt));
  }
}

// ------------------------------------------------------------ session (re)boot
async function initSession(s) {
  session = s;
  document.title = session.title || "Projector";
  setPlaying(false);
  stream.clear();
  seq = null;
  index = 0;
  nFrames = 1;
  reqId++;
  loopA = loopB = null;
  timeline = null;
  buildRail();

  if (panels) panels.dispose();
  panels = new PanelManager($("views-stack"), session, $("views-menu"));
  panels.onChanged = saveStateSoon;
  window.__projector = { panels, stream, session };   // debug hook (console)
  fillSelect($("add-kind"), panels.addOptions());

  let st = null;
  try {
    st = (await api.state()).state;
  } catch { /* no sidecar */ }

  if (!(st && st.views && panels.restore(st.views))) {
    // Default layout: big 3D on the left column, image + BEV stacked on the right.
    if (panels.cloudKeys.length) panels.add("3d", panels.cloudKeys[0], 0);
    if (panels.imageKeys.length) panels.add("img", panels.imageKeys[0], 1);
    if (panels.cloudKeys.length) panels.add("bev", panels.cloudKeys[0], 1);
  }

  bookmarks = (st && st.bookmarks) || [];
  renderBookmarks();
  accum = (st && st.accum && typeof st.accum === "object") ? st.accum : { back: 0, fwd: 0 };
  $("accum-back").value = String(accum.back);
  $("accum-fwd").value = String(accum.fwd);
  $("accum-panel").hidden = !session.channels.some((c) => c.kind === "pose");
  speed = (st && st.speed) || 1;
  $("speed").value = String(speed);
  $("frames-panel").hidden = true;
  renderPlugins();

  if (session.sequences.length) {
    const pos = st && st.position && session.sequences.some((q) => q.id === st.position.seq)
      ? st.position : null;
    setSequence(pos ? pos.seq : session.sequences[0].id, pos ? pos.index : 0);
    if (st && st.loop && st.loop.a !== null && st.loop.b !== null) {
      loopA = st.loop.a;
      loopB = st.loop.b;
      paintLoop();
    }
  } else {
    $("status").textContent = "no dataset — Open…";
    syncFooter();
    openModal();
  }
}

// ------------------------------------------------------------ Open-dataset modal
let fsPath = null;        // directory currently listed
let fsSelected = null;    // probed dataset {path, channels, is_synchronous, sequences}
let fsCycle = null;       // Tab/arrow completion cycle {base, names, idx}
let fsMode = "dataset";   // "dataset" (open flow) | "script" (pick a transforms .py)

function openModal(mode = "dataset") {
  fsMode = mode;
  $("fs-modal").hidden = false;
  $("fs-title").textContent = mode === "script" ? "Load a transforms script" : "Open an apairo dataset";
  $("fs-input").focus();
  browse(fsPath);
}
function closeModal() { $("fs-modal").hidden = true; }

function renderListing(listing) {
  fsPath = listing.path;
  const box = $("fs-list");
  box.replaceChildren();
  if (listing.parent) {
    const up = document.createElement("div");
    up.className = "fs-entry dir";
    up.textContent = "..";
    up.onclick = () => browse(listing.parent);
    box.appendChild(up);
  }
  for (const d of listing.dirs) {
    const row = document.createElement("div");
    row.className = "fs-entry dir" + (d.is_dataset ? " dataset" : "");
    row.textContent = d.name + "/";
    const full = `${listing.path}/${d.name}`.replace("//", "/");
    row.onclick = () => browse(full);
    box.appendChild(row);
  }
  for (const f of listing.files || []) {
    const row = document.createElement("div");
    row.className = "fs-entry file";
    row.textContent = f;
    const full = `${listing.path}/${f}`.replace("//", "/");
    row.onclick = () => loadScript(full);
    box.appendChild(row);
  }
}

async function browse(path) {
  fsCycle = null;
  let listing;
  try {
    listing = await api.fs(path, fsMode === "script" ? ".py" : null);
  } catch (e) {
    $("fs-error").textContent = String(e.message || e);
    return;
  }
  $("fs-input").value = listing.path + (listing.path === "/" ? "" : "/");
  $("fs-error").textContent = "";
  renderListing(listing);
  if (fsMode === "dataset" && listing.is_dataset) select(listing.path);
  else $("fs-form").hidden = true;
}

// ------------------------------------------------------- transforms (plugins)
let pluginSpecs = [];     // discovered candidates [{id, name, params, doc}]
let pluginActive = [];    // last applied [{id, cloud, params, error}]
let pluginScripts = new Set();   // loaded script paths (session state)

async function loadScript(path) {
  $("fs-error").textContent = "importing…";
  try {
    const d = await api.pluginsLoad(path);
    pluginSpecs = d.specs;
    pluginScripts.add(path);
    closeModal();
    renderPlugins();
    saveStateSoon();
  } catch (e) {
    $("fs-error").textContent = String(e.message || e);
  }
}

function renderPlugins() {
  const box = $("plugins-list");
  box.replaceChildren();
  const activeById = new Map(pluginActive.map((a) => [a.id, a]));
  for (const spec of pluginSpecs) {
    const row = document.createElement("div");
    row.className = "plugin-row";
    const head = document.createElement("div");
    head.className = "plugin-head";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.dataset.id = spec.id;
    cb.checked = activeById.has(spec.id);
    cb.onchange = () => { $("btn-apply-plugins").hidden = false; };
    const name = document.createElement("span");
    name.className = "plugin-name";
    name.textContent = spec.name;
    name.title = spec.doc || spec.id;
    head.append(cb, name);
    row.appendChild(head);

    const act = activeById.get(spec.id);
    if (Object.keys(spec.params).length) {
      const params = document.createElement("div");
      params.className = "plugin-params";
      for (const [k, v] of Object.entries(spec.params)) {
        const lab = document.createElement("label");
        lab.className = "opt";
        lab.textContent = k;
        const inp = document.createElement("input");
        inp.type = "number";
        inp.step = "any";
        inp.className = "opt-num";
        inp.dataset.id = spec.id;
        inp.dataset.param = k;
        inp.value = String(act && act.params[k] !== undefined ? act.params[k] : v);
        inp.onchange = () => { $("btn-apply-plugins").hidden = false; };
        lab.appendChild(inp);
        params.appendChild(lab);
      }
      row.appendChild(params);
    }
    if (act && act.error) {
      const err = document.createElement("div");
      err.className = "plugin-error";
      err.textContent = act.error;
      row.appendChild(err);
    }
    box.appendChild(row);
  }
  $("btn-apply-plugins").hidden = pluginSpecs.length === 0;
}

async function applyPlugins() {
  const active = [];
  for (const cb of $("plugins-list").querySelectorAll("input[type=checkbox]:checked")) {
    const id = cb.dataset.id;
    const params = {};
    for (const inp of $("plugins-list").querySelectorAll(`input[data-param][data-id="${CSS.escape(id)}"]`)) {
      params[inp.dataset.param] = +inp.value;
    }
    active.push({ id, params });
  }
  $("btn-apply-plugins").disabled = true;
  try {
    const d = await api.pluginsActive(active);
    pluginActive = d.plugins.active;
    stream.clear();                     // transform outputs change the frames
    await initSession(d.session);
  } catch (e) {
    $("status").textContent = String(e.message || e);
  } finally {
    $("btn-apply-plugins").disabled = false;
  }
}

// Shell-style completion: longest common prefix first; when no progress is left
// (e.g. the input ends with "/"), Tab / arrows cycle through the entries.
async function fsComplete(delta = 1) {
  const input = $("fs-input");
  if (fsCycle) {
    fsCycle.idx = (fsCycle.idx + delta + fsCycle.names.length) % fsCycle.names.length;
    input.value = fsCycle.base + fsCycle.names[fsCycle.idx] + "/";
    fsHighlight();
    return;
  }
  const v = input.value;
  const slash = v.lastIndexOf("/");
  const dir = slash <= 0 ? "/" : v.slice(0, slash);
  const partial = v.slice(slash + 1);
  let d;
  try {
    d = await api.fs(dir);
  } catch {
    return;
  }
  renderListing(d);
  const names = d.dirs.map((e) => e.name).filter((n) => n.startsWith(partial));
  if (!names.length) return;
  const base = d.path === "/" ? "/" : d.path + "/";
  if (names.length === 1) {
    input.value = base + names[0] + "/";
    browse(input.value);
    return;
  }
  let lcp = names[0];
  for (const n of names) while (!n.startsWith(lcp)) lcp = lcp.slice(0, -1);
  if (lcp.length > partial.length) {
    input.value = base + lcp;
    return;
  }
  fsCycle = { base, names, idx: delta > 0 ? 0 : names.length - 1 };
  input.value = base + fsCycle.names[fsCycle.idx] + "/";
  fsHighlight();
}

function fsHighlight() {
  const rows = [...$("fs-list").querySelectorAll(".fs-entry")];
  rows.forEach((r) => r.classList.remove("selected"));
  if (!fsCycle) return;
  const want = fsCycle.names[fsCycle.idx] + "/";
  const row = rows.find((r) => r.textContent === want);
  if (row) {
    row.classList.add("selected");
    row.scrollIntoView({ block: "nearest" });
  }
}

async function select(path) {
  $("fs-error").textContent = "probing…";
  let info;
  try {
    info = await api.probe(path);
  } catch (e) {
    $("fs-error").textContent = String(e.message || e);
    return;
  }
  fsSelected = info;
  $("fs-error").textContent = "";
  $("fs-form").hidden = false;
  $("fs-ds-name").textContent = info.path.split("/").pop() +
    (info.sequences.length ? ` — ${info.sequences.length} sequences` : "") +
    (info.is_synchronous ? "" : " (async — opens as an event timeline)");

  const chBox = $("fs-channels");
  chBox.replaceChildren();
  for (const ch of info.channels) {
    const lab = document.createElement("label");
    lab.className = "fs-ch";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = ch;
    cb.checked = ["lidar", "camera", "pose"].includes(ch)
      || /ground_truth|trav|label|pred/.test(ch);
    lab.append(cb, document.createTextNode(" " + ch));
    chBox.appendChild(lab);
  }
}

async function openSelected() {
  if (!fsSelected) return;
  const channels = [...$("fs-channels").querySelectorAll("input:checked")].map((c) => c.value);
  const payload = {
    path: fsSelected.path,
    channels: channels.length ? channels : null,
    every: +$("fs-every").value || 1,
  };
  $("fs-error").textContent = "opening…";
  $("fs-open").disabled = true;
  try {
    const s = await api.open(payload);
    closeModal();
    initSession(s);
  } catch (e) {
    $("fs-error").textContent = String(e.message || e);
  } finally {
    $("fs-open").disabled = false;
  }
}

// ------------------------------------------------------------------ boot
async function boot() {
  let s = await api.session();

  // Restore transforms saved with this dataset (scripts re-imported, then applied)
  // before the first render so their virtual channels exist from the start.
  try {
    const st = (await api.state()).state;
    if (st && st.plugins && st.plugins.scripts && st.plugins.scripts.length) {
      for (const path of st.plugins.scripts) {
        try {
          const d = await api.pluginsLoad(path);
          pluginSpecs = d.specs;
          pluginScripts.add(path);
        } catch (e) {
          console.warn("plugin script:", path, e);
        }
      }
      if (st.plugins.active && st.plugins.active.length) {
        const d = await api.pluginsActive(st.plugins.active);
        pluginActive = d.plugins.active;
        s = d.session;
      }
    }
  } catch { /* no sidecar */ }

  basisGutter(document.querySelector('[data-gutter="rail"]'), document.querySelector(".rail"), "x");

  $("add-view").onclick = () => {
    const v = $("add-kind").value;
    const sep = v.indexOf(":");
    panels.add(v.slice(0, sep), v.slice(sep + 1));
  };
  $("frame").oninput = () => { setPlaying(false); show(+$("frame").value); };
  $("btn-prev").onclick = () => { setPlaying(false); show(index - 1); };
  $("btn-next").onclick = () => { setPlaying(false); show(index + 1); };
  $("btn-play").onclick = () => setPlaying(!playing);
  $("speed").onchange = () => { speed = +$("speed").value; saveStateSoon(); };
  const onAccum = () => {
    accum = { back: Math.max(0, +$("accum-back").value || 0),
              fwd: Math.max(0, +$("accum-fwd").value || 0) };
    show(index);
  };
  $("accum-back").onchange = onAccum;
  $("accum-fwd").onchange = onAccum;
  $("btn-loop-a").onclick = () => setLoopBound("a");
  $("btn-loop-b").onclick = () => setLoopBound("b");
  $("btn-mark").onclick = toggleBookmark;
  $("btn-tracks").onclick = () => {
    $("btn-tracks").classList.toggle("active");
    $("tracks").hidden = !$("btn-tracks").classList.contains("active") || !timeline;
    drawTracks(null);
  };
  $("tracks-canvas").addEventListener("mousedown", tracksSeek);
  $("sync-apply").onclick = applySync;

  $("btn-open").onclick = () => openModal("dataset");
  $("btn-load-script").onclick = () => openModal("script");
  $("btn-apply-plugins").onclick = applyPlugins;
  $("fs-close").onclick = closeModal;
  $("fs-open").onclick = openSelected;
  $("fs-input").addEventListener("input", () => { fsCycle = null; });
  $("fs-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") browse($("fs-input").value);
    else if (e.key === "Tab") { e.preventDefault(); fsComplete(e.shiftKey ? -1 : 1); }
    else if (e.key === "ArrowDown") { e.preventDefault(); fsComplete(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); fsComplete(-1); }
  });

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (!$("fs-modal").hidden) return;
    if (e.key === "ArrowLeft") { setPlaying(false); show(index - 1); }
    else if (e.key === "ArrowRight") { setPlaying(false); show(index + 1); }
    else if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === "m") toggleBookmark();
  });
  window.addEventListener("beforeunload", () => { saveState(); });

  await initSession(s);
}

boot().catch((e) => {
  $("status").textContent = String(e);
  console.error(e);
});
