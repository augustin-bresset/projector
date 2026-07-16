// Projector API client: /api/* calls + numpy array decoding.
//
// Arrays travel as {dtype, shape, data(base64)}: we decode them into a TypedArray
// straight over the buffer (no extra copy, semantics preserved).

const TYPED = {
  float32: Float32Array, float64: Float64Array,
  int8: Int8Array, int16: Int16Array, int32: Int32Array,
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array,
  int64: BigInt64Array, uint64: BigUint64Array,
};

export function decodeArray(o) {
  if (!o) return null;
  const bin = atob(o.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const Ctor = TYPED[o.dtype] || Uint8Array;
  return { data: new Ctor(bytes.buffer), shape: o.shape, dtype: o.dtype };
}

// One frame message → usable object (decoded channels).
export function decodeFrame(d) {
  const channels = {};
  for (const [name, arr] of Object.entries(d.channels || {})) channels[name] = decodeArray(arr);
  return {
    seq: d.seq,
    index: d.index,
    nFrames: d.n_frames,
    timestamps: d.timestamps,   // null = synchronous
    indices: d.indices || null, // per-channel event counters (async)
    channels,                   // {name: {data: TypedArray, shape, dtype}}
  };
}

async function getJson(path) {
  const r = await fetch(path);
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.detail || `${path} → ${r.status}`);
  }
  return r.json();
}

async function postJson(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.detail || `${path} → ${r.status}`);
  }
  return r.json();
}

export const api = {
  session: () => getJson("/api/session"),
  frame: async (seq, index) =>
    decodeFrame(await getJson(`/api/frame?seq=${encodeURIComponent(seq)}&index=${index}`)),
  trajectory: async (seq) => {
    const d = await getJson(`/api/trajectory?seq=${encodeURIComponent(seq)}`);
    return decodeArray(d.points);   // {data, shape:[N,3]} or null
  },
  // User transforms (scripts imported server-side, executed in memory).
  plugins: () => getJson("/api/plugins"),
  pluginsLoad: (path) => postJson("/api/plugins/load", { path }),
  pluginsUnload: (path) => postJson("/api/plugins/unload", { path }),
  pluginsActive: (active) => postJson("/api/plugins/active", { active }),
  // Open-dataset flow.
  fs: (path, files = null) => {
    const q = new URLSearchParams();
    if (path) q.set("path", path);
    if (files) q.set("files", files);
    const qs = q.toString();
    return getJson("/api/fs" + (qs ? `?${qs}` : ""));
  },
  probe: (path) => getJson(`/api/probe?path=${encodeURIComponent(path)}`),
  open: (payload) => postJson("/api/open", payload),
  sync: (payload) => postJson("/api/sync", payload),
  last: () => getJson("/api/last"),
  // Session sidecar (per dataset, stored server-side in the user data dir).
  state: () => getJson("/api/state"),
  saveState: (state) => postJson("/api/state", state),
};
