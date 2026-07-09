// Binary frame stream over websocket, with an LRU cache and pipelined prefetch.
//
// One message = <u32 header length> + JSON header + concatenated raw buffers, header
// and buffers 8-byte aligned (see server/protocol.py) — TypedArrays are mapped
// straight over the message buffer, zero copy.

const TYPED = {
  float32: Float32Array, float64: Float64Array,
  int8: Int8Array, int16: Int16Array, int32: Int32Array,
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array,
  int64: BigInt64Array, uint64: BigUint64Array,
};

const CACHE_MAX = 40;   // frames kept client-side (scrub-back is instant)

function decodePacked(buf) {
  const view = new DataView(buf);
  const hlen = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  const base = 4 + hlen;
  const channels = {};
  for (const ch of header.channels) {
    const Ctor = TYPED[ch.dtype] || Uint8Array;
    const count = ch.shape.reduce((a, b) => a * b, 1);
    channels[ch.name] = {
      data: new Ctor(buf, base + ch.offset, count),
      shape: ch.shape,
      dtype: ch.dtype,
    };
  }
  return {
    seq: header.seq,
    index: header.index,
    nFrames: header.n_frames,
    back: header.back || 0,
    fwd: header.fwd || 0,
    timestamps: header.timestamps,
    indices: header.indices || null,   // per-channel event counters (async)
    channels,
  };
}

export class FrameStream {
  constructor(onStatus = null) {
    this.onStatus = onStatus;    // optional: connection state changes ("open"/"closed")
    this.ws = null;
    this.cache = new Map();      // "seq:index" → frame (insertion order = LRU)
    this.waiting = new Map();    // "seq:index" → [resolve, ...]
    this.inflight = new Set();   // keys requested but not yet answered
    this._connecting = null;
    this.stats = { sent: 0, recv: 0 };   // debug / tests
  }

  _connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return Promise.resolve();
    if (this._connecting) return this._connecting;
    this._connecting = new Promise((resolve, reject) => {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${location.host}/ws/frames`);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => { this.ws = ws; this._connecting = null; this.onStatus?.("open"); resolve(); };
      ws.onerror = () => { this._connecting = null; reject(new Error("websocket failed")); };
      ws.onclose = () => { this.ws = null; this.onStatus?.("closed"); };
      ws.onmessage = (e) => this._onFrame(decodePacked(e.data));
    });
    return this._connecting;
  }

  _onFrame(frame) {
    this.stats.recv++;
    const key = `${frame.seq}:${frame.index}:${frame.back}:${frame.fwd}`;
    this.inflight.delete(key);
    this.cache.delete(key);              // re-insert = most recently used
    this.cache.set(key, frame);
    while (this.cache.size > CACHE_MAX) {
      this.cache.delete(this.cache.keys().next().value);
    }
    const waiters = this.waiting.get(key);
    if (waiters) { this.waiting.delete(key); for (const w of waiters) w(frame); }
  }

  async _send(seq, index, back, fwd) {
    const key = `${seq}:${index}:${back}:${fwd}`;
    if (this.cache.has(key) || this.inflight.has(key)) return;
    await this._connect();
    this.inflight.add(key);
    this.stats.sent++;
    this.ws.send(JSON.stringify({ seq, index, back, fwd }));
  }

  // Fetch one frame (cache hit resolves immediately).
  async request(seq, index, back = 0, fwd = 0) {
    const key = `${seq}:${index}:${back}:${fwd}`;
    const hit = this.cache.get(key);
    if (hit) { this.cache.delete(key); this.cache.set(key, hit); return hit; }
    const p = new Promise((resolve) => {
      const w = this.waiting.get(key);
      if (w) w.push(resolve); else this.waiting.set(key, [resolve]);
    });
    await this._send(seq, index, back, fwd);
    return p;
  }

  // Fire-and-forget: pipeline the next `count` frames after `index`, plus
  // `behind` frames before it (scrubbing goes both ways).
  prefetch(seq, index, count, nFrames, back = 0, fwd = 0, behind = 0) {
    for (let i = index + 1; i <= Math.min(index + count, nFrames - 1); i++) {
      this._send(seq, i, back, fwd).catch(() => {});
    }
    for (let i = index - 1; i >= Math.max(index - behind, 0); i--) {
      this._send(seq, i, back, fwd).catch(() => {});
    }
  }

  clear() { this.cache.clear(); }
}
