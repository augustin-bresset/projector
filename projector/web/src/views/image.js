// Draw an image channel (camera) onto a canvas. Raw images
// (H,W,3) / (H,W,4) / (H,W), uint8 (or normalized if another dtype).

function normalize(arr) {
  if (arr.dtype === "uint8") return arr.data;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < arr.data.length; i++) { const v = Number(arr.data[i]); if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi <= lo) hi = lo + 1;
  const out = new Uint8Array(arr.data.length);
  for (let i = 0; i < arr.data.length; i++) out[i] = ((Number(arr.data[i]) - lo) / (hi - lo)) * 255;
  return out;
}

export function drawImage(canvas, arr) {
  const [h, w] = arr.shape;
  const ch = arr.shape.length > 2 ? arr.shape[2] : 1;
  const src = normalize(arr);
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const px = img.data;
  for (let i = 0; i < w * h; i++) {
    const o = i * 4, s = i * ch;
    if (ch === 1) { px[o] = px[o + 1] = px[o + 2] = src[s]; px[o + 3] = 255; }
    else { px[o] = src[s]; px[o + 1] = src[s + 1]; px[o + 2] = src[s + 2]; px[o + 3] = ch > 3 ? src[s + 3] : 255; }
  }
  ctx.putImageData(img, 0, 0);
}
