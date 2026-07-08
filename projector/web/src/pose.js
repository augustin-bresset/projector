// Minimal rigid-pose math over decoded arrays — mirrors core/poses.py.
// Poses arrive as (4,4) row-major, (7,) [x,y,z, qx,qy,qz,qw], or (3,) position.

// Decoded pose array → row-major 4x4 Float64Array, or null.
export function poseToMatrix(arr) {
  if (!arr) return null;
  const d = arr.data;
  if (arr.shape.length === 2 && arr.shape[0] === 4 && arr.shape[1] === 4) {
    return Float64Array.from(d);
  }
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  if (arr.shape.length === 1 && arr.shape[0] === 7) {
    const [x, y, z, qx, qy, qz, qw] = d;
    const n = qx * qx + qy * qy + qz * qz + qw * qw;
    const s = n > 1e-12 ? 2 / n : 0;
    m[0] = 1 - s * (qy * qy + qz * qz); m[1] = s * (qx * qy - qz * qw); m[2] = s * (qx * qz + qy * qw);
    m[4] = s * (qx * qy + qz * qw); m[5] = 1 - s * (qx * qx + qz * qz); m[6] = s * (qy * qz - qx * qw);
    m[8] = s * (qx * qz - qy * qw); m[9] = s * (qy * qz + qx * qw); m[10] = 1 - s * (qx * qx + qy * qy);
    m[3] = x; m[7] = y; m[11] = z;
    return m;
  }
  if (arr.shape.length === 1 && arr.shape[0] === 3) {
    m[3] = d[0]; m[7] = d[1]; m[11] = d[2];
    return m;
  }
  return null;
}

// Inverse of a rigid transform (row-major 4x4): [R t]⁻¹ = [Rᵀ -Rᵀt].
export function invertRigid(m) {
  const o = new Float64Array(16);
  o[15] = 1;
  o[0] = m[0]; o[1] = m[4]; o[2] = m[8];
  o[4] = m[1]; o[5] = m[5]; o[6] = m[9];
  o[8] = m[2]; o[9] = m[6]; o[10] = m[10];
  o[3] = -(o[0] * m[3] + o[1] * m[7] + o[2] * m[11]);
  o[7] = -(o[4] * m[3] + o[5] * m[7] + o[6] * m[11]);
  o[11] = -(o[8] * m[3] + o[9] * m[7] + o[10] * m[11]);
  return o;
}

export function applyRigid(m, x, y, z) {
  return [
    m[0] * x + m[1] * y + m[2] * z + m[3],
    m[4] * x + m[5] * y + m[6] * z + m[7],
    m[8] * x + m[9] * y + m[10] * z + m[11],
  ];
}

export function translationOf(m) {
  return [m[3], m[7], m[11]];
}
