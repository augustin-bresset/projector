// Non-point scene furniture: the ground reference grid and the rerun-style
// orbit-pivot crosshair. Both are owned by the scene the viewer hands them.

import * as THREE from "three";

function smoothstep(edge0, edge1, x) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

// -- ground reference grid ----------------------------------------------------

// Ground-plane reference grid: a single shader-lit quad rather than
// GridHelper's fixed line list, so it can do three things GridHelper can't —
// anti-aliased 1 m minor lines, brighter/thicker 10 m major lines, and a soft
// fade instead of a hard edge.
const GRID_VERT = `
  varying vec3 vWorldPos;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }`;

const GRID_FRAG = `
  varying vec3 vWorldPos;
  uniform vec3 uMinorColor;
  uniform vec3 uMajorColor;
  uniform float uCellSize;     // minor line spacing (1 m)
  uniform float uSectionSize;  // major line spacing (10 m)
  uniform float uOpacity;
  uniform float uFadeDistance; // camera distance the grid is lost by
  uniform float uPlaneHalf;    // plane half-extent, for a matching edge fade

  // Anti-aliased grid-line coverage at the given coord spacing and line
  // width (in line-widths). Fades itself to 0 once a cell covers less than
  // ~a screen pixel — past that point the lines would alias into flat grey,
  // so this spacing simply drops out and only a coarser one (called
  // separately, with a bigger width) keeps reading as lines.
  float gridLine(vec2 coord, float width) {
    vec2 deriv = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / max(deriv * width, 1e-6);
    float line = 1.0 - clamp(min(grid.x, grid.y), 0.0, 1.0);
    float fade = 1.0 - clamp(max(deriv.x, deriv.y) - 1.0, 0.0, 1.0);
    return line * fade;
  }

  void main() {
    float minor = gridLine(vWorldPos.xy / uCellSize, 1.0);
    float major = gridLine(vWorldPos.xy / uSectionSize, 1.8);
    float coverage = max(minor, major);
    if (coverage <= 0.0) discard;
    vec3 color = mix(uMinorColor, uMajorColor, major);

    // Lost-in-the-distance fade: from the camera (so it reads whichever way
    // you look, not just radially from the grid's centre) and a matching
    // fade toward the plane's own edge so that boundary never shows as a hard
    // cutoff either.
    float camFade = 1.0 - smoothstep(uFadeDistance * 0.4, uFadeDistance, distance(vWorldPos, cameraPosition));
    float edgeFade = 1.0 - smoothstep(uPlaneHalf * 0.7, uPlaneHalf, length(vWorldPos.xy));

    float alpha = coverage * uOpacity * mix(1.0, 1.5, major) * camFade * edgeFade;
    if (alpha <= 0.002) discard;
    gl_FragColor = vec4(color, alpha);
  }`;

// A faint 1 m / 10 m reference grid on the world XY plane (Z = 0), scaled to
// the cloud so it reads as ground scale rather than an arbitrary fixed
// extent. It fades out — toward the camera and toward its own edge —
// instead of stopping abruptly, and its 1 m lines self-fade once the camera
// pulls back far enough that they'd alias, leaving only the thicker 10 m
// lines standing.
export class ReferenceGrid {
  constructor(scene) {
    this.scene = scene;
    this.mesh = null;
  }

  // (Re)build for a cloud of the given bounding radius.
  rebuild(radius) {
    this.dispose();
    const fadeDistance = Math.max(radius * 4, 20);
    const half = fadeDistance * 1.15;
    const geom = new THREE.PlaneGeometry(half * 2, half * 2);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uMinorColor: { value: new THREE.Color(0x333c4d) },
        uMajorColor: { value: new THREE.Color(0x5a6478) },
        uCellSize: { value: 1.0 },
        uSectionSize: { value: 10.0 },
        uOpacity: { value: 0.35 },
        uFadeDistance: { value: fadeDistance },
        uPlaneHalf: { value: half },
      },
      vertexShader: GRID_VERT,
      fragmentShader: GRID_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      extensions: { derivatives: true },
    });
    // PlaneGeometry already lies flat in the local XY plane (normal +Z) —
    // no rotation needed for our Z-up world, unlike GridHelper's default XZ.
    this.mesh = new THREE.Mesh(geom, material);
    this.scene.add(this.mesh);
  }

  dispose() {
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh = null;
  }
}

// -- orbit-pivot crosshair ----------------------------------------------------

// Stays fully visible for this long after the last camera move, then fades
// out — makes the TrackballControls pivot legible instead of an invisible
// point somewhere in the cloud.
const LINGER_MS = 350;
const FADE_MS = 100;
// Crosshair half-length as a fraction of the current camera-to-pivot distance,
// so it reads the same size on screen whether you're framing the whole cloud
// or zoomed into one corner.
const SIZE = 0.03;

// The rerun-style orbit-pivot crosshair: three short segments centred on the
// orbit target, fixed to world axes (not the camera's, which can roll freely
// under TrackballControls) so it also doubles as a "how tilted am I"
// reference. Depth-tested on purpose (like rerun): the cloud occludes it, so
// how much of it shows tells you how deep the pivot sits in the scene.
export class OrbitIndicator {
  constructor(scene) {
    this.scene = scene;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6 * 3), 3));
    geom.setDrawRange(0, 6);
    const material = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    this.lines = new THREE.LineSegments(geom, material);
    this.lines.visible = false;
    this.lines.frustumCulled = false; // its extent doesn't track the geometry's bounding sphere
    scene.add(this.lines);
    this._until = 0; // linger deadline; fade-out runs for FADE_MS past it
    this._fadeIn = false;
    this._fadeChangeTime = 0;
  }

  // Call on every camera move: restarts the linger window.
  poke(now) {
    this._until = now + LINGER_MS;
  }

  // Update position and fade for the current frame. Returns {active, dirty}:
  // `active` while the render loop must be kept alive (lingering or mid-fade),
  // `dirty` when this frame's visuals changed and need a redraw.
  tick(now, camera, target) {
    const active = now < this._until + FADE_MS;
    if (!active) {
      if (this.lines.visible) {
        this.lines.visible = false;
        return { active: false, dirty: true };
      }
      return { active: false, dirty: false };
    }
    const showing = now < this._until;
    if (showing !== this._fadeIn) {
      this._fadeChangeTime = now;
      this._fadeIn = showing;
    }
    const elapsed = now - this._fadeChangeTime;
    const fade = this._fadeIn ? smoothstep(0, FADE_MS, elapsed) : smoothstep(FADE_MS, 0, elapsed);

    this.lines.visible = fade > 0.001;
    if (this.lines.visible) {
      const half = camera.position.distanceTo(target) * SIZE * fade;
      const pos = this.lines.geometry.attributes.position.array;
      // Up: half-length, drawn upward only (mirrors rerun — reads as "ground"
      // without a stray line poking below the pivot). Right/forward: full
      // length, both ways.
      pos[0] = target.x; pos[1] = target.y; pos[2] = target.z;
      pos[3] = target.x; pos[4] = target.y; pos[5] = target.z + half * 0.5;
      pos[6] = target.x - half; pos[7] = target.y; pos[8] = target.z;
      pos[9] = target.x + half; pos[10] = target.y; pos[11] = target.z;
      pos[12] = target.x; pos[13] = target.y - half; pos[14] = target.z;
      pos[15] = target.x; pos[16] = target.y + half; pos[17] = target.z;
      this.lines.geometry.attributes.position.needsUpdate = true;
    }
    return { active: true, dirty: true };
  }

  dispose() {
    this.scene.remove(this.lines);
    this.lines.geometry.dispose();
    this.lines.material.dispose();
  }
}
