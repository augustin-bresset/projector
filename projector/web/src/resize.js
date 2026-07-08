// Resize handles (gutters), no dependency.
// - growGutter: redistributes flex-grow between two neighbors of a flex container.
// - basisGutter: adjusts the fixed width/height (flex-basis) of an element.

function startDrag(axis, onMove) {
  const cursor = axis === "x" ? "col-resize" : "row-resize";
  document.body.style.cursor = cursor;
  document.body.style.userSelect = "none";
  const move = (ev) => onMove(axis === "x" ? ev.clientX : ev.clientY);
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
}

export function growGutter(gutter, a, b, axis, min = 120) {
  const dim = axis === "x" ? "width" : "height";
  gutter.addEventListener("mousedown", (e) => {
    e.preventDefault();
    const ra = a.getBoundingClientRect()[dim], rb = b.getBoundingClientRect()[dim];
    const total = ra + rb;
    const gTot = (parseFloat(getComputedStyle(a).flexGrow) || 1) + (parseFloat(getComputedStyle(b).flexGrow) || 1);
    const start = axis === "x" ? e.clientX : e.clientY;
    startDrag(axis, (cur) => {
      const na = Math.max(min, Math.min(total - min, ra + (cur - start)));
      const fa = na / total;
      a.style.flexGrow = (gTot * fa).toFixed(4);
      b.style.flexGrow = (gTot * (1 - fa)).toFixed(4);
    });
  });
}

export function basisGutter(gutter, el, axis, min = 180) {
  const dim = axis === "x" ? "width" : "height";
  gutter.addEventListener("mousedown", (e) => {
    e.preventDefault();
    const base = el.getBoundingClientRect()[dim];
    const start = axis === "x" ? e.clientX : e.clientY;
    startDrag(axis, (cur) => {
      el.style.flex = `0 0 ${Math.max(min, base + (cur - start))}px`;
    });
  });
}
