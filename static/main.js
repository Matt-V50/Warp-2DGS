// ---------------------------------------------------------------------------
// Interactive kernel demo: D(t) and G(r * D(x / r)) for one axis.
// PWL: knots (-1,-1), (c1,d1), (1,1); identity for |t| > 1 (Eq. 6 of the paper).
// PWL-sr: c1 and q, R = exp(q), d1 = -1 + 2 R h0 / (h1 + R h0) (Eqs. 12-14).
// ---------------------------------------------------------------------------
(function demo() {
  const R_SCALE = 3, KNOT_MAX = 0.95, Q_MAX = 3;
  const state = { mode: "pwl", c1: 0.35, d1: -0.35, q: 1.2 };
  const el = id => document.getElementById(id);
  const warpCanvas = el("warp-canvas"), kernelCanvas = el("kernel-canvas");
  const sliders = { c1: el("c1"), d1: el("d1"), q: el("q") };
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function srKnot(c1, q) {
    const R = Math.exp(q), h0 = c1 + 1, h1 = 1 - c1;
    return -1 + (2 * R * h0) / (h1 + R * h0);
  }
  function srQ(c1, d1) {   // inverse of srKnot for dragging in sr mode
    const h0 = c1 + 1, h1 = 1 - c1;
    return clamp(Math.log(((d1 + 1) * h1) / ((1 - d1) * h0)), -Q_MAX, Q_MAX);
  }
  const knot = () => ({ c: state.c1, d: state.mode === "sr" ? srKnot(state.c1, state.q) : state.d1 });

  function D(t) {
    if (t <= -1 || t >= 1) return t;
    const { c, d } = knot();
    return t < c ? -1 + ((d + 1) * (t + 1)) / (c + 1) : d + ((1 - d) * (t - c)) / (1 - c);
  }

  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1, size = canvas.clientWidth || 360;
    canvas.width = size * dpr; canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, size };
  }

  // maps data coordinates to the canvas of a square plot with padding
  function frame(size, xr, yr, pad = 22) {
    const w = size - 2 * pad;
    return {
      x: v => pad + ((v - xr[0]) / (xr[1] - xr[0])) * w,
      y: v => size - pad - ((v - yr[0]) / (yr[1] - yr[0])) * w,
      ix: px => xr[0] + ((px - pad) / w) * (xr[1] - xr[0]),
      iy: py => yr[0] + ((size - pad - py) / w) * (yr[1] - yr[0]),
    };
  }

  function curve(ctx, f, x0, x1, map, n = 400) {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const x = x0 + ((x1 - x0) * i) / n, px = map.x(x), py = map.y(f(x));
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
  }

  const WARP_RANGE = [-1.3, 1.3];
  let warpMap = null;

  function drawWarp() {
    const { ctx, size } = setup(warpCanvas);
    const m = (warpMap = frame(size, WARP_RANGE, WARP_RANGE));
    ctx.clearRect(0, 0, size, size);
    // learnable square and axes
    ctx.fillStyle = "rgba(76, 175, 80, 0.10)";
    ctx.fillRect(m.x(-1), m.y(1), m.x(1) - m.x(-1), m.y(-1) - m.y(1));
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(m.x(0), m.y(-1.3)); ctx.lineTo(m.x(0), m.y(1.3));
    ctx.moveTo(m.x(-1.3), m.y(0)); ctx.lineTo(m.x(1.3), m.y(0)); ctx.stroke();
    // identity
    ctx.setLineDash([6, 5]); ctx.strokeStyle = "#8fc79a"; ctx.lineWidth = 1.5;
    curve(ctx, t => t, -1.3, 1.3, m, 2);
    ctx.setLineDash([]);
    // D(t)
    ctx.strokeStyle = css("--accent-2"); ctx.lineWidth = 2.5;
    curve(ctx, D, -1.3, 1.3, m);
    // fixed endpoints and the learnable knot
    ctx.fillStyle = css("--text");
    for (const p of [-1, 1]) { ctx.beginPath(); ctx.arc(m.x(p), m.y(p), 4, 0, 2 * Math.PI); ctx.fill(); }
    const { c, d } = knot();
    ctx.fillStyle = css("--accent");
    ctx.beginPath(); ctx.arc(m.x(c), m.y(d), 8, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
  }

  function drawKernel() {
    const { ctx, size } = setup(kernelCanvas);
    const m = frame(size, [-4.5, 4.5], [-0.05, 1.08]);
    ctx.clearRect(0, 0, size, size);
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(m.x(-4.5), m.y(0)); ctx.lineTo(m.x(4.5), m.y(0)); ctx.stroke();
    for (const s of [-3, 3]) {   // warped domain [-r, r]
      ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(m.x(s), m.y(-0.05)); ctx.lineTo(m.x(s), m.y(1.08)); ctx.stroke();
    }
    ctx.setLineDash([4, 4]); ctx.strokeStyle = "#9aa1ad"; ctx.lineWidth = 1.5;
    curve(ctx, x => Math.exp(-0.5 * x * x), -4.5, 4.5, m);
    ctx.setLineDash([]);
    ctx.strokeStyle = "#ef7d2d"; ctx.lineWidth = 2.5;
    curve(ctx, x => { const u = R_SCALE * D(x / R_SCALE); return Math.exp(-0.5 * u * u); }, -4.5, 4.5, m, 600);
  }

  function sync() {
    const { d } = knot();
    sliders.c1.value = state.c1; sliders.d1.value = d; sliders.q.value = state.q;
    el("c1-out").textContent = state.c1.toFixed(2);
    el("d1-out").textContent = d.toFixed(2);
    el("q-out").textContent = state.q.toFixed(2);
    el("d1-row").querySelector("input").disabled = state.mode === "sr";
    el("q-row").hidden = state.mode !== "sr";
    drawWarp(); drawKernel();
  }

  sliders.c1.addEventListener("input", e => { state.c1 = +e.target.value; sync(); });
  sliders.d1.addEventListener("input", e => { state.d1 = +e.target.value; sync(); });
  sliders.q.addEventListener("input", e => { state.q = +e.target.value; sync(); });
  el("reset").addEventListener("click", () => { state.c1 = 0; state.d1 = 0; state.q = 0; sync(); });

  document.querySelectorAll(".seg button").forEach(btn => btn.addEventListener("click", () => {
    const { d } = knot();
    state.mode = btn.dataset.mode;
    if (state.mode === "sr") state.q = srQ(state.c1, d); else state.d1 = d;   // keep the knot in place
    document.querySelectorAll(".seg button").forEach(b => {
      b.classList.toggle("active", b === btn); b.setAttribute("aria-checked", b === btn);
    });
    sync();
  }));

  // drag the knot directly on the D(t) plot
  let dragging = false;
  function moveKnot(e) {
    const rect = warpCanvas.getBoundingClientRect();
    const c = clamp(warpMap.ix(e.clientX - rect.left), -KNOT_MAX, KNOT_MAX);
    const d = clamp(warpMap.iy(e.clientY - rect.top), -KNOT_MAX, KNOT_MAX);
    state.c1 = c;
    if (state.mode === "sr") state.q = srQ(c, d); else state.d1 = d;
    sync();
  }
  warpCanvas.addEventListener("pointerdown", e => {
    dragging = true; warpCanvas.classList.add("dragging"); warpCanvas.setPointerCapture(e.pointerId); moveKnot(e);
  });
  warpCanvas.addEventListener("pointermove", e => { if (dragging) moveKnot(e); });
  const stop = () => { dragging = false; warpCanvas.classList.remove("dragging"); };
  warpCanvas.addEventListener("pointerup", stop);
  warpCanvas.addEventListener("pointercancel", stop);
  window.addEventListener("resize", sync);
  sync();
})();

// ---------------------------------------------------------------------------
// Before / after comparison slider.
// ---------------------------------------------------------------------------
(function compare() {
  const SCENES = [
    { name: "Counter", id: "counter_DSCF6089" },
    { name: "Room", id: "room_DSCF4971" },
    { name: "Bonsai", id: "bonsai_DSCF5597" },
    { name: "Bonsai", id: "bonsai_DSCF5637" },
    { name: "Stump", id: "stump_DSC9237" },
  ];
  const dir = "static/images/compare/";
  const left = document.getElementById("cmp-left"), right = document.getElementById("cmp-right");
  const clip = document.getElementById("cmp-left-clip"), handle = document.getElementById("cmp-handle");
  const range = document.getElementById("cmp-range"), gt = document.getElementById("gt-toggle");
  const tagLeft = document.getElementById("tag-left"), thumbs = document.getElementById("thumbs");
  let current = SCENES[0];

  function show(scene) {
    current = scene;
    right.src = `${dir}${scene.id}_ours.jpg`;
    left.src = `${dir}${scene.id}_${gt.checked ? "gt" : "2dgs"}.jpg`;
    tagLeft.textContent = gt.checked ? "Ground truth" : "2DGS";
    thumbs.querySelectorAll("button").forEach(b => {
      const on = b.dataset.id === scene.id;
      b.classList.toggle("active", on); b.setAttribute("aria-selected", on);
    });
  }
  function place(pct) {
    clip.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handle.style.left = `${pct}%`;
  }

  for (const scene of SCENES) {
    const b = document.createElement("button");
    b.type = "button"; b.dataset.id = scene.id; b.title = scene.name;
    b.setAttribute("role", "tab");
    b.innerHTML = `<img src="${dir}${scene.id}_ours.jpg" alt="${scene.name}" loading="lazy">`;
    b.addEventListener("click", () => show(scene));
    thumbs.appendChild(b);
  }
  range.addEventListener("input", () => place(+range.value));
  gt.addEventListener("change", () => show(current));
  place(50);
  show(current);
})();

// ---------------------------------------------------------------------------
// Copy BibTeX.
// ---------------------------------------------------------------------------
document.getElementById("copy-bib").addEventListener("click", async e => {
  const text = document.getElementById("bib").textContent;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  e.target.textContent = "Copied";
  setTimeout(() => (e.target.textContent = "Copy"), 1500);
});
