/* chipscan viewer front end: OpenSeadragon canvas + virtualised candidate list + inspector + keyboard review. */
(() => {
  const COLORS = { anomaly: "#ff7a1a", speck: "#8fd3ff", die2die: "#ff4fb8", short: "#ff5c5c", open: "#e15cff", particle: "#ffab4a", protrusion: "#ffe066", notch: "#5ce1ff", void: "#5cb8ff", missing_feature: "#b77cff", extra_feature: "#b77cff", unclassified: "#4fcb7b" };
  const SEV = { anomaly: 3, speck: 2, die2die: 4, short: 5, open: 5, missing_feature: 4, extra_feature: 4, particle: 3, protrusion: 2, notch: 2, void: 1, unclassified: 0 };
  const CLASS_KEYS = { "1": "short", "2": "open", "3": "particle", "4": "protrusion", "5": "notch", "6": "void" };
  const LABEL = { die2die: "die-to-die", false_alarm: "false alarm", missing_feature: "missing feature", extra_feature: "extra feature" };
  const ROW_H = 64;
  const MIN_LABELS = 5;

  const $ = (id) => document.getElementById(id);
  const label = (s) => LABEL[s] || String(s).replace(/_/g, " ");
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, n = 2) => (v == null || Number.isNaN(v) ? "–" : (+v).toFixed(n));
  const store = { get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };

  let runs = [], run = null, defects = [], view = [], active = -1, selected = new Set();
  let viewer = null, layers = { design: null, pred: null };
  let measuring = false, measurePts = [];
  let rowsFrame = 0;

  // ---------------------------------------------------------------- data
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      const t = await r.text();
      let msg = t; try { msg = JSON.parse(t).detail || t; } catch (e) {}
      throw new Error(msg);
    }
    return r.json();
  }

  async function notify(message, variant = "primary", icon = "info-circle") {
    const el = Object.assign(document.createElement("sl-alert"), { variant, closable: true, duration: 6000 });
    el.innerHTML = `<sl-icon slot="icon" name="${icon}"></sl-icon>${esc(message)}`;
    document.body.append(el);
    await customElements.whenDefined("sl-alert");
    el.toast();
  }
  const fail = (e) => notify(e.message || String(e), "danger", "exclamation-octagon");

  async function loadRuns() {
    runs = await api("/api/runs");
    const sel = $("run");
    sel.innerHTML = runs.map((r, i) => `<sl-option value="r${i}">${esc(r.name)}<span slot="suffix" class="mono muted">${r.n_defects.toLocaleString()}</span></sl-option>`).join("");
    sel.addEventListener("sl-change", () => openRun(runs[+sel.value.slice(1)].name));
    const wanted = new URLSearchParams(location.search).get("run");
    const name = wanted && runs.some(r => r.name === wanted) ? wanted : runs[0].name;
    await sel.updateComplete;
    sel.value = `r${runs.findIndex(r => r.name === name)}`;
    await openRun(name);
  }

  async function openRun(name) {
    run = runs.find(r => r.name === name);
    const wantedId = new URLSearchParams(location.search).get("d");
    history.replaceState(null, "", `?run=${encodeURIComponent(name)}`);
    const reg = run.registration || {};
    const meta = [
      `<b>${run.width.toLocaleString()} × ${run.height.toLocaleString()}</b> px`,
      `<b>${fmt(run.pixel_size_um, 3)}</b> µm/px`,
      esc(run.segmenter || ""),
    ];
    if (!run.anomaly_run && reg.rotation_deg != null) {
      meta.push(`rot <b>${fmt(reg.rotation_deg)}°</b> · scale <b>${fmt(reg.scale, 4)}</b>${reg.flipped ? " · mirrored" : ""}`);
      meta.push(`fine rms <b>${run.fine_rms_px != null ? fmt(run.fine_rms_px) + " px" : "n/a"}</b>`);
    }
    $("runinfo").innerHTML = meta.filter(Boolean).map(m => `<span class="meta">${m}</span>`).join("");
    $("btn-export").href = `/api/runs/${encodeURIComponent(name)}/export.csv`;
    defects = await api(`/api/runs/${encodeURIComponent(name)}/defects`);
    selected.clear(); active = -1; updateSelCount();
    $("sens").classList.toggle("hidden", !run.anomaly_run);
    $("learn-result").classList.add("hidden");
    if (run.anomaly_run && run.extract_params) {
      const p = run.extract_params;
      $("thr").value = +p.threshold; $("minpx").value = String(p.min_area_px);
      $("specks").checked = p.include_specks !== false; $("die2die").checked = p.include_die2die !== false;
    }
    $("layer-design").classList.toggle("hidden", !!run.anomaly_run);
    buildViewer();
    refreshList();
    refreshStats();
    renderInspector();
    if (wantedId != null) {
      const i = view.findIndex(d => String(d.id) === wantedId);
      if (i >= 0) viewer.addOnceHandler("open", () => setTimeout(() => setActive(i, true), 0));
    }
  }

  // ---------------------------------------------------------------- viewer
  function tileSource(kind, ext) {
    const n = run.levels;
    return {
      width: run.width, height: run.height, tileSize: 256, tileOverlap: 0, minLevel: 0, maxLevel: n - 1,
      getTileUrl: (level, x, y) => `/api/runs/${encodeURIComponent(run.name)}/${kind}/${n - 1 - level}/${x}/${y}.${ext}`,
    };
  }

  function buildViewer() {
    if (viewer) { viewer.destroy(); viewer = null; }
    // a fresh navigator element each time: OSD styles and owns it
    const host = $("navigator-host"); host.innerHTML = '<div id="navigator"></div>';
    viewer = OpenSeadragon({
      id: "viewer", tileSources: tileSource("tiles", "jpg"),
      showNavigationControl: false,
      showNavigator: true, navigatorId: "navigator", navigatorAutoFade: false, navigatorAutoResize: true,
      navigatorBackground: "#000", navigatorBorderColor: "transparent", navigatorDisplayRegionColor: "#4fcb7b", navigatorOpacity: 1,
      maxZoomPixelRatio: 8, minZoomImageRatio: 0.5, animationTime: 0.5, gestureSettingsMouse: { clickToZoom: false },
      crossOriginPolicy: false, imageSmoothingEnabled: false, visibilityRatio: 0.2,
    });
    viewer.addHandler("open", () => {
      viewer.addTiledImage({ tileSource: tileSource("design", "png"), opacity: $("layer-design").checked && !run.anomaly_run ? 1 : 0, success: e => { layers.design = e.item; } });
      viewer.addTiledImage({ tileSource: tileSource("pred", "png"), opacity: $("layer-pred").checked ? 1 : 0, success: e => { layers.pred = e.item; } });
      updateZoom();
    });
    viewer.addHandler("update-viewport", () => { safeDraw(); updateZoom(); });
    viewer.addHandler("resize", safeDraw);
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => { try { if (viewer.forceResize) viewer.forceResize(); viewer.forceRedraw(); } catch (e) {} safeDraw(); });
      ro.observe($("viewer-wrap"));
      viewer.addOnceHandler("close", () => ro.disconnect());
    }
    viewer.addHandler("canvas-click", onCanvasClick);
    new OpenSeadragon.MouseTracker({ element: viewer.container, moveHandler: onMove }).setTracking(true);
  }

  // use the base image's own converters: with several layered images the viewport-level ones warn
  function baseImage() { return viewer && viewer.world.getItemCount() ? viewer.world.getItemAt(0) : null; }
  function toImage(pt) { const b = baseImage(); return b ? b.viewerElementToImageCoordinates(pt) : viewer.viewport.viewerElementToImageCoordinates(pt); }
  function toScreen(x, y) { const b = baseImage(); const p = new OpenSeadragon.Point(x, y); return b ? b.imageToViewerElementCoordinates(p) : viewer.viewport.imageToViewerElementCoordinates(p); }
  function imageRect(x, y, w, h) { const b = baseImage(); return b ? b.imageToViewportRectangle(x, y, w, h) : viewer.viewport.imageToViewportRectangle(x, y, w, h); }
  function imageZoom() { // screen px per image px
    const b = baseImage(); if (!b) return 0;
    return b.viewportToImageZoom(viewer.viewport.getZoom(true));
  }

  function updateZoom() { const z = imageZoom(); $("zoom-val").textContent = z ? (z >= 1 ? `${(z * 100).toFixed(0)}%` : `${(z * 100).toFixed(z < 0.1 ? 1 : 0)}%`) : "–"; }

  function safeDraw() { try { drawOverlay(); } catch (e) { console.warn("overlay draw failed", e); } }

  function drawOverlay() {
    const cv = $("overlay"), wrap = $("viewer-wrap"), dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!viewer || !viewer.viewport || !run || !baseImage()) return;
    const zoom = imageZoom();
    const act = view[active];
    if ($("layer-defects").checked) {
      for (const d of defects) {
        const b = d.bbox_px; if (!b || d === act) continue;
        const p0 = toScreen(b.x0, b.y0), p1 = toScreen(b.x1, b.y1);
        if (p1.x < -20 || p1.y < -20 || p0.x > W + 20 || p0.y > H + 20) continue;
        strokeDefect(ctx, d, p0, p1, zoom, false);
      }
    }
    if (act && act.bbox_px) {
      const b = act.bbox_px, p0 = toScreen(b.x0, b.y0), p1 = toScreen(b.x1, b.y1);
      strokeDefect(ctx, act, p0, p1, zoom, true);
      drawLabel(ctx, `#${act.id} · ${label(act.class)}`, (p0.x + p1.x) / 2, Math.min(p0.y, p1.y) - 10, COLORS[act.class] || COLORS.unclassified, W);
    }
    if (measurePts.length) {
      ctx.strokeStyle = "#4fcb7b"; ctx.fillStyle = "#4fcb7b"; ctx.lineWidth = 2; ctx.beginPath();
      measurePts.forEach((p, k) => { const s = toScreen(p.x, p.y); k ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
      ctx.stroke();
      measurePts.forEach(p => { const s = toScreen(p.x, p.y); ctx.beginPath(); ctx.arc(s.x, s.y, 3.5, 0, 6.283); ctx.fill(); });
    }
  }

  function strokeDefect(ctx, d, p0, p1, zoom, isActive) {
    const col = COLORS[d.class] || COLORS.unclassified;
    ctx.strokeStyle = col; ctx.lineWidth = isActive ? 2.5 : 1.25;
    ctx.globalAlpha = d.review.status === "false_alarm" ? 0.45 : 1;
    ctx.setLineDash(d.review.status === "false_alarm" ? [4, 3] : []);
    if (zoom > 1.5 && d.polygons_px && d.polygons_px.length) {
      for (const poly of d.polygons_px) {
        ctx.beginPath();
        poly.forEach((p, k) => { const s = toScreen(p[0], p[1]); k ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
        ctx.closePath(); ctx.stroke();
      }
    } else if (!isActive && Math.max(p1.x - p0.x, p1.y - p0.y) < 4) {
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, 2, 0, 6.283); ctx.fill();
    } else {
      const w = Math.max(p1.x - p0.x, 8), h = Math.max(p1.y - p0.y, 8), cx = (p0.x + p1.x) / 2, cy = (p0.y + p1.y) / 2, pad = isActive ? 6 : 3;
      ctx.strokeRect(cx - w / 2 - pad, cy - h / 2 - pad, w + 2 * pad, h + 2 * pad);
    }
    if (selected.has(d.id)) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(Math.max(p0.x, p1.x) + 7, Math.min(p0.y, p1.y) - 7, 3.5, 0, 6.283); ctx.fill(); }
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  }

  function roundRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

  function drawLabel(ctx, text, cx, bottom, col, W) {
    ctx.font = "600 11.5px Inter, sans-serif";
    const tw = ctx.measureText(text).width, w = tw + 18, h = 22;
    let x = Math.min(Math.max(cx - w / 2, 6), W - w - 6), y = bottom - h;
    if (y < 6) y = 6;
    ctx.fillStyle = "rgba(11,13,16,.88)"; roundRect(ctx, x, y, w, h, 6); ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x + 9, y + h / 2, 3, 0, 6.283); ctx.fill();
    ctx.fillStyle = "#e6e9ee"; ctx.textBaseline = "middle"; ctx.fillText(text, x + 15, y + h / 2 + 0.5);
  }

  function onMove(e) {
    if (!viewer || !run || !baseImage()) return;
    const p = toImage(e.position);
    $("readout").textContent = `x ${p.x.toFixed(0).padStart(6)}  y ${p.y.toFixed(0).padStart(6)} px    ${(p.x * run.pixel_size_um).toFixed(1)}, ${(p.y * run.pixel_size_um).toFixed(1)} µm`;
  }

  async function onCanvasClick(e) {
    if (!e.quick) return;
    const p = toImage(e.position);
    if (measuring) {
      measurePts.push(p); drawOverlay();
      if (measurePts.length === 2) {
        const [a, b] = measurePts;
        try {
          const r = await api(`/api/runs/${encodeURIComponent(run.name)}/measure?x0=${a.x}&y0=${a.y}&x1=${b.x}&y1=${b.y}`);
          $("measure-out").innerHTML = `<div class="total">Cut length ${fmt(r.length_um, 1)} µm</div>` + r.runs.map(x => `<span>${esc(x.phase)}</span><span class="v">${fmt(x.length_um)} µm</span>`).join("");
          $("card-measure").classList.remove("hidden");
        } catch (err) { fail(err); }
        setMeasuring(false, true);
      }
      return;
    }
    // click on an outline selects it
    const z = imageZoom() || 1;
    let best = -1, bestD = 1e9;
    view.forEach((d, i) => {
      const b = d.bbox_px; if (!b) return;
      const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2, dd = Math.hypot(cx - p.x, cy - p.y);
      const r = Math.max(b.x1 - b.x0, b.y1 - b.y0) / 2 + 12 / z;
      if (dd < r && dd < bestD) { best = i; bestD = dd; }
    });
    if (best >= 0) setActive(best, false);
  }

  function setMeasuring(on, keepPts) {
    measuring = on; if (!keepPts) measurePts = [];
    $("btn-measure").classList.toggle("active", on);
    $("viewer-wrap").classList.toggle("measuring", on);
    $("mode").classList.toggle("hidden", !on);
    safeDraw();
  }

  // ---------------------------------------------------------------- list (virtualised)
  function parseFilter(q) {
    const toks = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return d => toks.every(t => {
      if (t.startsWith("p>")) return d.p_real != null && d.p_real > parseFloat(t.slice(2));
      if (t.startsWith("p<")) return d.p_real != null && d.p_real < parseFloat(t.slice(2));
      if (t.startsWith(">")) return d.area_um2 > parseFloat(t.slice(1));
      if (t.startsWith("<")) return d.area_um2 < parseFloat(t.slice(1));
      if (t.startsWith("#")) return String(d.id) === t.slice(1);
      return d.class.includes(t) || d.kind.includes(t) || d.review.status.includes(t) || (t === "selected" && selected.has(d.id));
    });
  }

  function refreshList() {
    const f = parseFilter($("q").value || ""), sort = $("sort").value || "severity";
    view = defects.filter(f);
    const key = { severity: d => -(SEV[d.class] || 0) * 1e6 - d.area_um2, area: d => -d.area_um2, score: d => -d.score, id: d => d.id,
      p_real: d => -(d.p_real == null ? -1 : d.p_real), uncertain: d => (d.review.status !== "unreviewed" ? 1 : 0) + Math.abs((d.p_real == null ? 0.5 : d.p_real) - 0.5) }[sort] || (d => d.id);
    view.sort((a, b) => key(a) - key(b));
    $("list-count").textContent = view.length === defects.length ? `${view.length.toLocaleString()} candidates` : `${view.length.toLocaleString()} of ${defects.length.toLocaleString()}`;
    $("list-spacer").style.height = `${Math.max(view.length * ROW_H, 1)}px`;
    renderRows();
    renderChips();
    safeDraw();
  }

  function rowHtml(d, i) {
    const kind = d.kind.replace("_metal", "");
    const sub = [`${fmt(d.area_um2, 1)} µm²`, `score ${fmt(d.score)}`];
    if (kind !== d.class) sub.unshift(label(kind));
    if (d.repeat_ncc != null) sub.push(`twin ${fmt(d.repeat_ncc)}`);
    if (d.p_real != null) sub.push(`<b>p ${fmt(d.p_real)}</b>`);
    if (d.review.cls && d.review.cls !== d.class) sub.push(`→ <b>${esc(label(d.review.cls))}</b>`);
    return `<div class="row ${i === active ? "active" : ""}" data-i="${i}" style="top:${i * ROW_H}px">
      <img loading="lazy" src="/api/runs/${encodeURIComponent(run.name)}/thumb/${d.id}.jpg" alt="">
      <div class="row-main">
        <div class="row-title c-${esc(d.class)}"><span class="dot"></span><span class="cls">${esc(label(d.class))}</span><span class="mono">#${d.id}</span>${selected.has(d.id) ? '<span class="sel" title="in re-image list">●</span>' : ""}</div>
        <div class="row-sub">${sub.join(" · ")}</div>
      </div>
      <span class="pill ${d.review.status}">${esc(label(d.review.status))}</span>
    </div>`;
  }

  function renderRows() {
    cancelAnimationFrame(rowsFrame);
    const list = $("list"), sp = $("list-spacer");
    if (!view.length) { sp.innerHTML = `<div class="list-empty">No candidates match this filter.</div>`; return; }
    const top = list.scrollTop, h = list.clientHeight || 600;
    const i0 = Math.max(0, Math.floor(top / ROW_H) - 6), i1 = Math.min(view.length, Math.ceil((top + h) / ROW_H) + 6);
    let html = "";
    for (let i = i0; i < i1; i++) html += rowHtml(view[i], i);
    sp.innerHTML = html;
  }

  function scrollToRow(i) {
    const list = $("list"), y = i * ROW_H;
    if (y < list.scrollTop) list.scrollTop = y;
    else if (y + ROW_H > list.scrollTop + list.clientHeight) list.scrollTop = y + ROW_H - list.clientHeight;
    renderRows();
  }

  let statsCache = null;
  function renderChips() {
    if (!statsCache) return;
    const q = ($("q").value || "").trim().toLowerCase();
    $("class-chips").innerHTML = Object.entries(statsCache.by_class).sort((a, b) => b[1].n - a[1].n).map(([c, v]) =>
      `<button class="chip c-${esc(c)} ${q === c ? "on" : ""}" data-cls="${esc(c)}" title="Show only ${esc(label(c))}"><span class="dot"></span>${esc(label(c))}<span class="n">${v.n.toLocaleString()}</span></button>`).join("");
  }

  async function refreshStats() {
    try {
      const s = await api(`/api/runs/${encodeURIComponent(run.name)}/stats`);
      statsCache = s;
      const real = Object.values(s.by_class).reduce((a, v) => a + v.real, 0), fa = Object.values(s.by_class).reduce((a, v) => a + v.false_alarm, 0);
      $("st-n").textContent = s.n.toLocaleString();
      $("st-rev").textContent = s.n_reviewed.toLocaleString();
      $("st-prec").textContent = real + fa ? `${(100 * real / (real + fa)).toFixed(0)}%` : "–";
      $("st-progress").value = s.n ? Math.max(s.n_reviewed ? 0.5 : 0, (100 * s.n_reviewed) / s.n) : 0;
      $("lm-real-n").textContent = real; $("lm-false-n").textContent = fa;
      $("lm-real").value = Math.min(100, (100 * real) / MIN_LABELS); $("lm-false").value = Math.min(100, (100 * fa) / MIN_LABELS);
      $("btn-learn").disabled = real < MIN_LABELS || fa < MIN_LABELS;
      renderChips();
    } catch (e) { fail(e); }
  }

  // ---------------------------------------------------------------- inspector
  function renderInspector() {
    const d = view[active];
    $("defect-empty").classList.toggle("hidden", !!d);
    $("defect-body").classList.toggle("hidden", !d);
    $("prev").disabled = active <= 0; $("next").disabled = active >= view.length - 1;
    if (!d) return;
    const src = `/api/runs/${encodeURIComponent(run.name)}/thumb/${d.id}.jpg?size=320`;
    if ($("d-thumb").getAttribute("src") !== src) $("d-thumb").src = src;
    $("d-class").className = `cls-tag c-${d.class}`;
    $("d-class").innerHTML = `<span class="dot"></span>${esc(label(d.class))}`;
    $("d-id").textContent = `#${d.id}`;
    const props = [
      ["Detector", label(d.kind.replace("_metal", ""))],
      ["Area", `${fmt(d.area_um2, 2)} µm² · ${d.area_px != null ? d.area_px.toLocaleString() : "–"} px`],
      ["Score", fmt(d.score)],
    ];
    if (d.z != null) props.push(["Contrast", `${fmt(d.z, 1)} σ`]);
    if (d.n_twins != null) props.push(["Twins compared", d.n_twins]);
    if (d.repeat_ncc != null) props.push(["Twin match", fmt(d.repeat_ncc)]);
    if (d.p_real != null) props.push(["Learned P(real)", fmt(d.p_real)]);
    if (d.centroid_um) props.push(["Position", `${fmt(d.centroid_um[0], 1)}, ${fmt(d.centroid_um[1], 1)} µm`]);
    if (d.review.reviewer) props.push(["Reviewed by", esc(d.review.reviewer)]);
    $("d-props").innerHTML = props.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    document.querySelectorAll(".v-btn").forEach(b => b.classList.toggle("on", b.dataset.status === d.review.status));
    $("d-cls").value = d.review.cls || "";
    $("d-select").classList.toggle("on", selected.has(d.id));
  }

  function setActive(i, fly) {
    if (i < 0 || i >= view.length) return;
    active = i;
    scrollToRow(i);
    const d = view[i];
    history.replaceState(null, "", `?run=${encodeURIComponent(run.name)}&d=${d.id}`);
    if (d && fly !== false && d.bbox_px && viewer) {
      const b = d.bbox_px, size = Math.max(b.x1 - b.x0, b.y1 - b.y0) * 6 + 80;
      const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
      viewer.viewport.fitBounds(imageRect(cx - size / 2, cy - size / 2, size, size), false);
    }
    renderInspector();
    safeDraw();
  }

  async function setReview(status, cls) {
    const d = view[active]; if (!d) return;
    const body = { defect_id: d.id, status, cls: cls || d.review.cls || null, reviewer: $("reviewer").value || null };
    if (cls && status === undefined) body.status = d.review.status === "unreviewed" ? "real" : d.review.status;
    try {
      await api(`/api/runs/${encodeURIComponent(run.name)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (e) { fail(e); return; }
    d.review = { ...d.review, status: body.status, cls: body.cls, reviewer: body.reviewer };
    renderRows(); renderInspector(); safeDraw(); refreshStats();
  }

  function toggleSelected() {
    const d = view[active]; if (!d) return;
    selected.has(d.id) ? selected.delete(d.id) : selected.add(d.id);
    updateSelCount(); renderRows(); renderInspector(); safeDraw();
  }
  function updateSelCount() { $("sel-count").textContent = selected.size; $("sel-count").classList.toggle("hidden", !selected.size); }

  // ---------------------------------------------------------------- keys & controls
  function typingTarget(e) {
    return e.composedPath().some(el => el.tagName && (["INPUT", "TEXTAREA", "SL-INPUT", "SL-SELECT", "SL-RANGE"].includes(el.tagName) || el.isContentEditable));
  }
  document.addEventListener("keydown", e => {
    if (e.metaKey || e.ctrlKey || e.altKey || typingTarget(e) || $("help").open) return;
    if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); setActive(Math.min(active + 1, view.length - 1), true); }
    else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); setActive(Math.max(active - 1, 0), true); }
    else if (e.key === "r") setReview("real");
    else if (e.key === "f") setReview("false_alarm");
    else if (e.key === "u") setReview("unsure");
    else if (e.key === "x") toggleSelected();
    else if (CLASS_KEYS[e.key]) setReview(undefined, CLASS_KEYS[e.key]);
    else if (e.key === "m") setMeasuring(!measuring);
    else if (e.key === "?") $("help").show();
    else if (e.key === "Escape") { setMeasuring(false); }
  });

  function wire() {
    $("list").addEventListener("scroll", () => { cancelAnimationFrame(rowsFrame); rowsFrame = requestAnimationFrame(renderRows); }, { passive: true });
    $("list").addEventListener("click", e => { const row = e.target.closest(".row"); if (row) setActive(+row.dataset.i, true); });
    if (window.ResizeObserver) new ResizeObserver(() => renderRows()).observe($("list"));
    $("class-chips").addEventListener("click", e => {
      const c = e.target.closest(".chip"); if (!c) return;
      const q = $("q"); q.value = (q.value || "").trim().toLowerCase() === c.dataset.cls ? "" : c.dataset.cls;
      active = -1; $("list").scrollTop = 0; refreshList(); renderInspector();
    });
    const onFilter = () => { active = -1; $("list").scrollTop = 0; refreshList(); renderInspector(); };
    $("q").addEventListener("sl-input", onFilter);
    $("q").addEventListener("sl-clear", onFilter);
    $("sort").addEventListener("sl-change", onFilter);

    $("btn-measure").addEventListener("click", () => setMeasuring(!measuring));
    $("measure-close").addEventListener("click", () => { $("card-measure").classList.add("hidden"); measurePts = []; safeDraw(); });
    $("btn-help").addEventListener("click", () => $("help").show());
    $("btn-positions").addEventListener("click", () => {
      if (!selected.size) { notify("Select candidates for re-imaging with X first.", "warning", "exclamation-triangle"); return; }
      location.href = `/api/runs/${encodeURIComponent(run.name)}/positions.xml?ids=${[...selected].join(",")}`;
    });

    $("prev").addEventListener("click", () => setActive(active - 1, true));
    $("next").addEventListener("click", () => setActive(active + 1, true));
    document.querySelectorAll(".v-btn").forEach(b => b.addEventListener("click", () => setReview(b.dataset.status)));
    $("d-cls").addEventListener("sl-change", e => { const v = e.target.value; const d = view[active]; if (v && d && v !== d.review.cls) setReview(undefined, v); });
    $("d-select").addEventListener("click", toggleSelected);

    $("zoom-in").addEventListener("click", () => viewer && viewer.viewport.zoomBy(1.6));
    $("zoom-out").addEventListener("click", () => viewer && viewer.viewport.zoomBy(1 / 1.6));
    $("zoom-fit").addEventListener("click", () => viewer && viewer.viewport.goHome());

    $("layer-design").addEventListener("sl-change", e => { if (layers.design) layers.design.setOpacity(e.target.checked ? 1 : 0); });
    $("layer-pred").addEventListener("sl-change", e => { if (layers.pred) layers.pred.setOpacity(e.target.checked ? 1 : 0); });
    $("layer-defects").addEventListener("sl-change", safeDraw);

    $("thr").tooltipFormatter = v => v.toFixed(2);
    $("btn-apply").addEventListener("click", async () => {
      const btn = $("btn-apply"); btn.loading = true;
      try {
        const q = `threshold=${$("thr").value}&min_area_px=${parseInt($("minpx").value, 10) || 1}&merge_px=8&include_specks=${$("specks").checked}&include_die2die=${$("die2die").checked}`;
        const r = await api(`/api/runs/${encodeURIComponent(run.name)}/reextract?${q}`, { method: "POST" });
        defects = await api(`/api/runs/${encodeURIComponent(run.name)}/defects`);
        active = -1; refreshList(); refreshStats(); renderInspector();
        notify(`Re-extracted ${r.n.toLocaleString()} candidates.`, "success", "check2-circle");
      } catch (e) { fail(e); }
      btn.loading = false;
    });
    $("btn-learn").addEventListener("click", async () => {
      const btn = $("btn-learn"); btn.loading = true;
      try {
        const r = await api(`/api/runs/${encodeURIComponent(run.name)}/learn`, { method: "POST" });
        defects = await api(`/api/runs/${encodeURIComponent(run.name)}/defects`);
        $("sort").value = "p_real"; active = -1; $("list").scrollTop = 0; refreshList(); refreshStats(); renderInspector();
        const m = r.metrics;
        $("learn-result").innerHTML = `Trained on <b>${m.n_pos}</b> real and <b>${m.n_neg}</b> false${m.val_acc != null ? ` · held-out accuracy <b>${(100 * m.val_acc).toFixed(0)}%</b>` : ""}. List sorted by learned confidence.`;
        $("learn-result").classList.remove("hidden");
        notify(`Ranked ${r.n_scored.toLocaleString()} candidates.`, "success", "stars");
      } catch (e) { fail(e); }
      btn.loading = false;
    });

    $("reviewer").value = store.get("chipscan.reviewer") || "";
    $("reviewer").addEventListener("sl-change", e => store.set("chipscan.reviewer", e.target.value));
  }

  const TAGS = ["sl-select", "sl-option", "sl-input", "sl-button", "sl-switch", "sl-range", "sl-dialog", "sl-progress-bar", "sl-icon-button", "sl-badge"];
  Promise.all(TAGS.map(t => customElements.whenDefined(t)))
    .then(() => { wire(); return loadRuns(); })
    .catch(e => { document.body.innerHTML = `<pre style="padding:20px;color:#f26b6b">${esc(e.stack || e)}</pre>`; });
})();
