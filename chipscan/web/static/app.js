/* chipscan viewer front end: add scans, track jobs, OpenSeadragon canvas, virtualised findings list,
   three finding groups with risk levels, inspector, keyboard review. */
(() => {
  const COLORS = { anomaly: "#ff7a1a", speck: "#8fd3ff", die2die: "#ff4fb8", short: "#ff5c5c", open: "#e15cff", particle: "#ffab4a", protrusion: "#ffe066", notch: "#5ce1ff", void: "#5cb8ff", missing_feature: "#b77cff", extra_feature: "#b77cff", unclassified: "#4fcb7b" };
  const GROUP_COLORS = { speck: "#0a84ff", trace: "#bf5af2", component: "#ff375f" };
  const LEVEL_COLORS = { critical: "#ff4d5e", high: "#ff9f40", medium: "#f2d24e", low: "#7d8796" };
  const LEVEL_WIDTH = { critical: 2.4, high: 1.9, medium: 1.4, low: 1.1 };
  const SEV = { anomaly: 3, speck: 2, die2die: 4, short: 5, open: 5, missing_feature: 4, extra_feature: 4, particle: 3, protrusion: 2, notch: 2, void: 1, unclassified: 0 };
  const CLASS_KEYS = { "1": "short", "2": "open", "3": "particle", "4": "protrusion", "5": "notch", "6": "void" };
  const LABEL = {
    die2die: "die-to-die", false_alarm: "false alarm", missing_feature: "missing feature", extra_feature: "extra feature",
    trace: "trace error", component_defect: "component defect", component_deformed: "deformed component", component_missing: "missing component", probe_mark: "probe mark",
  };
  const GROUP_LABEL = { speck: "Speck", trace: "Trace error", component: "Component error" };
  const ROW_H = 64;
  const MIN_LABELS = 5;

  const $ = (id) => document.getElementById(id);
  const label = (s) => LABEL[s] || String(s).replace(/_/g, " ");
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, n = 2) => (v == null || Number.isNaN(v) ? "–" : (+v).toFixed(n));
  const fmtBytes = (b) => b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`;
  const store = { get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };
  const colorOf = (d) => (d.group && GROUP_COLORS[d.group]) || COLORS[d.class] || COLORS.unclassified;

  let runs = [], run = null, defects = [], view = [], active = -1, selected = new Set();
  let viewer = null, layers = { design: null, pred: null };
  let measuring = false, measurePts = [];
  let rowsFrame = 0;
  const groupOn = { speck: true, trace: true, component: true };
  const levelOn = { critical: true, high: true, medium: true, low: true };

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

  async function notify(message, variant = "primary", icon = "info-circle", action) {
    const el = Object.assign(document.createElement("sl-alert"), { variant, closable: true, duration: action ? 12000 : 6000 });
    el.innerHTML = `<sl-icon slot="icon" name="${icon}"></sl-icon>${esc(message)}`;
    if (action) {
      const b = Object.assign(document.createElement("sl-button"), { size: "small", variant: "text", textContent: action.label });
      b.style.marginLeft = "8px";
      b.addEventListener("click", () => { el.hide(); action.run(); });
      el.append(b);
    }
    document.body.append(el);
    await customElements.whenDefined("sl-alert");
    el.toast();
  }
  const fail = (e) => notify(e.message || String(e), "danger", "exclamation-octagon");

  function renderRunOptions(sel) {
    // full scans first; earlier single-model experiment runs grouped underneath
    const opt = (r) => {
      const i = runs.indexOf(r);
      // the select shows an option's unslotted text, so only the name goes there
      return `<sl-option value="r${i}">${esc(r.name)}<small slot="suffix">${r.n_defects.toLocaleString()} found</small></sl-option>`;
    };
    const scans = runs.filter(r => r.scan_run), other = runs.filter(r => !r.scan_run);
    let html = scans.length ? `<small>Scans</small>${scans.map(opt).join("")}` : "";
    if (other.length) html += `${scans.length ? "<sl-divider></sl-divider>" : ""}<small>Earlier experiments</small>${other.map(opt).join("")}`;
    sel.innerHTML = html;
  }

  async function loadRuns(prefer) {
    runs = await api("/api/runs");
    const sel = $("run");
    renderRunOptions(sel);
    $("welcome").classList.toggle("hidden", runs.length > 0);
    if (!runs.length) return;
    const wanted = prefer || new URLSearchParams(location.search).get("run");
    const name = wanted && runs.some(r => r.name === wanted) ? wanted : (run && runs.some(r => r.name === run.name) ? run.name : runs[0].name);
    await sel.updateComplete;
    sel.value = `r${runs.findIndex(r => r.name === name)}`;
    await openRun(name);
  }


  async function openRun(name) {
    run = runs.find(r => r.name === name);
    const wantedId = new URLSearchParams(location.search).get("d");
    history.replaceState(null, "", `?run=${encodeURIComponent(name)}`);
    const reg = run.registration || {};
    const meta = [["", `${run.width.toLocaleString()} × ${run.height.toLocaleString()} px`], ["", `${fmt(run.pixel_size_um, 3)} µm/px`]];
    if (run.scan_run && run.scan) {
      const st = run.scan.straighten || {}, lay = run.scan.layout || {};
      meta.push(["tilt fixed", `${fmt(st.angle_deg, 3)}°`]);
      meta.push(["tiling", st.seams ? `fixed, ${(st.frame_px || []).map(v => Math.round(v)).join(" × ")} px frames` : "no seams"]);
      meta.push(["layout from", lay.design_used ? "GDS + model" : "model"]);
    } else {
      if (run.segmenter) meta.push(["Method", esc(run.segmenter)]);
      if (!run.anomaly_run && reg.rotation_deg != null) {
        meta.push(["Registration", `rot ${fmt(reg.rotation_deg)}° · scale ${fmt(reg.scale, 4)}${reg.flipped ? " · mirrored" : ""}`]);
        meta.push(["Fine rms", run.fine_rms_px != null ? fmt(run.fine_rms_px) + " px" : "n/a"]);
      }
    }
    $("runinfo").innerHTML = meta.map(([k, v]) => k ? `${k} <b>${v}</b>` : `<b>${v}</b>`).join('<span class="sep">·</span>');
    $("runinfo").title = meta.map(([k, v]) => k ? `${k}: ${v}` : v).join("\n");
    $("btn-export").href = `/api/runs/${encodeURIComponent(name)}/export.csv`;
    defects = await api(`/api/runs/${encodeURIComponent(name)}/defects`);
    selected.clear(); active = -1; updateSelCount();
    const scan = !!run.scan_run;
    $("sens").classList.toggle("hidden", !run.anomaly_run);
    $("sens-scan").classList.toggle("hidden", !scan);
    $("group-filter").classList.toggle("hidden", !scan);
    $("risk-filter").classList.toggle("hidden", !scan);
    $("class-chips").classList.toggle("hidden", scan);
    $("layout-legend").classList.toggle("hidden", !scan);
    $("learn-result").classList.add("hidden");
    $("layer-design-label").textContent = scan ? "Layout (traces · components)" : "Design edges";
    $("layer-pred-label").textContent = scan ? "Detector heatmap" : "Prediction heatmap";
    if (scan) {
      const sp = (run.scan && run.scan.spec) || {};
      $("thr-speck").value = +(sp.speck_threshold ?? 0.5); $("thr-trace").value = +(sp.trace_threshold ?? 0.5); $("thr-trace").hidden = sp.report_traces !== true; $("minpx-scan").value = String(sp.min_area_px ?? 4);
      $("sort").value = "risk";
      $("layer-design").checked = false;
    } else if ($("sort").value === "risk") {
      $("sort").value = "severity";
    }
    if (run.anomaly_run && run.extract_params) {
      const p = run.extract_params;
      $("thr").value = +p.threshold; $("minpx").value = String(p.min_area_px);
      $("specks").checked = p.include_specks !== false; $("die2die").checked = p.include_die2die !== false;
    }
    $("layer-design").classList.toggle("hidden", !!run.anomaly_run);
    buildViewer();
    refreshMetrology(name);
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
    const host = $("navigator-host"); host.innerHTML = '<div id="navigator"></div>';
    viewer = OpenSeadragon({
      id: "viewer", tileSources: tileSource("tiles", "jpg"),
      showNavigationControl: false,
      showNavigator: true, navigatorId: "navigator", navigatorAutoFade: false, navigatorAutoResize: true,
      navigatorBackground: "#e8e8ed", navigatorBorderColor: "transparent", navigatorDisplayRegionColor: "#0071e3", navigatorOpacity: 1,
      maxZoomPixelRatio: 8, minZoomImageRatio: 0.5, animationTime: 0.5, gestureSettingsMouse: { clickToZoom: false },
      crossOriginPolicy: false, imageSmoothingEnabled: false, visibilityRatio: 0.2,
    });
    viewer.addHandler("open", () => {
      viewer.addTiledImage({ tileSource: tileSource("design", "png"), opacity: $("layer-design").checked && !run.anomaly_run ? 1 : 0, success: e => { layers.design = e.item; } });
      viewer.addTiledImage({ tileSource: tileSource("pred", "png"), opacity: $("layer-pred").checked ? 1 : 0, success: e => { layers.pred = e.item; } });
      updateZoom();
    });
    // our review keys own the keyboard: OpenSeadragon's own (r rotates, arrows / w a s d pan, 0 homes) would fight them
    viewer.addHandler("canvas-key", e => { e.preventDefaultAction = true; });
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

  function baseImage() { return viewer && viewer.world.getItemCount() ? viewer.world.getItemAt(0) : null; }
  function toImage(pt) { const b = baseImage(); return b ? b.viewerElementToImageCoordinates(pt) : viewer.viewport.viewerElementToImageCoordinates(pt); }
  function toScreen(x, y) { const b = baseImage(); const p = new OpenSeadragon.Point(x, y); return b ? b.imageToViewerElementCoordinates(p) : viewer.viewport.imageToViewerElementCoordinates(p); }
  function imageRect(x, y, w, h) { const b = baseImage(); return b ? b.imageToViewportRectangle(x, y, w, h) : viewer.viewport.imageToViewportRectangle(x, y, w, h); }
  function imageZoom() { const b = baseImage(); if (!b) return 0; return b.viewportToImageZoom(viewer.viewport.getZoom(true)); }

  function updateZoom() { const z = imageZoom(); $("zoom-val").textContent = z ? (z >= 1 ? `${(z * 100).toFixed(0)}%` : `${(z * 100).toFixed(z < 0.1 ? 1 : 0)}%`) : "–"; }

  function safeDraw() { try { drawOverlay(); } catch (e) { console.warn("overlay draw failed", e); } }

  function visible(d) {
    if (!run || !run.scan_run) return true;
    return groupOn[d.group] !== false && levelOn[d.risk_level] !== false;
  }


  // ---------------------------------------------------------------- live findings heatmap
  // Findings per mm² in 500 µm squares, from what the list shows now (risk and group chips,
  // search box), leaving out false alarms; rebuilt when those change, drawn under the outlines.
  const HEAT_CELL_UM = 500;
  const INFERNO = [[0, 0, 4], [66, 10, 104], [147, 38, 103], [221, 81, 58], [252, 165, 10], [252, 255, 164]];
  let heat = null;  // { canvas, cell, top }
  function heatColor(t) {
    const x = t * (INFERNO.length - 1), i = Math.min(Math.floor(x), INFERNO.length - 2), f = x - i, a = INFERNO[i], b = INFERNO[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function blur1d(g, w, h, horiz) {
    const k = [0.054, 0.242, 0.399, 0.242, 0.054], out = new Float32Array(g.length);  // Gaussian, sigma 1 cell
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let v = 0;
      for (let j = -2; j <= 2; j++) {
        const xx = horiz ? x + j : x, yy = horiz ? y : y + j;
        if (xx >= 0 && xx < w && yy >= 0 && yy < h) v += g[yy * w + xx] * k[j + 2];
      }
      out[y * w + x] = v;
    }
    return out;
  }
  function buildHeat() {
    heat = null;
    if (!run || !$("layer-heat").checked) return;
    const px = run.pixel_size_um || 1, cell = HEAT_CELL_UM / px;
    const gw = Math.max(1, Math.ceil(run.width / cell)), gh = Math.max(1, Math.ceil(run.height / cell));
    let g = new Float32Array(gw * gh);
    for (const d of view) {
      const b = d.bbox_px; if (!b || d.review.status === "false_alarm") continue;
      const cx = Math.min(gw - 1, Math.max(0, Math.floor((b.x0 + b.x1) / 2 / cell))), cy = Math.min(gh - 1, Math.max(0, Math.floor((b.y0 + b.y1) / 2 / cell)));
      g[cy * gw + cx] += 1;
    }
    g = blur1d(blur1d(g, gw, gh, true), gw, gh, false);
    const perMm2 = 1 / (HEAT_CELL_UM / 1000) ** 2;
    const pos = Array.from(g).filter(v => v > 1e-3).sort((a, b) => a - b);
    const top = Math.max(pos.length ? pos[Math.min(pos.length - 1, Math.floor(pos.length * 0.99))] : 1, 1e-6);
    const c = document.createElement("canvas"); c.width = gw; c.height = gh;
    const ctx = c.getContext("2d"), img = ctx.createImageData(gw, gh);
    for (let i = 0; i < g.length; i++) {
      const t = Math.min(g[i] / top, 1), rgb = heatColor(t);
      img.data[4 * i] = rgb[0]; img.data[4 * i + 1] = rgb[1]; img.data[4 * i + 2] = rgb[2];
      img.data[4 * i + 3] = Math.min(t * 4, 0.8) * 255;
    }
    ctx.putImageData(img, 0, 0);
    heat = { canvas: c, cell, w: gw, h: gh };
    $("heat-top").textContent = `${(top * perMm2).toLocaleString(undefined, { maximumSignificantDigits: 3 })}+ per mm²`;
  }
  function drawHeat(ctx) {
    if (!heat) return;
    const p0 = toScreen(0, 0), p1 = toScreen(heat.w * heat.cell, heat.h * heat.cell);
    ctx.save(); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(heat.canvas, p0.x, p0.y, p1.x - p0.x, p1.y - p0.y);
    ctx.restore();
  }

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
    drawHeat(ctx);
    if ($("layer-defects").checked) {
      // low risk first so the important outlines sit on top
      const order = run.scan_run ? [...defects].sort((a, b) => (a.risk || 0) - (b.risk || 0)) : defects;
      for (const d of order) {
        const b = d.bbox_px; if (!b || d === act || !visible(d)) continue;
        const p0 = toScreen(b.x0, b.y0), p1 = toScreen(b.x1, b.y1);
        if (p1.x < -20 || p1.y < -20 || p0.x > W + 20 || p0.y > H + 20) continue;
        strokeDefect(ctx, d, p0, p1, zoom, false);
      }
    }
    if (act && act.bbox_px) {
      const b = act.bbox_px, p0 = toScreen(b.x0, b.y0), p1 = toScreen(b.x1, b.y1);
      strokeDefect(ctx, act, p0, p1, zoom, true);
      const txt = run.scan_run ? `#${act.id} · ${label(act.class)} · risk ${fmt(act.risk, 0)}` : `#${act.id} · ${label(act.class)}`;
      drawLabel(ctx, txt, (p0.x + p1.x) / 2, Math.min(p0.y, p1.y) - 10, colorOf(act), W);
    }
    if (measurePts.length) {
      ctx.strokeStyle = "#4fcb7b"; ctx.fillStyle = "#4fcb7b"; ctx.lineWidth = 2; ctx.beginPath();
      measurePts.forEach((p, k) => { const s = toScreen(p.x, p.y); k ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
      ctx.stroke();
      measurePts.forEach(p => { const s = toScreen(p.x, p.y); ctx.beginPath(); ctx.arc(s.x, s.y, 3.5, 0, 6.283); ctx.fill(); });
    }
  }

  function strokeDefect(ctx, d, p0, p1, zoom, isActive) {
    const col = colorOf(d);
    const base = run.scan_run ? (LEVEL_WIDTH[d.risk_level] || 1.25) : 1.25;
    ctx.strokeStyle = col; ctx.lineWidth = isActive ? base + 1.2 : base;
    ctx.globalAlpha = d.review.status === "false_alarm" ? 0.45 : (run.scan_run && d.risk_level === "low" ? 0.7 : 1);
    ctx.setLineDash(d.review.status === "false_alarm" ? [4, 3] : []);
    if (zoom > 1.5 && d.polygons_px && d.polygons_px.length) {
      for (const poly of d.polygons_px) {
        ctx.beginPath();
        poly.forEach((p, k) => { const s = toScreen(p[0], p[1]); k ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
        ctx.closePath(); ctx.stroke();
      }
    } else if (!isActive && Math.max(p1.x - p0.x, p1.y - p0.y) < 4) {
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, run.scan_run && (d.risk_level === "critical" || d.risk_level === "high") ? 3 : 2, 0, 6.283); ctx.fill();
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
      if (t.startsWith("risk>")) return (d.risk || 0) > parseFloat(t.slice(5));
      if (t.startsWith(">")) return d.area_um2 > parseFloat(t.slice(1));
      if (t.startsWith("<")) return d.area_um2 < parseFloat(t.slice(1));
      if (t.startsWith("#")) return String(d.id) === t.slice(1);
      if (t === "onjj") return !!d.on_jj;
      return d.class.includes(t) || d.kind.includes(t) || (d.group || "").includes(t) || d.review.status.includes(t) || (t === "selected" && selected.has(d.id));
    });
  }

  function refreshList() {
    const f = parseFilter($("q").value || ""), sort = $("sort").value || "severity";
    view = defects.filter(d => visible(d) && f(d));
    const key = { risk: d => -(d.risk || 0), severity: d => -(SEV[d.class] || 0) * 1e6 - d.area_um2, area: d => -d.area_um2, score: d => -d.score, id: d => d.id,
      p_real: d => -(d.p_real == null ? -1 : d.p_real), uncertain: d => (d.review.status !== "unreviewed" ? 1 : 0) + Math.abs((d.p_real == null ? 0.5 : d.p_real) - 0.5) }[sort] || (d => d.id);
    view.sort((a, b) => key(a) - key(b));
    $("list-count").textContent = view.length === defects.length ? `${view.length.toLocaleString()} findings` : `${view.length.toLocaleString()} of ${defects.length.toLocaleString()}`;
    $("list-spacer").style.height = `${Math.max(view.length * ROW_H, 1)}px`;
    renderRows();
    renderChips();
    buildHeat();
    safeDraw();
  }

  function rowHtml(d, i) {
    if (run.scan_run) {
      const sub = [`${fmt(d.size_um, 1)} µm`];
      if (d.group !== "component") sub.push(d.on_jj ? "<b>on JJ</b>" : `JJ ${fmt(d.d_jj_um, 0)} µm`);
      sub.push(d.on_trace ? "on trace" : `trace ${fmt(d.d_trace_um, 1)} µm`);
      if (d.group === "component" && d.n_ref) sub.push(`vs ${d.n_ref}`);
      const rv = d.review.status !== "unreviewed" ? `<span class="rv-dot ${d.review.status}" title="${esc(label(d.review.status))}"></span>` : "";
      return `<div class="row ${i === active ? "active" : ""}" data-i="${i}" style="top:${i * ROW_H}px">
        <img loading="lazy" src="/api/runs/${encodeURIComponent(run.name)}/thumb/${d.id}.jpg" alt="">
        <div class="row-main">
          <div class="row-title g-${esc(d.group)}"><span class="gdot"></span><span class="cls">${esc(label(d.class))}</span><span class="mono">#${d.id}</span>${rv}${selected.has(d.id) ? '<span class="sel" title="in re-image list">●</span>' : ""}</div>
          <div class="row-sub">${sub.join(" · ")}</div>
        </div>
        <span class="risk-pill lv-${d.risk_level}" title="${esc(d.risk_level)} risk">${fmt(d.risk, 0)}</span>
      </div>`;
    }
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
    if (!view.length) { sp.innerHTML = `<div class="list-empty">No findings match these filters.</div>`; return; }
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
    if (run && run.scan_run) {
      // group cards count everything in the group; level chips count within the enabled groups
      for (const g of ["speck", "trace", "component"]) {
        const n = defects.filter(d => d.group === g).length;
        const c = defects.filter(d => d.group === g && (d.risk_level === "critical" || d.risk_level === "high")).length;
        $(`gn-${g}`).textContent = n.toLocaleString();
        $(`gc-${g}`).textContent = c ? `${c.toLocaleString()} high+` : "";
        const card = document.querySelector(`.group-card[data-group="${g}"]`);
        card.classList.toggle("on", groupOn[g]);
        card.classList.toggle("hidden", g === "trace" && n === 0);  // trace errors are off by default
      }
      for (const lv of ["critical", "high", "medium", "low"]) {
        $(`lv-${lv}`).textContent = defects.filter(d => groupOn[d.group] && d.risk_level === lv).length.toLocaleString();
        document.querySelector(`#risk-filter .chip[data-level="${lv}"]`).classList.toggle("on", levelOn[lv]);
      }
      return;
    }
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
  function bar(name, v, hint) {
    const pct = Math.round(100 * Math.max(0, Math.min(1, v || 0)));
    return `<div class="rf"><span class="rf-name">${name}</span><span class="rf-track"><span class="rf-fill" style="width:${pct}%"></span></span><span class="rf-val mono">${hint}</span></div>`;
  }

  function renderInspector() {
    const d = view[active];
    $("defect-empty").classList.toggle("hidden", !!d);
    $("defect-body").classList.toggle("hidden", !d);
    $("prev").disabled = active <= 0; $("next").disabled = active >= view.length - 1;
    if (!d) return;
    const src = `/api/runs/${encodeURIComponent(run.name)}/thumb/${d.id}.jpg?size=320`;
    if ($("d-thumb").getAttribute("src") !== src) $("d-thumb").src = src;
    $("d-class").className = run.scan_run ? `cls-tag g-${d.group}` : `cls-tag c-${d.class}`;
    $("d-class").innerHTML = `<span class="${run.scan_run ? "gdot" : "dot"}"></span>${esc(label(d.class))}`;
    $("d-id").textContent = `#${d.id}`;
    const rb = $("d-risk");
    if (run.scan_run && d.risk != null) {
      const f = d.risk_factors || {};
      rb.innerHTML = `<div class="risk-head"><div class="risk-score lv-${d.risk_level}">${fmt(d.risk, 0)}</div>
          <div class="risk-meta"><span class="risk-level lv-${d.risk_level}">${esc(d.risk_level)} risk</span><span class="muted">${esc(GROUP_LABEL[d.group] || d.group)}</span></div></div>
        ${bar("Near a JJ", f.jj, d.on_jj ? "on JJ" : `${fmt(d.d_jj_um, 1)} µm`)}
        ${bar("Near a trace", f.trace, d.on_trace ? "on trace" : `${fmt(d.d_trace_um, 1)} µm`)}
        ${bar("Size", f.size, `${fmt(d.size_um, 1)} µm`)}
        ${bar("Confidence", f.confidence, fmt(f.confidence, 2))}`;
      rb.classList.remove("hidden");
    } else rb.classList.add("hidden");
    const props = [
      ["Detector", run.scan_run ? esc(GROUP_LABEL[d.group] || d.group) : label(d.kind.replace("_metal", ""))],
      ["Area", `${fmt(d.area_um2, 2)} µm² · ${d.area_px != null ? d.area_px.toLocaleString() : "–"} px`],
      ["Score", fmt(d.score)],
    ];
    if (d.group === "component") {
      if (d.reference) props.push(["Compared with", `${d.n_ref} ${d.reference === "line" ? "in its line" : "look-alikes"}`]);
      if (d.ncc != null) props.push(["Match to reference", fmt(d.ncc)]);
    }
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
      const f = findingFrame(d);
      // jump, don't fly: during review the animation (and the blurry tiles it passes over) is dead time
      viewer.viewport.fitBounds(imageRect(f.cx - f.size / 2, f.cy - f.size / 2, f.size, f.size), true);
      for (let k = 1; k <= 3; k++) prefetchFinding(view[i + k]);
      prefetchFinding(view[i - 1]);
    }
    renderInspector();
    safeDraw();
  }

  // The same review zoom for every finding: it fills ~1/5 of the view, but never past 4 screen px per image
  // px (tiny specks keep ~150 px of context to judge what they sit on); a big one shows whole with its surroundings.
  const REVIEW_FILL = 0.2, REVIEW_MAX_ZOOM = 4;
  function findingFrame(d) {
    const b = d.bbox_px, ext = Math.max(b.x1 - b.x0, b.y1 - b.y0, 1);
    const el = viewer && viewer.container, short = Math.min(el && el.clientWidth || 800, el && el.clientHeight || 600);
    const z = Math.min((short * REVIEW_FILL) / ext, REVIEW_MAX_ZOOM);  // screen px per image px
    return { cx: (b.x0 + b.x1) / 2, cy: (b.y0 + b.y1) / 2, size: short / z };
  }

  // warm the browser cache with the tiles a finding's view will ask for, so the next jump is sharp at once
  const prefetched = new Set();
  function prefetchFinding(d) {
    if (!d || !d.bbox_px || !viewer || !run) return;
    const el = viewer.container, cw = el.clientWidth || 800, ch = el.clientHeight || 600;
    const f = findingFrame(d), per = f.size / Math.min(cw, ch);  // image px per screen px
    const top = run.levels - 1, base = Math.floor(Math.log2(Math.max(1, per / (window.devicePixelRatio || 1))));
    const kinds = [["tiles", "jpg"]];
    if ($("layer-pred").checked) kinds.push(["pred", "png"]);
    // the sharp level and every coarser one (the viewer paints coarse tiles first; each level is 4x fewer)
    for (let L = Math.min(base, top); L <= top; L++) {
      const span = 256 * 2 ** L;
      const x0 = Math.max(0, Math.floor((f.cx - (cw * per) / 2) / span)), x1 = Math.floor(Math.min(run.width - 1, f.cx + (cw * per) / 2) / span);
      const y0 = Math.max(0, Math.floor((f.cy - (ch * per) / 2) / span)), y1 = Math.floor(Math.min(run.height - 1, f.cy + (ch * per) / 2) / span);
      for (const [kind, ext] of kinds)
        for (let y = y0; y <= y1; y++)
          for (let x = x0; x <= x1; x++) {
            const url = `/api/runs/${encodeURIComponent(run.name)}/${kind}/${L}/${x}/${y}.${ext}`;
            if (prefetched.has(url)) continue;
            prefetched.add(url);
            new Image().src = url;
          }
    }
  }

  async function setReview(status, cls) {
    const d = view[active]; if (!d) return;
    const body = { defect_id: d.id, status, cls: cls || d.review.cls || null, reviewer: $("reviewer").value || null };
    if (cls && status === undefined) body.status = d.review.status === "unreviewed" ? "real" : d.review.status;
    try {
      await api(`/api/runs/${encodeURIComponent(run.name)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (e) { fail(e); return; }
    d.review = { ...d.review, status: body.status, cls: body.cls, reviewer: body.reviewer };
    renderRows(); renderInspector(); buildHeat(); safeDraw(); refreshStats();
  }

  function toggleSelected() {
    const d = view[active]; if (!d) return;
    selected.has(d.id) ? selected.delete(d.id) : selected.add(d.id);
    updateSelCount(); renderRows(); renderInspector(); safeDraw();
  }
  function updateSelCount() { $("sel-count").textContent = selected.size; $("sel-count").classList.toggle("hidden", !selected.size); }

  // ---------------------------------------------------------------- add scan
  let pickedFile = null, pickedGds = null;

  function openAdd() {
    pickedFile = null; pickedGds = null;
    $("drop-file").classList.add("hidden"); $("drop").classList.remove("has-file");
    $("gds-name").textContent = "Choose a GDS file";
    $("upload-progress").classList.add("hidden");
    $("add-submit").disabled = false; $("add-submit").loading = false;
    $("add-dialog").show();
  }

  function setFile(f) {
    pickedFile = f;
    $("drop-file").innerHTML = `<sl-icon name="file-earmark-image"></sl-icon><span>${esc(f.name)}</span><span class="muted mono">${fmtBytes(f.size)}</span>`;
    $("drop-file").classList.remove("hidden");
    $("drop").classList.add("has-file");
    if (!$("scan-name").value) $("scan-name").placeholder = f.name.replace(/\.[^.]+$/, "");
  }

  function submitScan() {
    const path = ($("image-path").value || "").trim();
    if (!pickedFile && !path) { notify("Choose an image or give a path on this computer.", "warning", "exclamation-triangle"); return; }
    const fd = new FormData();
    if (pickedFile) fd.append("image", pickedFile, pickedFile.name); else fd.append("image_path", path);
    fd.append("name", $("scan-name").value || "");
    fd.append("pixel_size_um", $("scan-px").value || "");
    fd.append("straighten", $("scan-straighten").checked ? "true" : "false");
    if (pickedGds) fd.append("gds", pickedGds, pickedGds.name);
    fd.append("gds_path", $("gds-path").value || "");
    fd.append("gds_cell", $("gds-cell").value || "");
    fd.append("trace_layers", $("gds-trace").value || "");
    fd.append("component_layers", $("gds-comp").value || "");
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/scans");
    $("add-submit").loading = true;
    if (pickedFile) {
      $("upload-progress").classList.remove("hidden");
      xhr.upload.onprogress = e => {
        if (!e.lengthComputable) return;
        const pct = (100 * e.loaded) / e.total;
        $("up-bar").value = pct; $("up-pct").textContent = `${pct.toFixed(0)}%`;
        $("up-label").textContent = pct >= 100 ? "Starting…" : `Uploading ${fmtBytes(e.loaded)} of ${fmtBytes(e.total)}`;
      };
    }
    xhr.onload = () => {
      $("add-submit").loading = false;
      if (xhr.status >= 200 && xhr.status < 300) {
        const job = JSON.parse(xhr.responseText);
        $("add-dialog").hide();
        $("image-path").value = ""; $("scan-name").value = "";
        notify(`Started “${job.name}”. You can keep reviewing while it runs.`, "primary", "activity", { label: "Show progress", run: () => $("jobs-drawer").show() });
        pollJobs(true);
      } else {
        let msg = xhr.responseText; try { msg = JSON.parse(msg).detail; } catch (e) {}
        $("upload-progress").classList.add("hidden");
        fail(new Error(msg || `upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => { $("add-submit").loading = false; fail(new Error("upload failed: connection lost")); };
    xhr.send(fd);
  }

  // ---------------------------------------------------------------- jobs
  let jobsTimer = 0, lastJobs = {};
  const STAGE_ICON = { done: "check-circle-fill", running: "arrow-repeat", pending: "circle" };

  function renderJobs(jobs) {
    const live = jobs.filter(j => j.status === "queued" || j.status === "running").length;
    $("jobs-count").textContent = live; $("jobs-count").classList.toggle("hidden", !live);
    if (!jobs.length) { $("jobs-list").innerHTML = `<div class="empty"><sl-icon name="inbox"></sl-icon><div>No scans started in this session.</div></div>`; return; }
    $("jobs-list").innerHTML = jobs.map(j => {
      const done = j.stages.filter(s => s.status === "done").length;
      const cur = j.stages.find(s => s.status === "running");
      const overall = (100 * (done + (cur ? cur.progress : 0))) / j.stages.length;
      const head = j.status === "done" ? `<sl-badge variant="success" pill>done</sl-badge>` : j.status === "failed" ? `<sl-badge variant="danger" pill>failed</sl-badge>` : j.status === "queued" ? `<sl-badge variant="neutral" pill>queued</sl-badge>` : `<sl-badge variant="primary" pill>${overall.toFixed(0)}%</sl-badge>`;
      return `<div class="job ${j.status}">
        <div class="job-head"><span class="job-name">${esc(j.name)}</span>${head}</div>
        ${j.status === "running" ? `<sl-progress-bar value="${overall}"></sl-progress-bar>` : ""}
        <ol class="stages">${j.stages.map(s => `<li class="st-${s.status}"><sl-icon name="${STAGE_ICON[s.status] || "circle"}"></sl-icon><span>${esc(s.label)}</span>${s.status === "running" && s.progress > 0 ? `<span class="mono muted">${(100 * s.progress).toFixed(0)}%</span>` : ""}</li>`).join("")}</ol>
        ${j.error ? `<div class="job-error">${esc(j.error)}</div>` : ""}
        ${j.status === "done" ? `<sl-button size="small" variant="primary" class="job-open" data-name="${esc(j.name)}"><sl-icon slot="prefix" name="box-arrow-up-right"></sl-icon>Open results</sl-button>` : ""}
      </div>`;
    }).join("");
  }

  async function refreshMetrology(name) {
    let m = { available: false };
    try { m = await api(`/api/runs/${encodeURIComponent(name)}/metrology`); } catch (e) {}
    $("metro-card").classList.toggle("hidden", !m.available);
    if (!m.available) return;
    $("metro-n").textContent = `${m.measured.toLocaleString()} sites`;
    $("metro-body").innerHTML = m.structures.map(s => {
      const bad = Math.abs(s.bias_um) > 0.15 * s.nominal_um;
      return `<tr class="metro-row" data-x="${s.worst_at_px[0]}" data-y="${s.worst_at_px[1]}">
        <td><span class="metro-kind ${s.kind}">${s.kind === "gap" ? "Gap" : "Line"}</span></td>
        <td class="mono">${s.nominal_um} µm</td><td class="mono ${bad ? "metro-bad" : ""}">${s.mean_um.toFixed(2)}</td><td class="mono muted">${s.std_um.toFixed(2)}</td></tr>`;
    }).join("");
  }

  async function refreshWatch() {
    let ws = [];
    try { ws = await api("/api/watch"); } catch (e) { return; }
    $("watch-list").innerHTML = ws.map(w => `<div class="watch-item"><sl-icon name="broadcast"></sl-icon>
      <div class="watch-text"><b>Watching</b><span class="mono muted">${esc(w.path)}</span><span class="muted">${w.queued.length} image${w.queued.length === 1 ? "" : "s"} inspected</span></div>
      <sl-icon-button name="x-lg" label="Stop watching" class="watch-stop" data-path="${esc(w.path)}"></sl-icon-button></div>`).join("");
  }

  async function startWatch() {
    const path = $("watch-path").value.trim();
    if (!path) { notify("Enter the folder the microscope saves to.", "warning", "exclamation-triangle"); return; }
    const r = await fetch("/api/watch", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, pixel_size_um: $("watch-px").value || null, straighten: $("scan-straighten").checked, include_existing: $("watch-existing").checked }) });
    if (!r.ok) { const e = await r.json().catch(() => ({})); notify(e.detail || "Could not watch that folder.", "danger", "exclamation-octagon"); return; }
    $("add-dialog").hide();
    notify(`Watching ${path}: new images will be inspected automatically.`, "success", "broadcast");
    await refreshWatch(); pollJobs(true);
  }

  async function pollJobs(force) {
    clearTimeout(jobsTimer);
    let jobs = [];
    try { jobs = await api("/api/jobs"); } catch (e) { jobsTimer = setTimeout(pollJobs, 5000); return; }
    renderJobs(jobs);
    refreshWatch();
    for (const j of jobs) {
      const prev = lastJobs[j.id];
      if (prev && prev !== j.status) {
        if (j.status === "done") {
          await loadRunsKeep();
          notify(`“${j.name}” is ready.`, "success", "check2-circle", { label: "Open", run: () => openByName(j.name) });
        } else if (j.status === "failed") {
          notify(`“${j.name}” failed: ${j.error}`, "danger", "exclamation-octagon");
        }
      }
      lastJobs[j.id] = j.status;
    }
    const live = jobs.some(j => j.status === "queued" || j.status === "running");
    if (live || force) jobsTimer = setTimeout(pollJobs, live ? 1500 : 5000);
  }

  async function loadRunsKeep() {
    const cur = run && run.name;
    runs = await api("/api/runs");
    const sel = $("run");
    renderRunOptions(sel);
    $("welcome").classList.toggle("hidden", runs.length > 0);
    await sel.updateComplete;
    if (cur && runs.some(r => r.name === cur)) { sel.value = `r${runs.findIndex(r => r.name === cur)}`; run = runs.find(r => r.name === cur); }
    else if (runs.length) { await openByName(runs[0].name); }
  }

  async function openByName(name) {
    if (!runs.some(r => r.name === name)) runs = await api("/api/runs");
    const sel = $("run");
    sel.value = `r${runs.findIndex(r => r.name === name)}`;
    $("jobs-drawer").hide();
    await openRun(name);
  }

  // ---------------------------------------------------------------- keys & controls
  function typingTarget(e) {
    return e.composedPath().some(el => el.tagName && (["INPUT", "TEXTAREA", "SL-INPUT", "SL-SELECT", "SL-RANGE"].includes(el.tagName) || el.isContentEditable));
  }
  document.addEventListener("keydown", e => {
    if (e.metaKey || e.ctrlKey || e.altKey || typingTarget(e) || $("help").open || $("add-dialog").open || $("export-dialog").open) return;
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

  // ---------------------------------------------------------------- annotated image export
  const EX_HINT = {
    overview: "The whole scan shrunk to the width below, findings ringed in their risk colours. Good for slides and posters.",
    heatmap: "The scan in grey, coloured by how many findings there are per mm² (dark purple few, yellow most). Tick low-risk findings to map all dirt, not just the serious ones.",
    crop: "Exactly what is on screen now, at native resolution (a coarser level if it is very large).",
    tiff: "Every pixel, zoomable. Several GB; opens in QuPath, Fiji or PRECiV, not Preview. Takes a while.",
  };
  let exTimer = 0;
  function exKindChanged() {
    const k = $("ex-kind").value;
    $("ex-hint").textContent = EX_HINT[k] || "";
    $("ex-width").classList.toggle("hidden", k !== "overview" && k !== "heatmap");
  }
  async function pollExports() {
    clearTimeout(exTimer);
    let list = [];
    try { list = await api("/api/exports"); } catch (e) { return; }
    $("ex-list").innerHTML = list.map(j => {
      const label = { overview: "Overview", heatmap: "Heatmap", crop: "Current view", tiff: "Full TIFF" }[j.kind] || j.kind;
      const state = j.status === "done" ? `<a href="/api/exports/${encodeURIComponent(j.id)}/file" download>Download${j.size ? " " + fmtBytes(j.size) : ""}</a>`
        : j.status === "failed" ? `<span class="err">${esc(j.error || "failed")}</span>`
        : `${j.status === "queued" ? "Queued" : "Working"}${j.kind === "tiff" && j.progress ? ` ${Math.round(j.progress * 100)}%` : "…"}`;
      return `<div class="job"><div class="job-head"><b>${label}</b> · ${esc(j.run)}</div><div class="hint">${state}</div></div>`;
    }).join("");
    if (list.some(j => j.status === "queued" || j.status === "running")) exTimer = setTimeout(pollExports, 2000);
  }
  async function startExport() {
    if (!run || !viewer) return;
    const kind = $("ex-kind").value;
    const body = { kind, include_low: $("ex-low").checked, hide_false: $("ex-hide").checked, width: +$("ex-width").value };
    if (kind === "crop") {
      const el = viewer.container, a = toImage(new OpenSeadragon.Point(0, 0)), b = toImage(new OpenSeadragon.Point(el.clientWidth, el.clientHeight));
      body.bbox = [Math.floor(a.x), Math.floor(a.y), Math.ceil(b.x), Math.ceil(b.y)];
    }
    try {
      await api(`/api/runs/${encodeURIComponent(run.name)}/export-image`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (e) { fail(e); return; }
    pollExports();
  }

  function wire() {
    $("run").addEventListener("sl-change", () => { const v = $("run").value; if (v) openRun(runs[+v.slice(1)].name); });
    $("list").addEventListener("scroll", () => { cancelAnimationFrame(rowsFrame); rowsFrame = requestAnimationFrame(renderRows); }, { passive: true });
    $("list").addEventListener("click", e => { const row = e.target.closest(".row"); if (row) setActive(+row.dataset.i, true); });
    if (window.ResizeObserver) new ResizeObserver(() => renderRows()).observe($("list"));
    const onFilter = () => { active = -1; $("list").scrollTop = 0; refreshList(); renderInspector(); };
    $("class-chips").addEventListener("click", e => {
      const c = e.target.closest(".chip"); if (!c) return;
      const q = $("q"); q.value = (q.value || "").trim().toLowerCase() === c.dataset.cls ? "" : c.dataset.cls;
      onFilter();
    });
    $("group-filter").addEventListener("click", e => {
      const c = e.target.closest(".group-card"); if (!c) return;
      const g = c.dataset.group;
      if (e.shiftKey || e.metaKey) { for (const k in groupOn) groupOn[k] = k === g; } else groupOn[g] = !groupOn[g];
      if (!Object.values(groupOn).some(Boolean)) groupOn[g] = true;
      onFilter();
    });
    $("risk-filter").addEventListener("click", e => {
      const c = e.target.closest(".chip"); if (!c) return;
      const lv = c.dataset.level;
      levelOn[lv] = !levelOn[lv];
      if (!Object.values(levelOn).some(Boolean)) levelOn[lv] = true;
      onFilter();
    });
    $("q").addEventListener("sl-input", onFilter);
    $("q").addEventListener("sl-clear", onFilter);
    $("sort").addEventListener("sl-change", onFilter);

    $("btn-measure").addEventListener("click", () => setMeasuring(!measuring));
    $("measure-close").addEventListener("click", () => { $("card-measure").classList.add("hidden"); measurePts = []; safeDraw(); });
    $("btn-help").addEventListener("click", () => $("help").show());
    $("btn-positions").addEventListener("click", () => {
      if (!selected.size) { notify("Select findings for re-imaging with X first.", "warning", "exclamation-triangle"); return; }
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
    $("layer-heat").addEventListener("sl-change", e => {
      const on = e.target.checked;
      $("heat-legend").classList.toggle("hidden", !on);
      // 100k outlines bury the colour: hide them while the heatmap is on, bring them back after
      $("layer-defects").checked = !on;
      buildHeat(); safeDraw();
    });

    $("thr").tooltipFormatter = v => v.toFixed(2);
    $("thr-speck").tooltipFormatter = v => v.toFixed(2);
    $("thr-trace").tooltipFormatter = v => v.toFixed(2);
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
    $("btn-rethreshold").addEventListener("click", async () => {
      const btn = $("btn-rethreshold"); btn.loading = true;
      try {
        const q = `speck=${$("thr-speck").value}&trace=${$("thr-trace").value}&min_area_px=${parseInt($("minpx-scan").value, 10) || 1}`;
        const r = await api(`/api/runs/${encodeURIComponent(run.name)}/rethreshold?${q}`, { method: "POST" });
        defects = await api(`/api/runs/${encodeURIComponent(run.name)}/defects`);
        active = -1; refreshList(); refreshStats(); renderInspector();
        notify(`${r.n.toLocaleString()} findings at the new sensitivity.`, "success", "check2-circle");
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

    // add scan
    $("btn-add").addEventListener("click", openAdd);
    $("welcome-add").addEventListener("click", openAdd);
    $("add-cancel").addEventListener("click", () => $("add-dialog").hide());
    $("add-submit").addEventListener("click", submitScan);
    $("file-input").addEventListener("change", e => { if (e.target.files[0]) setFile(e.target.files[0]); });
    const drop = $("drop");
    ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });
    // dropping anywhere on the page opens the dialog with that file
    document.addEventListener("dragover", e => { if (!$("add-dialog").open && e.dataTransfer && [...e.dataTransfer.types].includes("Files")) e.preventDefault(); });
    document.addEventListener("drop", e => {
      if ($("add-dialog").open || !e.dataTransfer || !e.dataTransfer.files.length) return;
      e.preventDefault(); openAdd(); setFile(e.dataTransfer.files[0]);
    });
    $("gds-input").addEventListener("change", e => { const f = e.target.files[0]; if (f) { pickedGds = f; $("gds-name").textContent = `${f.name} · ${fmtBytes(f.size)}`; } });

    // jobs
    $("btn-jobs").addEventListener("click", () => { $("jobs-drawer").show(); pollJobs(true); });
    $("btn-export-image").addEventListener("click", () => { exKindChanged(); $("export-dialog").show(); pollExports(); });
    $("ex-kind").addEventListener("sl-change", exKindChanged);
    $("ex-go").addEventListener("click", startExport);
    $("ex-close").addEventListener("click", () => $("export-dialog").hide());
    $("jobs-list").addEventListener("click", e => { const b = e.target.closest(".job-open"); if (b) openByName(b.dataset.name); });

    $("reviewer").value = store.get("chipscan.reviewer") || "";
    $("reviewer").addEventListener("sl-change", e => store.set("chipscan.reviewer", e.target.value));
    $("watch-start").addEventListener("click", () => startWatch().catch(fail));
    $("metro-body").addEventListener("click", e => {
      const tr = e.target.closest(".metro-row"); if (!tr || !viewer) return;
      const x = +tr.dataset.x, y = +tr.dataset.y, s = 400;
      viewer.viewport.fitBounds(imageRect(x - s / 2, y - s / 2, s, s), true);
    });
    $("watch-list").addEventListener("click", async e => {
      const b = e.target.closest(".watch-stop"); if (!b) return;
      await fetch(`/api/watch?path=${encodeURIComponent(b.dataset.path)}`, { method: "DELETE" }); refreshWatch();
    });
  }

  const START_HASH = location.hash;
  const TAGS = ["sl-select", "sl-option", "sl-input", "sl-button", "sl-switch", "sl-range", "sl-dialog", "sl-drawer", "sl-progress-bar", "sl-icon-button", "sl-badge", "sl-details"];
  Promise.all(TAGS.map(t => customElements.whenDefined(t)))
    .then(() => { wire(); pollJobs(true); return loadRuns(); })
    .then(() => { if (START_HASH === "#add") openAdd(); else if (START_HASH === "#jobs") $("jobs-drawer").show(); })
    .catch(e => { document.body.innerHTML = `<pre style="padding:20px;color:#f26b6b">${esc(e.stack || e)}</pre>`; });
})();
