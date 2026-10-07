/* chipscan viewer front end: OpenSeadragon canvas + defect list + keyboard review. */
(() => {
  const COLORS = { anomaly: "#ff6a00", speck: "#8fd3ff", die2die: "#ff3fb3", short: "#ff4d4d", open: "#ff4dff", particle: "#ffa63d", protrusion: "#ffe14d", notch: "#4de3ff", void: "#4db8ff", missing_feature: "#c04dff", extra_feature: "#c04dff", unclassified: "#5ad17a" };
  const SEV = { anomaly: 3, speck: 2, die2die: 4, short: 5, open: 5, missing_feature: 4, extra_feature: 4, particle: 3, protrusion: 2, notch: 2, void: 1, unclassified: 0 };
  const CLASS_KEYS = { "1": "short", "2": "open", "3": "particle", "4": "protrusion", "5": "notch", "6": "void" };

  const $ = (id) => document.getElementById(id);
  let runs = [], run = null, defects = [], view = [], active = -1, selected = new Set();
  let viewer = null, layers = { design: null, pred: null };
  let measuring = false, measurePts = [];

  // ---------------------------------------------------------------- data
  async function api(path, opts) { const r = await fetch(path, opts); if (!r.ok) throw new Error(await r.text()); return r.json(); }

  async function loadRuns() {
    runs = await api("/api/runs");
    const sel = $("run");
    sel.innerHTML = runs.map(r => `<option value="${r.name}">${r.name} (${r.n_defects})</option>`).join("");
    sel.onchange = () => openRun(sel.value);
    const wanted = new URLSearchParams(location.search).get("run");
    openRun(wanted && runs.some(r => r.name === wanted) ? wanted : runs[0].name);
  }

  async function openRun(name) {
    run = runs.find(r => r.name === name);
    history.replaceState(null, "", `?run=${encodeURIComponent(name)}`);
    const reg = run.registration;
    $("runinfo").textContent = `${run.width}×${run.height} px · ${run.pixel_size_um.toFixed(3)} µm/px · ${run.segmenter} · rot ${reg.rotation_deg.toFixed(2)}° scale ${reg.scale.toFixed(4)}${reg.flipped ? " mirrored" : ""} · fine rms ${run.fine_rms_px != null ? run.fine_rms_px.toFixed(2) + " px" : "n/a"}`;
    $("btn-export").href = `/api/runs/${name}/export.csv`;
    defects = await api(`/api/runs/${name}/defects`);
    selected.clear(); active = -1;
    $("sens").classList.toggle("hidden", !run.anomaly_run);
    if (run.anomaly_run && run.extract_params) {
      $("thr").value = run.extract_params.threshold; $("thr-val").textContent = (+run.extract_params.threshold).toFixed(2);
      $("minpx").value = run.extract_params.min_area_px; $("specks").checked = run.extract_params.include_specks !== false; $("die2die").checked = run.extract_params.include_die2die !== false;
    }
    buildViewer();
    refreshList();
    refreshStats();
  }

  // ---------------------------------------------------------------- viewer
  function tileSource(kind, ext) {
    const n = run.levels;
    return {
      width: run.width, height: run.height, tileSize: 256, tileOverlap: 0, minLevel: 0, maxLevel: n - 1,
      getTileUrl: (level, x, y) => `/api/runs/${run.name}/${kind}/${n - 1 - level}/${x}/${y}.${ext}`,
    };
  }

  function buildViewer() {
    if (viewer) { viewer.destroy(); viewer = null; }
    viewer = OpenSeadragon({
      id: "viewer", prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
      tileSources: tileSource("tiles", "jpg"), showNavigator: true, navigatorPosition: "TOP_RIGHT",
      maxZoomPixelRatio: 8, minZoomImageRatio: 0.5, animationTime: 0.6, gestureSettingsMouse: { clickToZoom: false },
      crossOriginPolicy: false, imageSmoothingEnabled: false,
    });
    viewer.addHandler("open", () => {
      viewer.addTiledImage({ tileSource: tileSource("design", "png"), opacity: $("layer-design").checked ? 1 : 0, success: e => { layers.design = e.item; } });
      viewer.addTiledImage({ tileSource: tileSource("pred", "png"), opacity: $("layer-pred").checked ? 1 : 0, success: e => { layers.pred = e.item; } });
    });
    viewer.addHandler("update-viewport", safeDraw);
    viewer.addHandler("resize", safeDraw);
    // OSD's own canvas can lag behind a window/pane resize; force it to follow the container
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => { try { if (viewer.forceResize) viewer.forceResize(); viewer.forceRedraw(); } catch (e) {} safeDraw(); });
      ro.observe($("viewer-wrap"));
      viewer.addOnceHandler("close", () => ro.disconnect());
    }
    viewer.addHandler("canvas-click", onCanvasClick);
    new OpenSeadragon.MouseTracker({ element: viewer.container, moveHandler: onMove }).setTracking(true);
  }

  // use the base image's own converters: with several layered images the viewport-level ones warn
  function baseImage() { return viewer.world.getItemCount() ? viewer.world.getItemAt(0) : null; }
  function toImage(pt) { const b = baseImage(); return b ? b.viewerElementToImageCoordinates(pt) : viewer.viewport.viewerElementToImageCoordinates(pt); }
  function toScreen(x, y) { const b = baseImage(); const p = new OpenSeadragon.Point(x, y); return b ? b.imageToViewerElementCoordinates(p) : viewer.viewport.imageToViewerElementCoordinates(p); }
  function imageRect(x, y, w, h) { const b = baseImage(); return b ? b.imageToViewportRectangle(x, y, w, h) : viewer.viewport.imageToViewportRectangle(x, y, w, h); }

  function safeDraw() { try { drawOverlay(); } catch (e) { console.warn("overlay draw failed", e); } }

  function drawOverlay() {
    const cv = $("overlay"), wrap = $("viewer-wrap");
    cv.width = wrap.clientWidth; cv.height = wrap.clientHeight;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!viewer || !viewer.viewport) return;
    const zoom = viewer.viewport.getZoom(true) * viewer.viewport.getContainerSize().x / run.width; // screen px per image px
    if ($("layer-defects").checked) {
      for (let i = 0; i < defects.length; i++) {
        const d = defects[i], b = d.bbox_px; if (!b) continue;
        const p0 = toScreen(b.x0, b.y0), p1 = toScreen(b.x1, b.y1);
        if (p1.x < 0 || p1.y < 0 || p0.x > cv.width || p0.y > cv.height) continue;
        const col = COLORS[d.class] || COLORS.unclassified;
        const isActive = view[active] === d;
        ctx.lineWidth = isActive ? 2.5 : 1.2; ctx.strokeStyle = col;
        if (d.review.status === "false_alarm") ctx.setLineDash([3, 3]); else ctx.setLineDash([]);
        if (zoom > 1.5 && d.polygons_px.length) {
          for (const poly of d.polygons_px) {
            ctx.beginPath();
            poly.forEach((p, k) => { const s = toScreen(p[0], p[1]); k ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
            ctx.closePath(); ctx.stroke();
          }
        } else {
          const w = Math.max(p1.x - p0.x, 8), h = Math.max(p1.y - p0.y, 8);
          const cx = (p0.x + p1.x) / 2, cy = (p0.y + p1.y) / 2;
          ctx.strokeRect(cx - w / 2 - 3, cy - h / 2 - 3, w + 6, h + 6);
        }
        if (selected.has(d.id)) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p1.x + 6, p0.y - 6, 3, 0, 6.28); ctx.fill(); }
        if (isActive) { ctx.fillStyle = col; ctx.font = "12px sans-serif"; ctx.fillText(`#${d.id} ${d.class}`, p0.x, p0.y - 8); }
      }
      ctx.setLineDash([]);
    }
    if (measurePts.length) {
      ctx.strokeStyle = "#5ad17a"; ctx.lineWidth = 1.5; ctx.beginPath();
      measurePts.forEach((p, k) => { const s = toScreen(p.x, p.y); k ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
      ctx.stroke();
    }
  }

  function onMove(e) {
    if (!viewer || !run) return;
    const p = toImage(e.position);
    $("readout").textContent = `px ${p.x.toFixed(0)}, ${p.y.toFixed(0)}  ·  ${(p.x * run.pixel_size_um).toFixed(1)}, ${(p.y * run.pixel_size_um).toFixed(1)} µm (image frame)`;
  }

  async function onCanvasClick(e) {
    if (!e.quick) return;
    const p = toImage(e.position);
    if (measuring) {
      measurePts.push(p); drawOverlay();
      if (measurePts.length === 2) {
        const [a, b] = measurePts;
        const r = await api(`/api/runs/${run.name}/measure?x0=${a.x}&y0=${a.y}&x1=${b.x}&y1=${b.y}`);
        $("measure-out").classList.remove("hidden");
        $("measure-out").innerHTML = `<b>cut ${r.length_um.toFixed(1)} µm</b><br>` + r.runs.map(x => `${x.phase.padEnd(5)} ${x.length_um.toFixed(2)} µm`).join("<br>");
        measuring = false; $("btn-measure").classList.remove("active");
      }
      return;
    }
    // click on a defect box selects it
    let best = -1, bestD = 1e9;
    view.forEach((d, i) => { const b = d.bbox_px; if (!b) return; const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2; const dd = Math.hypot(cx - p.x, cy - p.y); const r = Math.max(b.x1 - b.x0, b.y1 - b.y0) / 2 + 12 / (viewer.viewport.getZoom(true) * viewer.viewport.getContainerSize().x / run.width); if (dd < r && dd < bestD) { best = i; bestD = dd; } });
    if (best >= 0) setActive(best, false);
  }

  // ---------------------------------------------------------------- list
  function parseFilter(q) {
    const toks = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return d => toks.every(t => {
      if (t.startsWith(">")) return d.area_um2 > parseFloat(t.slice(1));
      if (t.startsWith("<")) return d.area_um2 < parseFloat(t.slice(1));
      if (t.startsWith("#")) return String(d.id) === t.slice(1);
      return d.class.includes(t) || d.kind.includes(t) || d.review.status.includes(t) || (t === "selected" && selected.has(d.id));
    });
  }

  function refreshList() {
    const f = parseFilter($("q").value), sort = $("sort").value;
    view = defects.filter(f);
    const key = { severity: d => -(SEV[d.class] || 0) * 1e6 - d.area_um2, area: d => -d.area_um2, score: d => -d.score, id: d => d.id }[sort];
    view.sort((a, b) => key(a) - key(b));
    $("list").innerHTML = view.map((d, i) => `
      <div class="row ${i === active ? "active" : ""}" data-i="${i}">
        <img loading="lazy" src="/api/runs/${run.name}/thumb/${d.id}.jpg" alt="">
        <div><div class="title cls-${d.class}">${d.class} <span class="muted">#${d.id}</span>${selected.has(d.id) ? '<span class="sel">●</span>' : ""}</div>
          <div class="sub">${d.kind.replace("_metal", "")} · ${d.area_um2.toFixed(1)} µm² · score ${d.score.toFixed(2)}${d.repeat_ncc != null ? ` · twin ${d.repeat_ncc.toFixed(2)}` : ""}${d.review.cls && d.review.cls !== d.class ? ` · reviewed as <b>${d.review.cls}</b>` : ""}</div></div>
        <span class="badge ${d.review.status}">${d.review.status.replace("_", " ")}</span>
      </div>`).join("");
    $("list").querySelectorAll(".row").forEach(el => el.onclick = () => setActive(+el.dataset.i, true));
    drawOverlay();
  }

  function refreshStats() {
    api(`/api/runs/${run.name}/stats`).then(s => {
      const parts = Object.entries(s.by_class).sort().map(([c, v]) => `<span class="cls-${c}">${c}</span> <b>${v.n}</b>${v.real + v.false_alarm ? ` <span class="muted">(${v.real}✓ ${v.false_alarm}✗)</span>` : ""}`);
      const reviewed = s.n_reviewed, real = Object.values(s.by_class).reduce((a, v) => a + v.real, 0), fa = Object.values(s.by_class).reduce((a, v) => a + v.false_alarm, 0);
      parts.unshift(`<b>${s.n}</b> defects · <b>${reviewed}</b> reviewed${real + fa ? ` · precision so far <b>${(100 * real / (real + fa)).toFixed(0)}%</b>` : ""}`);
      $("stats").innerHTML = parts.join(" · ");
    });
  }

  function setActive(i, fly) {
    active = i;
    $("list").querySelectorAll(".row").forEach((el, k) => el.classList.toggle("active", k === i));
    const el = $("list").querySelector(`.row[data-i="${i}"]`); if (el) el.scrollIntoView({ block: "nearest" });
    const d = view[i];
    if (d && fly !== false && d.bbox_px) {
      const b = d.bbox_px, size = Math.max(b.x1 - b.x0, b.y1 - b.y0) * 6 + 80;
      const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
      viewer.viewport.fitBounds(imageRect(cx - size / 2, cy - size / 2, size, size), false);
    }
    drawOverlay();
  }

  async function setReview(status, cls) {
    const d = view[active]; if (!d) return;
    const body = { defect_id: d.id, status, cls: cls || d.review.cls || null, reviewer: $("reviewer").value || null };
    if (cls && status === undefined) body.status = d.review.status === "unreviewed" ? "real" : d.review.status;
    await api(`/api/runs/${run.name}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    d.review = { status: body.status, cls: body.cls, note: d.review.note };
    const keep = active; refreshList(); refreshStats(); active = keep; setActive(keep, false);
  }

  // ---------------------------------------------------------------- keys & buttons
  document.addEventListener("keydown", e => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); if (active < view.length - 1) setActive(active + 1, true); }
    else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); if (active > 0) setActive(active - 1, true); }
    else if (e.key === "r") setReview("real");
    else if (e.key === "f") setReview("false_alarm");
    else if (e.key === "u") setReview("unsure");
    else if (e.key === "x") { const d = view[active]; if (d) { selected.has(d.id) ? selected.delete(d.id) : selected.add(d.id); refreshList(); setActive(active, false); } }
    else if (CLASS_KEYS[e.key]) setReview(undefined, CLASS_KEYS[e.key]);
    else if (e.key === "m") toggleMeasure();
    else if (e.key === "Escape") { measuring = false; measurePts = []; $("btn-measure").classList.remove("active"); $("measure-out").classList.add("hidden"); drawOverlay(); }
  });
  function toggleMeasure() { measuring = !measuring; measurePts = []; $("btn-measure").classList.toggle("active", measuring); if (!measuring) $("measure-out").classList.add("hidden"); safeDraw(); }
  $("btn-measure").onclick = toggleMeasure;
  $("thr").oninput = e => { $("thr-val").textContent = (+e.target.value).toFixed(2); };
  $("btn-apply").onclick = async () => {
    $("btn-apply").disabled = true; $("btn-apply").textContent = "working…";
    try {
      const q = `threshold=${$("thr").value}&min_area_px=${$("minpx").value}&merge_px=8&include_specks=${$("specks").checked}&include_die2die=${$("die2die").checked}`;
      const r = await api(`/api/runs/${run.name}/reextract?${q}`, { method: "POST" });
      defects = await api(`/api/runs/${run.name}/defects`); active = -1; refreshList(); refreshStats();
      $("btn-apply").textContent = `apply (${r.n})`;
    } catch (e) { alert(e); $("btn-apply").textContent = "apply"; }
    $("btn-apply").disabled = false;
  };
  $("btn-positions").onclick = () => { const ids = [...selected].join(","); if (!ids) { alert("select defects with x first"); return; } location.href = `/api/runs/${run.name}/positions.xml?ids=${ids}`; };
  $("layer-design").onchange = e => { if (layers.design) layers.design.setOpacity(e.target.checked ? 1 : 0); };
  $("layer-pred").onchange = e => { if (layers.pred) layers.pred.setOpacity(e.target.checked ? 1 : 0); };
  $("layer-defects").onchange = drawOverlay;
  $("q").oninput = () => { active = -1; refreshList(); };
  $("sort").onchange = () => { active = -1; refreshList(); };
  $("reviewer").value = localStorage.getItem("chipscan.reviewer") || "";
  $("reviewer").onchange = e => localStorage.setItem("chipscan.reviewer", e.target.value);

  loadRuns().catch(e => { document.body.innerHTML = `<pre style="padding:20px">${e}</pre>`; });
})();
