// ============================================================
// NIC Fraud Review Console — app.js
// Talks to the API at API_BASE (same-origin, via nginx). Endpoint contracts
// verified against src/api/handlers/*.py + src/api/schemas.py.
// ============================================================

let selectedAppId = null;
let topoKind = "ring";          // "ring" | "topology" — current modal view

// ---------- small utilities ----------

function toast(message, kind = "") {
  const stack = document.getElementById("toast-stack");
  const el = document.createElement("div");
  el.className = `toast ${kind}`.trim();
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

async function apiGetText(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error(`${res.status}`);
  }
  return res.text();
}

async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(payload.detail || `${res.status}`);
  }
  return payload;
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// Auto-size an iframe to its content height. srcdoc iframes inherit this page's
// origin, so reading contentDocument is allowed. Called on load and re-run a
// couple of times because the card's canvas/tabs settle after first paint.
function autosizeFrame(frame) {
  const measure = () => {
    try {
      const doc = frame.contentWindow.document;
      const h = Math.max(
        doc.body.scrollHeight, doc.documentElement.scrollHeight,
        doc.body.offsetHeight, doc.documentElement.offsetHeight,
      );
      if (h > 0) frame.style.height = h + 24 + "px";
    } catch (e) { /* cross-origin fallback: keep min-height from CSS */ }
  };
  measure();
  setTimeout(measure, 120);
  setTimeout(measure, 500);
}

// ---------- tab switching ----------

document.querySelectorAll("nav.tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav.tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "patterns") loadPatterns();
    if (btn.dataset.view === "admin") loadAdminConsole();
  });
});

// ---------- stat tiles ----------

async function loadStats() {
  try {
    const summary = await apiGet("/v3/monitoring/fraud-store-summary");
    document.getElementById("tile-confirmed").textContent = summary.n_confirmed;
    document.getElementById("tile-fp").textContent = summary.n_false_positives;
  } catch (e) {
    document.getElementById("tile-confirmed").textContent = "err";
    document.getElementById("tile-fp").textContent = "err";
  }

  try {
    const ckpt = await apiGet("/v3/model/checkpoint-info");
    document.getElementById("tile-checkpoint").textContent = ckpt.exists
      ? `${ckpt.size_mb?.toFixed(1) ?? "?"} MB`
      : "missing";
  } catch (e) {
    document.getElementById("tile-checkpoint").textContent = "err";
  }

  try {
    const drift = await apiGet("/v3/monitoring/drift");
    const tile = document.getElementById("tile-drift-wrap");
    const val = document.getElementById("tile-drift");
    val.textContent = drift.recommendation;
    tile.className = "tile " + (drift.drift_detected ? "warn" : "ok");
  } catch (e) {
    document.getElementById("tile-drift").textContent = "n/a";
  }
}

// ---------- review queue ----------

// Queue state. dismissedIds persists for the session so "removed" rows stay
// removed across filter/refresh (client-side triage; does NOT mutate server data).
let queueRows = [];
let queueIdCol = "application_id";
let queueScoreCol = "risk_score_v3";
const dismissedIds = new Set();
const selectedIds = new Set();

function riskBucket(v) {
  const n = Number(v);
  return n >= 0.66 ? "high" : n >= 0.33 ? "med" : "low";
}
function riskLabel(b) { return b === "high" ? "High" : b === "med" ? "Medium" : "Low"; }

async function loadQueue() {
  const body = document.getElementById("queue-body");
  body.innerHTML = `<div class="empty-state">Loading...</div>`;
  try {
    const rows = await apiGet("/v3/monitoring/top-suspicious?n=50");
    if (!rows.length) {
      body.innerHTML = `<div class="empty-state">No suspicious applications in the last run.</div>`;
      document.getElementById("queue-toolbar").style.display = "none";
      return;
    }
    const cols = Object.keys(rows[0]);
    queueIdCol = cols.includes("application_id") ? "application_id" : cols[0];
    queueScoreCol = cols.find((c) => c.toLowerCase().includes("score")) || cols[1];
    queueRows = rows;
    selectedIds.clear();
    document.getElementById("queue-toolbar").style.display = "flex";
    renderQueue();
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Could not load queue: ${escapeHtml(e.message)}</div>`;
  }
}

function renderQueue() {
  const body = document.getElementById("queue-body");
  const q = (document.getElementById("queue-search").value || "").trim().toLowerCase();
  const rf = document.getElementById("queue-risk-filter").value;

  const visible = queueRows.filter((row) => {
    const id = String(row[queueIdCol]);
    if (dismissedIds.has(id)) return false;
    if (q && !id.toLowerCase().includes(q)) return false;
    const b = riskBucket(row[queueScoreCol]);
    if (rf === "high" && b !== "high") return false;
    if (rf === "med" && b === "low") return false;
    return true;
  });

  if (!visible.length) {
    body.innerHTML = `<div class="empty-state">No applications match the current filter${dismissedIds.size ? " (some removed)" : ""}.</div>`;
    updateSelectionUI();
    return;
  }

  let html = "<table><thead><tr><th style='width:34px'></th><th>Risk</th><th>Application ID</th><th>Score</th><th></th></tr></thead><tbody>";
  for (const row of visible) {
    const id = String(row[queueIdCol]);
    const scoreNum = Number(row[queueScoreCol]);
    const score = Number.isFinite(scoreNum) ? scoreNum.toFixed(4) : row[queueScoreCol];
    const b = riskBucket(row[queueScoreCol]);
    const checked = selectedIds.has(id) ? "checked" : "";
    html += `<tr data-app-id="${escapeHtml(id)}">
      <td class="qcell-chk"><input type="checkbox" class="row-chk" ${checked}></td>
      <td><span class="risk-badge risk-${b}">${riskLabel(b)}</span></td>
      <td class="qcell-id">${escapeHtml(id)}</td>
      <td class="num">${escapeHtml(score)}</td>
      <td style="text-align:right"><button class="ghost open-row">Open &rarr;</button></td>
    </tr>`;
  }
  html += "</tbody></table>";
  body.innerHTML = html;

  body.querySelectorAll("tr[data-app-id]").forEach((tr) => {
    const id = tr.dataset.appId;
    const chk = tr.querySelector(".row-chk");
    chk.addEventListener("click", (e) => {
      e.stopPropagation();
      if (chk.checked) selectedIds.add(id); else selectedIds.delete(id);
      updateSelectionUI();
    });
    tr.addEventListener("click", (e) => {
      if (e.target.closest(".qcell-chk")) return;
      selectApp(id, tr);
    });
  });
  updateSelectionUI();
}

function updateSelectionUI() {
  const n = selectedIds.size;
  document.getElementById("qt-count").textContent = `${n} selected`;
  document.getElementById("btn-remove-selected").disabled = n === 0;
  document.getElementById("btn-export-selected").disabled = n === 0;
  document.getElementById("btn-label-selected").disabled = n === 0;
  document.getElementById("btn-restore-removed").style.display = dismissedIds.size ? "" : "none";
  const all = document.getElementById("chk-select-all");
  const chks = document.querySelectorAll("#queue-body .row-chk");
  all.checked = chks.length > 0 && [...chks].every((c) => c.checked);
}

// Open one application's reviewer card in the detail panel below the queue.
// (Distinct from the multi-select checkboxes: this is the single-row "Open"
// action — clicking a row, or its "Open →" button.)
async function selectApp(appId, rowEl) {
  selectedAppId = appId;
  document.querySelectorAll("#queue-body tr").forEach((r) => r.classList.remove("row-selected"));
  if (rowEl) rowEl.classList.add("row-selected");

  document.getElementById("detail-section").style.display = "block";
  document.getElementById("detail-app-id").textContent = appId;
  document.getElementById("detail-section").scrollIntoView({ behavior: "smooth", block: "start" });

  const frame = document.getElementById("card-frame");
  frame.style.height = "420px";
  frame.onload = () => autosizeFrame(frame);
  frame.srcdoc = "<p style='font-family:sans-serif;padding:14px;color:#c9d1d9;background:#0d1117;margin:0;'>Loading card…</p>";
  try {
    const html = await apiGetText(`/v3/monitoring/${encodeURIComponent(appId)}/card`);
    frame.srcdoc = html ?? "<p style='font-family:sans-serif;padding:14px;color:#c9d1d9;background:#0d1117;margin:0;'>No card for this application.</p>";
  } catch (e) {
    frame.srcdoc = `<p style='font-family:sans-serif;padding:14px;color:#ffb3b5;background:#0d1117;margin:0;'>Failed to load card: ${escapeHtml(e.message)}</p>`;
  }
}

document.getElementById("btn-refresh-queue").addEventListener("click", loadQueue);
document.getElementById("queue-search").addEventListener("input", renderQueue);
document.getElementById("queue-risk-filter").addEventListener("change", renderQueue);

document.getElementById("chk-select-all").addEventListener("change", (e) => {
  const on = e.target.checked;
  document.querySelectorAll("#queue-body tr[data-app-id]").forEach((tr) => {
    const id = tr.dataset.appId;
    if (on) selectedIds.add(id); else selectedIds.delete(id);
    const chk = tr.querySelector(".row-chk");
    if (chk) chk.checked = on;
  });
  updateSelectionUI();
});

document.getElementById("btn-remove-selected").addEventListener("click", () => {
  if (!selectedIds.size) return;
  const n = selectedIds.size;
  selectedIds.forEach((id) => dismissedIds.add(id));
  selectedIds.clear();
  if (selectedAppId && dismissedIds.has(selectedAppId)) {
    document.getElementById("detail-section").style.display = "none";
    selectedAppId = null;
  }
  renderQueue();
  toast(`Removed ${n} from the list (this session - server data untouched).`);
});

document.getElementById("btn-restore-removed").addEventListener("click", () => {
  const n = dismissedIds.size;
  dismissedIds.clear();
  renderQueue();
  toast(`Restored ${n} removed application(s).`);
});

// ---------- export ----------
// The endpoints return a zip with Content-Disposition: attachment, so pointing
// a transient anchor at them triggers a download without navigating away.
function downloadUrl(path) {
  const a = document.createElement("a");
  a.href = API_BASE + path;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

document.getElementById("btn-export-app").addEventListener("click", () => {
  if (!selectedAppId) { toast("Select an application first.", "error"); return; }
  downloadUrl(`/v3/monitoring/${encodeURIComponent(selectedAppId)}/export`);
  toast(`Preparing export for ${selectedAppId}…`);
});

document.getElementById("btn-export-bulk").addEventListener("click", () => {
  toast("Bundling all flagged applications — this can take a moment…");
  downloadUrl(`/v3/monitoring/export/bulk`);
});

document.getElementById("btn-export-selected").addEventListener("click", () => {
  if (!selectedIds.size) { toast("Select one or more applications first.", "error"); return; }
  const ids = [...selectedIds].map(encodeURIComponent).join(",");
  downloadUrl(`/v3/monitoring/export/selected?ids=${ids}`);
  toast(`Bundling ${selectedIds.size} selected application(s) — card, ring, evidence…`);
});

// ---------- batch label & retrain (multi-select → confirmed_fraud_store) ----------
// Each selected application gets a verdict (confirmed fraud + type, or false
// positive); the batch POSTs to /v3/supervisor/confirm-batch. Retrain is only
// dispatched on the explicit "Record + retrain" action (human-gated).

const batchModal = document.getElementById("batch-modal");

function batchRowHtml(id) {
  const opts = VALID_FRAUD_TYPES.map((t) => `<option value="${t}">${t}</option>`).join("");
  return `<div class="batch-row verdict-fraud" data-app-id="${escapeHtml(id)}">
    <span class="brow-id" title="${escapeHtml(id)}">${escapeHtml(id)}</span>
    <select class="brow-verdict">
      <option value="confirmed_fraud">confirmed fraud</option>
      <option value="false_positive">false positive</option>
    </select>
    <select class="brow-ftype">${opts}</select>
    <input class="brow-notes" type="text" placeholder="notes (optional)">
  </div>`;
}

function syncBatchRow(row) {
  const fraud = row.querySelector(".brow-verdict").value === "confirmed_fraud";
  row.querySelector(".brow-ftype").hidden = !fraud;
  row.classList.toggle("verdict-fraud", fraud);
  row.classList.toggle("verdict-fp", !fraud);
}

function openBatchModal() {
  if (!selectedIds.size) { toast("Select one or more applications first.", "error"); return; }
  const list = document.getElementById("batch-list");
  list.innerHTML = [...selectedIds].map(batchRowHtml).join("");
  document.getElementById("batch-count").textContent = selectedIds.size;
  document.getElementById("batch-result").style.display = "none";
  list.querySelectorAll(".batch-row").forEach((row) => {
    row.querySelector(".brow-verdict").addEventListener("change", () => syncBatchRow(row));
    syncBatchRow(row);
  });
  batchModal.classList.add("open");
}

function closeBatch() { batchModal.classList.remove("open"); }

function gatherBatchItems() {
  return [...document.querySelectorAll("#batch-list .batch-row")].map((row) => {
    const verdict = row.querySelector(".brow-verdict").value;
    const item = {
      application_id: row.dataset.appId,
      verdict,
      notes: row.querySelector(".brow-notes").value.trim(),
    };
    if (verdict === "confirmed_fraud") item.fraud_type = row.querySelector(".brow-ftype").value;
    return item;
  });
}

async function submitBatch(dispatch) {
  const by = document.getElementById("batch-by").value.trim();
  if (!by) { toast("Enter a reviewer name.", "error"); return; }
  const items = gatherBatchItems();
  const resultEl = document.getElementById("batch-result");
  resultEl.style.display = "block";
  resultEl.textContent = dispatch ? "Recording labels + dispatching retrain…" : "Recording labels…";
  const btnR = document.getElementById("btn-batch-record");
  const btnT = document.getElementById("btn-batch-retrain");
  btnR.disabled = btnT.disabled = true;
  try {
    const res = await apiPost("/v3/supervisor/confirm-batch", {
      items,
      confirmed_by: by,
      cycle: document.getElementById("batch-cycle").value.trim(),
      dispatch_retrain: dispatch,
      smoke_test: document.getElementById("batch-smoke").checked,
    });
    let msg = `Recorded ${res.n_recorded} label(s). Store now: ${res.n_confirmed} confirmed, ${res.n_false_positives} false-positive.`;
    if (res.errors && res.errors.length) {
      msg += `\nSkipped ${res.errors.length}:\n` + res.errors.map((e) => `  ${e.application_id}: ${e.error}`).join("\n");
    }
    resultEl.textContent = msg;
    if (res.retrain_dispatched) {
      const pj = document.getElementById("poll-job-id");
      if (pj) pj.value = res.job_id;
      pollJob(res.job_id, resultEl);
    }
    toast(res.retrain_dispatched ? "Labels recorded — retrain dispatched." : `Recorded ${res.n_recorded} label(s).`, "success");
    loadStats();
  } catch (e) {
    resultEl.textContent = `Failed: ${e.message}`;
    toast(`Failed: ${e.message}`, "error");
  } finally {
    btnR.disabled = btnT.disabled = false;
  }
}

document.getElementById("btn-label-selected").addEventListener("click", openBatchModal);
document.getElementById("btn-batch-close").addEventListener("click", closeBatch);
batchModal.addEventListener("click", (e) => { if (e.target === batchModal) closeBatch(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeBatch(); });
document.getElementById("btn-batch-record").addEventListener("click", () => submitBatch(false));
document.getElementById("btn-batch-retrain").addEventListener("click", () => submitBatch(true));
document.querySelectorAll(".batch-bulkset [data-bulk]").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#batch-list .batch-row").forEach((row) => {
      row.querySelector(".brow-verdict").value = b.dataset.bulk;
      syncBatchRow(row);
    });
  });
});

// ---------- topology detail modal ----------

const topoModal = document.getElementById("topo-modal");

function setSeg(kind) {
  topoKind = kind;
  document.getElementById("seg-ring").classList.toggle("on", kind === "ring");
  document.getElementById("seg-ego").classList.toggle("on", kind === "topology");
  document.getElementById("topo-title").firstChild.textContent =
    kind === "ring" ? "3D identity ring · " : "Ego-graph topology · ";
}

async function loadTopoFrame() {
  if (!selectedAppId) return;
  const frame = document.getElementById("topo-frame");
  frame.srcdoc = "<p style='font-family:sans-serif;padding:16px;color:#c9d1d9;background:#0d1117;margin:0;'>Rendering…</p>";
  try {
    const html = await apiGetText(`/v3/monitoring/${encodeURIComponent(selectedAppId)}/${topoKind}`);
    frame.srcdoc = html ?? "<p style='font-family:sans-serif;padding:16px;color:#c9d1d9;background:#0d1117;margin:0;'>No typed edges for this application — nothing to draw.</p>";
  } catch (e) {
    frame.srcdoc = `<p style='font-family:sans-serif;padding:16px;color:#ffb3b5;background:#0d1117;margin:0;'>Failed: ${escapeHtml(e.message)}</p>`;
  }
}

function openTopo(kind) {
  if (!selectedAppId) return;
  document.getElementById("topo-app-id").textContent = selectedAppId;
  setSeg(kind);
  topoModal.classList.add("open");
  loadTopoFrame();
}

document.getElementById("btn-open-ring").addEventListener("click", () => openTopo("ring"));
document.getElementById("btn-open-topology").addEventListener("click", () => openTopo("topology"));
document.getElementById("seg-ring").addEventListener("click", () => { setSeg("ring"); loadTopoFrame(); });
document.getElementById("seg-ego").addEventListener("click", () => { setSeg("topology"); loadTopoFrame(); });
document.getElementById("btn-topo-close").addEventListener("click", () => topoModal.classList.remove("open"));
topoModal.addEventListener("click", (e) => { if (e.target === topoModal) topoModal.classList.remove("open"); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") topoModal.classList.remove("open"); });
document.getElementById("btn-topo-popout").addEventListener("click", () => {
  if (!selectedAppId) return;
  window.open(`${API_BASE}/v3/monitoring/${encodeURIComponent(selectedAppId)}/${topoKind}`, "_blank");
});

// ---------- supervisor actions ----------
// confirm-fraud / mark-false-positive / clear-label are handled INSIDE the
// embedded reviewer card (its own form posts to /v3/supervisor/* directly), so
// the console doesn't duplicate them. "Flag for LOE" (patterns/confirm) is the
// one supervisor action the card has no control for, so it lives here — it's the
// entry point that populates the Pattern queue. subgraph is built from explicit
// prompts, not guessed (see deploy/README.md "Open items" — subgraph schema).

// Must match VALID_FRAUD_TYPES in src/confirmed_fraud_graph_store.py — the
// pattern store rejects anything else with a 422.
const VALID_FRAUD_TYPES = ["IP_CLUSTER", "FEE_INFLATION", "INCOME_VIOLATION", "NAME_COLLISION", "CROSS_CHANNEL", "OTHER"];
let loeType = "IP_CLUSTER";

const loeModal = document.getElementById("loe-modal");

// Build the clickable fraud-type chips once.
(function initLoeChips() {
  const wrap = document.getElementById("loe-type-chips");
  wrap.innerHTML = VALID_FRAUD_TYPES.map((t) =>
    `<button type="button" class="chip${t === loeType ? " on" : ""}" data-type="${t}">${t.replace(/_/g, " ")}</button>`
  ).join("");
  wrap.querySelectorAll(".chip").forEach((c) => {
    c.addEventListener("click", () => {
      loeType = c.dataset.type;
      wrap.querySelectorAll(".chip").forEach((x) => x.classList.toggle("on", x === c));
      // "OTHER" leans on the free-text description, so nudge focus there.
      if (loeType === "OTHER") document.getElementById("loe-notes").focus();
    });
  });
})();

function openLoe() {
  if (!selectedAppId) return;
  document.getElementById("loe-app-id").textContent = selectedAppId;
  document.getElementById("loe-nodes").value = selectedAppId;
  document.getElementById("loe-notes").value = "";
  document.getElementById("loe-by").value = "";
  document.getElementById("loe-edge").value = "shares_ip";
  loeType = "IP_CLUSTER";
  document.querySelectorAll("#loe-type-chips .chip").forEach((c) =>
    c.classList.toggle("on", c.dataset.type === "IP_CLUSTER"));
  loeModal.classList.add("open");
}

function closeLoe() { loeModal.classList.remove("open"); }

document.getElementById("btn-flag-loe").addEventListener("click", openLoe);
document.getElementById("btn-loe-close").addEventListener("click", closeLoe);
document.getElementById("btn-loe-cancel").addEventListener("click", closeLoe);
loeModal.addEventListener("click", (e) => { if (e.target === loeModal) closeLoe(); });

document.getElementById("btn-loe-submit").addEventListener("click", async () => {
  if (!selectedAppId) return;
  const notes = document.getElementById("loe-notes").value.trim();
  const nodesRaw = document.getElementById("loe-nodes").value.trim();
  const edgeType = document.getElementById("loe-edge").value;
  const confirmed_by = document.getElementById("loe-by").value.trim();

  if (!confirmed_by) { toast("Enter your reviewer name.", "error"); return; }
  if (!nodesRaw) { toast("List at least this application's ID.", "error"); return; }
  if (loeType === "OTHER" && !notes) { toast("Describe the pattern when the type is OTHER.", "error"); return; }

  const subgraph = {
    nodes: nodesRaw.split(",").map((s) => s.trim()).filter(Boolean),
    edges: [{ type: edgeType }],
  };

  try {
    const res = await apiPost("/v3/supervisor/patterns/confirm", {
      application_id: selectedAppId, fraud_type: loeType, subgraph, confirmed_by, notes,
    });
    toast(`Pattern recorded: ${res.pattern_id}. See "Pattern queue" to promote it.`, "success");
    closeLoe();
  } catch (e) {
    toast(`Failed: ${e.message}`, "error");
  }
});

// ---------- pattern queue (LOE) ----------

async function loadPatterns() {
  const body = document.getElementById("patterns-body");
  body.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const data = await apiGet("/v3/supervisor/patterns");
    if (!data.patterns.length) {
      body.innerHTML = `<div class="empty-state">No pending patterns. Flag one from a reviewer card first.</div>`;
      return;
    }
    body.innerHTML = data.patterns.map((p) => `
      <div class="pattern-item">
        <input type="checkbox" class="pattern-check" value="${escapeHtml(p.pattern_id)}">
        <div style="flex:1;">
          <strong>${escapeHtml(p.pattern_id)}</strong>
          — ${escapeHtml(p.fraud_type ?? "")}
          <span class="badge amber">pending</span>
          <pre>${escapeHtml(JSON.stringify(p.subgraph ?? {}, null, 0))}</pre>
        </div>
      </div>
    `).join("");
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Could not load patterns: ${escapeHtml(e.message)}</div>`;
  }
}

document.getElementById("btn-refresh-patterns").addEventListener("click", loadPatterns);

document.getElementById("btn-promote").addEventListener("click", async () => {
  const ids = Array.from(document.querySelectorAll(".pattern-check:checked")).map((c) => c.value);
  if (!ids.length) { toast("Select at least one pattern.", "error"); return; }
  const smoke_test = document.getElementById("chk-smoke-test").checked;
  const statusEl = document.getElementById("promote-job-status");
  statusEl.style.display = "block";
  statusEl.textContent = "Dispatching…";
  try {
    const res = await apiPost("/v3/supervisor/patterns/promote", { pattern_ids: ids, smoke_test });
    statusEl.textContent = `${res.message} job_id=${res.job_id}`;
    toast("Retrain dispatched.", "success");
    pollJob(res.job_id, statusEl);
    loadPatterns();
  } catch (e) {
    statusEl.textContent = `Failed: ${e.message}`;
    toast(`Failed: ${e.message}`, "error");
  }
});

async function pollJob(jobId, statusEl, intervalMs = 3000, maxTries = 200) {
  let tries = 0;
  const tick = async () => {
    tries += 1;
    try {
      const job = await apiGet(`/v3/training/jobs/${encodeURIComponent(jobId)}`);
      statusEl.textContent = `job_id=${jobId}  status=${job.status}` +
        (job.error ? `  error=${job.error}` : "");
      if (job.status === "complete" || job.status === "failed" || tries >= maxTries) return;
      setTimeout(tick, intervalMs);
    } catch (e) {
      statusEl.textContent = `job_id=${jobId}  poll failed: ${e.message}`;
    }
  };
  tick();
}

// ---------- admin / training panel ----------

function loadAdminConsole() {
  loadModelStats();
  loadHistory();
  loadDriftExplain();
}

// ---- drift explanation (plain-English full-retrain rationale) ----
async function loadDriftExplain() {
  const el = document.getElementById("drift-explain-body");
  el.innerHTML = `<div class="empty-state">Loading...</div>`;
  try {
    const d = await apiGet("/v3/monitoring/drift-explain?top=12");
    const drift = d.drift_detected;
    const recCls = drift ? "high" : "ok";
    const recWord = escapeHtml(d.recommendation);

    let feats = "";
    if (d.top_features && d.top_features.length) {
      const rows = d.top_features.map((f) => {
        const shift = (f.mean_shift === null || f.mean_shift === undefined)
          ? "-" : (f.mean_shift > 0 ? "+" : "") + Number(f.mean_shift).toFixed(4);
        const badge = f.drifted
          ? `<span class="risk-badge risk-high">drifted</span>`
          : `<span class="risk-badge risk-low">stable</span>`;
        return `<tr>
          <td class="qcell-id">${escapeHtml(f.feature)}</td>
          <td>${escapeHtml(f.direction)}</td>
          <td class="num">${escapeHtml(shift)}</td>
          <td class="num">${escapeHtml(String(f.ks_stat))}</td>
          <td class="num">${escapeHtml(String(f.p_value))}</td>
          <td>${badge}</td>
        </tr>`;
      }).join("");
      feats = `<table style="margin-top:12px;">
        <thead><tr><th>Feature</th><th>Direction</th><th>Mean shift</th><th>KS</th><th>p-value</th><th></th></tr></thead>
        <tbody>${rows}</tbody></table>`;
    }

    el.innerHTML = `
      <div class="drift-verdict drift-${recCls}">
        <div class="dv-rec">Recommendation: <strong>${recWord}</strong></div>
        <div class="dv-text">${escapeHtml(d.overall_explanation)}</div>
        <div class="dv-nums">
          <span>KS p-value <span class="num">${escapeHtml(String(d.p_value))}</span></span>
          <span>alert threshold <span class="num">${escapeHtml(String(d.ks_threshold))}</span></span>
          <span>${escapeHtml(String(d.n_drifted))} / ${escapeHtml(String(d.n_features))} features drifted</span>
        </div>
      </div>
      <p style="font-size:12.5px;color:var(--fg2);margin:14px 2px 0;">${escapeHtml(d.feature_summary)}</p>
      ${feats}`;
  } catch (e) {
    el.innerHTML = `<div class="empty-state">Could not load drift explanation: ${escapeHtml(e.message)}</div>`;
  }
}

document.getElementById("btn-refresh-drift-explain").addEventListener("click", loadDriftExplain);

// ---- model status strip (MLflow replacement) ----
async function loadModelStats() {
  const el = document.getElementById("model-stats");
  el.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const [s, drift] = await Promise.all([
      apiGet("/v3/model/stats"),
      apiGet("/v3/monitoring/drift").catch(() => null),
    ]);
    const ckpt = s.checkpoint || {};
    const m = s.latest_incremental_metrics || {};
    const prAucs = Object.entries(m)
      .filter(([k]) => k.startsWith("pr_auc_"))
      .map(([k, v]) => `${k.replace("pr_auc_", "")} ${Number(v).toFixed(3)}`)
      .join(" · ") || "—";
    const lastRun = s.latest_run
      ? `${s.latest_run.run_type} · ${new Date(s.latest_run.timestamp).toLocaleString()}`
      : "none yet";
    const stat = (k, v, cls = "") =>
      `<div class="stat ${cls}"><div class="k">${k}</div><div class="v ${String(v).length > 14 ? "small" : ""}">${escapeHtml(v)}</div></div>`;
    el.innerHTML =
      stat("Checkpoint", ckpt.exists ? `${ckpt.size_mb ?? "?"} MB` : "missing", "hl") +
      stat("Features / edges", ckpt.exists ? `${ckpt.n_features ?? "?"} / ${ckpt.n_edge_types ?? "?"}` : "—") +
      stat("Scored population", Number(s.n_scored ?? 0).toLocaleString()) +
      stat("Confirmed fraud", String(s.n_confirmed ?? 0)) +
      stat("False positives", String(s.n_false_positives ?? 0)) +
      stat("Drift", drift ? drift.recommendation : "n/a") +
      stat("Last eval PR-AUC", prAucs, "small") +
      stat("Last run", lastRun, "small") +
      stat("Total runs", String(s.n_runs ?? 0));
  } catch (e) {
    el.innerHTML = `<div class="empty-state">Failed to load stats: ${escapeHtml(e.message)}</div>`;
  }
}

// ---- run history table (MLflow replacement) ----
async function loadHistory() {
  const body = document.getElementById("history-body");
  body.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const data = await apiGet("/v3/model/registry?limit=25");
    if (!data.runs.length) { body.innerHTML = `<div class="empty-state">No runs recorded yet.</div>`; return; }
    let html = '<table><thead><tr><th>When</th><th>Type</th><th>Cycle</th><th>Metrics</th><th>Checkpoint</th></tr></thead><tbody>';
    for (const r of data.runs) {
      const when = new Date(r.timestamp).toLocaleString();
      const metrics = Object.entries(r.metrics || {})
        .map(([k, v]) => `${k}=${typeof v === "number" ? Number(v).toFixed(3) : v}`).join(", ") || "—";
      const ck = r.checkpoint && r.checkpoint.size_mb != null ? `${r.checkpoint.size_mb} MB` : "—";
      const smoke = r.smoke_test ? ' <span class="badge amber">smoke</span>' : "";
      html += `<tr style="cursor:default"><td>${escapeHtml(when)}</td><td>${escapeHtml(r.run_type)}${smoke}</td><td>${escapeHtml(r.cycle)}</td><td style="white-space:normal">${escapeHtml(metrics)}</td><td>${escapeHtml(ck)}</td></tr>`;
    }
    html += "</tbody></table>";
    body.innerHTML = html;
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

document.getElementById("btn-refresh-stats").addEventListener("click", loadModelStats);
document.getElementById("btn-refresh-history").addEventListener("click", loadHistory);

// ---- Step 1: intake — CSV upload ----
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
document.getElementById("btn-browse").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault(); dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) uploadDataset(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) uploadDataset(fileInput.files[0]); });

async function uploadDataset(file) {
  const out = document.getElementById("intake-report");
  out.style.display = "block";
  out.textContent = `Uploading ${file.name}…`;
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(API_BASE + "/v3/monitoring/upload-dataset", { method: "POST", body: fd });
    const rep = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(rep.detail || res.status);
    const ok = rep.schema_ok;
    out.textContent =
      `${ok ? "✓" : "✗"} ${rep.filename} — ${Number(rep.n_rows).toLocaleString()} rows, ${rep.n_cols}/${rep.expected_cols} cols\n` +
      `schema: ${ok ? "OK" : "MISSING: " + rep.missing_columns.slice(0, 8).join(", ")}\n` +
      (rep.extra_columns.length ? `extra cols (ignored): ${rep.extra_columns.slice(0, 8).join(", ")}\n` : "") +
      (rep.duplicate_ids.length ? `⚠ duplicate application_ids vs raw: ${rep.duplicate_ids.slice(0, 5).join(", ")}\n` : "") +
      `→ path: ${rep.dataset_path}`;
    document.getElementById("eval-dataset-path").value = rep.dataset_path;
    if (intakePurpose === "pattern") {
      document.getElementById("btn-pattern-test").disabled = !ok;
      document.getElementById("btn-pattern-ingest").disabled = !ok;
      document.getElementById("btn-pattern-retest").style.display = "none";
      document.getElementById("pattern-result").style.display = "none";
      toast(ok ? "Pattern CSV accepted — test or ingest below." : "Schema mismatch — cannot use this file.", ok ? "success" : "error");
    } else {
      toast(ok ? "Cohort accepted — ready to evaluate." : "Schema mismatch — cannot evaluate.", ok ? "success" : "error");
    }
  } catch (e) {
    out.textContent = `Failed: ${e.message}`;
    toast(`Upload failed: ${e.message}`, "error");
  }
}

// ---- intake purpose toggle: cohort vs new fraud pattern (relational LOE) ----
let intakePurpose = "cohort";
(function initPatternFtype() {
  const sel = document.getElementById("pattern-ftype");
  if (sel) sel.innerHTML = VALID_FRAUD_TYPES.map((t) => `<option value="${t}">${t}</option>`).join("");
})();

document.querySelectorAll("#intake-purpose [data-purpose]").forEach((b) => {
  b.addEventListener("click", () => {
    intakePurpose = b.dataset.purpose;
    document.querySelectorAll("#intake-purpose [data-purpose]").forEach((x) => x.classList.toggle("on", x === b));
    const isPattern = intakePurpose === "pattern";
    document.getElementById("pattern-actions").style.display = isPattern ? "block" : "none";
    document.querySelectorAll(".cohort-step").forEach((s) => { s.style.display = isPattern ? "none" : "block"; });
    document.getElementById("dz-note").textContent = isPattern
      ? "Full raw-schema rows for the new fraud ring (fresh application_ids). Test or ingest below after upload."
      : "Validated against the canonical raw schema before anything is touched.";
  });
});

function patternScoresText(res, label) {
  const rows = res.members
    .map((m) => `  ${m.application_id}: ${Number(m.hybrid_anomaly_score).toFixed(4)}`)
    .join("\n");
  return `${label} — ${res.n_members} members (mean ${Number(res.mean_score).toFixed(4)}, ` +
         `max ${Number(res.max_score).toFixed(4)}):\n${rows}`;
}

async function runPatternTest(label) {
  const path = document.getElementById("eval-dataset-path").value.trim();
  const out = document.getElementById("pattern-result");
  if (!path) { toast("Upload the pattern CSV first.", "error"); return; }
  out.style.display = "block";
  out.textContent = "Testing (rebuilds features + graph read-only — can take ~1 min)…";
  try {
    const res = await apiPost("/v3/supervisor/pattern/test", { dataset_path: path });
    out.textContent = patternScoresText(res, label || "Detection test");
  } catch (e) {
    out.textContent = `Failed: ${e.message}`;
    toast(`Test failed: ${e.message}`, "error");
  }
}

document.getElementById("btn-pattern-test").addEventListener("click", () => runPatternTest("Detection test (current model)"));
document.getElementById("btn-pattern-retest").addEventListener("click", () => runPatternTest("Detection test (after retrain)"));

document.getElementById("btn-pattern-ingest").addEventListener("click", async () => {
  const path = document.getElementById("eval-dataset-path").value.trim();
  const by = document.getElementById("pattern-by").value.trim();
  const out = document.getElementById("pattern-result");
  if (!path) { toast("Upload the pattern CSV first.", "error"); return; }
  if (!by) { toast("Enter a reviewer name.", "error"); return; }
  if (!confirm("This permanently adds the ring to the dataset + topology exposure and dispatches a retrain. Continue?")) return;
  out.style.display = "block";
  out.textContent = "Ingesting (permanent merge + feature/graph rebuild — can take ~1 min)…";
  const btnI = document.getElementById("btn-pattern-ingest");
  btnI.disabled = true;
  try {
    const res = await apiPost("/v3/supervisor/pattern/ingest", {
      dataset_path: path,
      fraud_type: document.getElementById("pattern-ftype").value,
      confirmed_by: by,
      notes: document.getElementById("pattern-notes").value.trim(),
      dispatch_retrain: true,
      smoke_test: document.getElementById("pattern-smoke").checked,
    });
    const te = res.topology_exposure;
    const rel = Object.entries(te.edges_per_relation).map(([k, v]) => `${k}:${v}`).join(", ");
    let msg = `Ingested pattern ${res.pattern_id}: ${res.n_members} members, ` +
              `${te.n_edges_directed} edges (${rel}) as exposure cluster ${te.cluster_id}. ` +
              `Confirmed ${res.n_confirmed_recorded}.`;
    if (res.confirm_errors && res.confirm_errors.length) msg += `\nSkipped ${res.confirm_errors.length} confirm(s).`;
    out.textContent = msg;
    if (res.retrain_dispatched) {
      const pj = document.getElementById("poll-job-id");
      if (pj) pj.value = res.job_id;
      pollJob(res.job_id, out);
      document.getElementById("btn-pattern-retest").style.display = "";
    }
    toast("Pattern ingested — retrain dispatched.", "success");
    loadStats();
  } catch (e) {
    out.textContent = `Failed: ${e.message}`;
    toast(`Ingest failed: ${e.message}`, "error");
  } finally {
    btnI.disabled = false;
  }
});

// ---- Step 2: evaluate ----
document.getElementById("btn-evaluate").addEventListener("click", async () => {
  const dataset_path = document.getElementById("eval-dataset-path").value.trim();
  const out = document.getElementById("eval-result");
  if (!dataset_path) { toast("Upload a cohort or enter a dataset path first.", "error"); return; }
  out.style.display = "block";
  out.textContent = "Evaluating… (rebuilds features + graph synchronously — can take minutes)";
  try {
    const res = await apiPost("/v3/monitoring/evaluate-dataset", { dataset_path });
    out.textContent = JSON.stringify(res, null, 2);
  } catch (e) {
    out.textContent = `Failed: ${e.message}`;
  }
});

// ---- Step 3: decide (acts on the active cohort) ----
document.querySelectorAll(".decide-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const action = btn.dataset.action;
    const dataset_path = document.getElementById("eval-dataset-path").value.trim();
    const cycle = document.getElementById("decision-cycle").value.trim() || "unknown";
    const decided_by = document.getElementById("decision-by").value.trim();
    const out = document.getElementById("decision-result");
    if (!decided_by) { toast("decided_by is required.", "error"); return; }
    if (action !== "none" && !dataset_path) { toast("Upload/enter a dataset first for incremental/full.", "error"); return; }
    if (action !== "none" &&
        !confirm(`"${action}" will MERGE ${dataset_path} into the raw data PERMANENTLY and retrain. Continue?`)) return;
    out.style.display = "block";
    out.textContent = "Submitting…";
    try {
      const res = await apiPost("/v3/training/decision", { dataset_path, action, cycle, decided_by, smoke_test: false });
      out.textContent = JSON.stringify(res, null, 2);
      if (res.job_id) {
        const js = document.getElementById("training-job-status");
        js.style.display = "block";
        document.getElementById("poll-job-id").value = res.job_id;
        pollJob(res.job_id, js);
      }
      loadHistory();
    } catch (e) {
      out.textContent = `Failed: ${e.message}`;
    }
  });
});

document.getElementById("btn-train-incremental").addEventListener("click", async () => {
  const cycle = document.getElementById("decision-cycle").value.trim() || "2026H1";
  const out = document.getElementById("training-job-status");
  out.style.display = "block";
  out.textContent = "Dispatching incremental training…";
  try {
    const res = await apiPost(`/v3/training/incremental?cycle=${encodeURIComponent(cycle)}&smoke_test=false`, {});
    out.textContent = JSON.stringify(res, null, 2);
    document.getElementById("poll-job-id").value = res.job_id;
    if (res.job_id) pollJob(res.job_id, out);
  } catch (e) {
    out.textContent = `Failed: ${e.message}`;
  }
});

document.getElementById("btn-train-full").addEventListener("click", async () => {
  if (!confirm("Trigger the FULL pipeline? On the CPU-only server this is slow and mutates production outputs.")) return;
  const out = document.getElementById("training-job-status");
  out.style.display = "block";
  out.textContent = "Dispatching full pipeline…";
  try {
    const res = await apiPost("/v3/training/full?smoke_test=false", {});
    out.textContent = JSON.stringify(res, null, 2);
    document.getElementById("poll-job-id").value = res.job_id;
    if (res.job_id) pollJob(res.job_id, out);
  } catch (e) {
    out.textContent = `Failed: ${e.message}`;
  }
});

document.getElementById("btn-poll-job").addEventListener("click", () => {
  const jobId = document.getElementById("poll-job-id").value.trim();
  const out = document.getElementById("training-job-status");
  if (!jobId) return;
  out.style.display = "block";
  pollJob(jobId, out);
});

document.getElementById("btn-rollback").addEventListener("click", async () => {
  const versioned_path = document.getElementById("rollback-path").value.trim();
  if (!versioned_path) { toast("Enter a checkpoint path.", "error"); return; }
  if (!confirm(`Roll back to ${versioned_path}? This replaces the live checkpoint.`)) return;
  try {
    await apiPost("/v3/model/rollback", { versioned_path });
    toast("Rollback dispatched.", "success");
    loadModelStats();
    loadHistory();
  } catch (e) {
    toast(`Failed: ${e.message}`, "error");
  }
});

// ---------- init ----------

loadStats();
loadQueue();
