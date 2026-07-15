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

async function loadQueue() {
  const body = document.getElementById("queue-body");
  body.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const rows = await apiGet("/v3/monitoring/top-suspicious?n=20");
    if (!rows.length) {
      body.innerHTML = `<div class="empty-state">No suspicious applications in the last run.</div>`;
      return;
    }
    const cols = Object.keys(rows[0]);
    const idCol = cols.includes("application_id") ? "application_id" : cols[0];
    const scoreCol = cols.find((c) => c.toLowerCase().includes("score")) || cols[1];

    let html = "<table><thead><tr><th>Application ID</th><th>Risk score</th><th></th></tr></thead><tbody>";
    for (const row of rows) {
      const appId = row[idCol];
      const score = typeof row[scoreCol] === "number" ? row[scoreCol].toFixed(4) : row[scoreCol];
      html += `<tr data-app-id="${escapeHtml(appId)}">
        <td>${escapeHtml(appId)}</td>
        <td>${escapeHtml(score)}</td>
        <td style="text-align:right"><button class="ghost open-row">Open →</button></td>
      </tr>`;
    }
    html += "</tbody></table>";
    body.innerHTML = html;

    body.querySelectorAll("tr[data-app-id]").forEach((tr) => {
      tr.addEventListener("click", () => selectApp(tr.dataset.appId, tr));
    });
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Could not load queue: ${escapeHtml(e.message)}</div>`;
  }
}

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
}

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
    toast(ok ? "Cohort accepted — ready to evaluate." : "Schema mismatch — cannot evaluate.", ok ? "success" : "error");
  } catch (e) {
    out.textContent = `Failed: ${e.message}`;
    toast(`Upload failed: ${e.message}`, "error");
  }
}

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
