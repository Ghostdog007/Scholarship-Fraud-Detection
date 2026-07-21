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
    if (btn.dataset.view === "patterns") { loadPatterns(); loadFlaggedHistory(); }
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

// Pagination. The queue pulls the full flagged set (every one of these has a
// reviewer card) and pages through it client-side, 50 rows per page, rather than
// hard-capping at the top 50. selectedIds persists across pages, so a reviewer
// can gather a ring that spans several pages before flagging it for LOE.
// To widen this to the entire scored population (15k), raise QUEUE_FETCH_N —
// but rows past the ~500 carded apps have no card, so "Open" will 404 for them.
let queuePage = 0;
const QUEUE_PAGE_SIZE = 50;
const QUEUE_FETCH_N = 500;

// Cohort mode: null = the committed base run; else the name of an evaluated
// cohort (read-only preview). Cohort scores are the PRE-FUSION hybrid score on an
// arbitrary positive scale, so we bucket badges by within-cohort percentile
// (computed in loadQueue) instead of the base run's 0-1 risk cutoffs.
let currentCohort = null;
let cohortScoreHi = null, cohortScoreMd = null;

function queueEndpoint() {
  return currentCohort
    ? `/v3/monitoring/cohort/${encodeURIComponent(currentCohort)}/top-suspicious?n=${QUEUE_FETCH_N}`
    : `/v3/monitoring/top-suspicious?n=${QUEUE_FETCH_N}`;
}
function cardEndpoint(id) {
  return currentCohort
    ? `/v3/monitoring/cohort/${encodeURIComponent(currentCohort)}/${encodeURIComponent(id)}/card`
    : `/v3/monitoring/${encodeURIComponent(id)}/card`;
}

function riskBucket(v) {
  const n = Number(v);
  if (currentCohort && cohortScoreHi !== null) {
    return n >= cohortScoreHi ? "high" : n >= cohortScoreMd ? "med" : "low";
  }
  return n >= 0.66 ? "high" : n >= 0.33 ? "med" : "low";
}
function riskLabel(b) { return b === "high" ? "High" : b === "med" ? "Medium" : "Low"; }

async function loadQueue() {
  const body = document.getElementById("queue-body");
  body.innerHTML = `<div class="empty-state">Loading...</div>`;
  try {
    const rows = await apiGet(queueEndpoint());
    if (!rows.length) {
      const what = currentCohort ? `cohort "${escapeHtml(currentCohort)}"` : "the last run";
      body.innerHTML = `<div class="empty-state">No scored applications in ${what}.</div>`;
      document.getElementById("queue-toolbar").style.display = "none";
      document.getElementById("queue-pager").style.display = "none";
      return;
    }
    const cols = Object.keys(rows[0]);
    queueIdCol = cols.includes("application_id") ? "application_id" : cols[0];
    queueScoreCol = cols.find((c) => c.toLowerCase().includes("score")) || cols[1];
    queueRows = rows;
    // Within-cohort percentile buckets (80th=high, 50th=med) so pre-fusion badges
    // and the risk filter stay meaningful; the base run keeps its 0-1 cutoffs.
    if (currentCohort) {
      const vals = rows.map((r) => Number(r[queueScoreCol])).filter(Number.isFinite).sort((a, b) => a - b);
      const q = (p) => (vals.length ? vals[Math.min(vals.length - 1, Math.floor(p * vals.length))] : 0);
      cohortScoreHi = q(0.80); cohortScoreMd = q(0.50);
    } else {
      cohortScoreHi = cohortScoreMd = null;
    }
    selectedIds.clear();
    queuePage = 0;
    document.getElementById("queue-toolbar").style.display = "flex";
    applyCohortModeUI();
    renderQueue();
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Could not load queue: ${escapeHtml(e.message)}</div>`;
  }
}

function renderQueue() {
  const body = document.getElementById("queue-body");
  const q = (document.getElementById("queue-search").value || "").trim().toLowerCase();
  const rf = document.getElementById("queue-risk-filter").value;

  const filtered = queueRows.filter((row) => {
    const id = String(row[queueIdCol]);
    if (dismissedIds.has(id)) return false;
    if (q && !id.toLowerCase().includes(q)) return false;
    const b = riskBucket(row[queueScoreCol]);
    if (rf === "high" && b !== "high") return false;
    if (rf === "med" && b === "low") return false;
    return true;
  });

  if (!filtered.length) {
    body.innerHTML = `<div class="empty-state">No applications match the current filter${dismissedIds.size ? " (some removed)" : ""}.</div>`;
    renderPager(0, 0);
    updateSelectionUI();
    return;
  }

  // Clamp the current page in case a filter/removal shrank the list.
  const totalPages = Math.max(1, Math.ceil(filtered.length / QUEUE_PAGE_SIZE));
  if (queuePage > totalPages - 1) queuePage = totalPages - 1;
  const start = queuePage * QUEUE_PAGE_SIZE;
  const visible = filtered.slice(start, start + QUEUE_PAGE_SIZE);

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
  renderPager(filtered.length, totalPages);
  updateSelectionUI();
}

// Prev / Next pager under the queue table. Shows the current page, page count,
// and how many flagged rows matched the active filter. Buttons re-render the
// same page slice — selection persists via selectedIds across page changes.
function renderPager(nFiltered, totalPages) {
  const pager = document.getElementById("queue-pager");
  if (nFiltered <= QUEUE_PAGE_SIZE) { pager.style.display = "none"; return; }
  const from = queuePage * QUEUE_PAGE_SIZE + 1;
  const to = Math.min(nFiltered, from + QUEUE_PAGE_SIZE - 1);
  pager.style.display = "flex";
  pager.innerHTML = `
    <span class="pager-info">Showing <b>${from}–${to}</b> of <b>${nFiltered}</b> flagged</span>
    <div class="spacer"></div>
    <button id="pager-prev" ${queuePage === 0 ? "disabled" : ""}>← Prev</button>
    <span class="pager-page">Page ${queuePage + 1} / ${totalPages}</span>
    <button id="pager-next" ${queuePage >= totalPages - 1 ? "disabled" : ""}>Next →</button>`;
  pager.querySelector("#pager-prev").addEventListener("click", () => {
    if (queuePage > 0) { queuePage -= 1; renderQueue(); document.getElementById("queue-body").scrollIntoView({ behavior: "smooth", block: "start" }); }
  });
  pager.querySelector("#pager-next").addEventListener("click", () => {
    if (queuePage < totalPages - 1) { queuePage += 1; renderQueue(); document.getElementById("queue-body").scrollIntoView({ behavior: "smooth", block: "start" }); }
  });
}

function updateSelectionUI() {
  const n = selectedIds.size;
  document.getElementById("qt-count").textContent = `${n} selected`;
  const cohort = !!currentCohort;  // cohort: export works; only training actions gated
  document.getElementById("btn-remove-selected").disabled = n === 0;
  document.getElementById("btn-export-selected").disabled = n === 0;
  document.getElementById("btn-label-selected").disabled = n === 0 || cohort;
  document.getElementById("btn-flag-loe-selected").disabled = n === 0 || cohort;
  document.getElementById("btn-restore-removed").style.display = dismissedIds.size ? "" : "none";
  const all = document.getElementById("chk-select-all");
  const chks = document.querySelectorAll("#queue-body .row-chk");
  all.checked = chks.length > 0 && [...chks].every((c) => c.checked);
}

// ---------- LOE coverage (soft, IP-only dedup hint) ----------
// Cross-checks an application against the persistent flagged-pattern store on the
// shares_ip edge; if its cluster may already be flagged, shows a soft warning
// (never a block) telling the reviewer to verify via the 3D ring. Heuristic by
// design — backed by src/confirmed_fraud_graph_store.ip_coverage_for_app.
function renderCoverageInto(el, cov) {
  if (!el) return;
  if (!cov || !cov.covered || !cov.matches.length) { el.style.display = "none"; el.innerHTML = ""; return; }
  const parts = cov.matches.map((m) => {
    const where = m.is_member
      ? "this app is listed in it"
      : `shares an IP with ${m.n_shared_ip} of its member${m.n_shared_ip === 1 ? "" : "s"}`;
    const tag = m.in_exposure ? "already in LOE exposure"
      : m.state === "CONFIRMED" ? "pending confirmation"
      : String(m.state || "").toLowerCase();
    return `<span class="cov-pats">${escapeHtml(m.pattern_id)}</span> (${escapeHtml(m.fraud_type || "?")}, ${escapeHtml(tag)}) — ${where}`;
  });
  el.style.display = "block";
  el.innerHTML =
    `⚠ <b>Looks like this cluster may already be flagged.</b> ` +
    parts.join("; ") + `. ` +
    `This is a heuristic match on <code>shares_ip</code>, not a confirmation — ` +
    `open the <b>◎ 3D identity ring</b> to check it is the same ring before flagging. ` +
    `If it is, there is no need to re-add it. Cross-check under ` +
    `<b>Pattern queue → Flagged history</b>.`;
}

async function loadCoverage(appId, el) {
  if (!el) return;
  el.style.display = "none"; el.innerHTML = "";
  try {
    const cov = await apiGet(`/v3/supervisor/patterns/coverage/${encodeURIComponent(appId)}`);
    renderCoverageInto(el, cov);
  } catch (e) { /* coverage is advisory — never block the card/modal on its failure */ }
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

  // Cohort preview: the read-only views/export have full parity (card, 3D ring,
  // ego-graph, export). Only Flag-for-LOE stays gated — it feeds training and
  // needs the cohort ingested. Coverage (a committed-graph check) is meaningless
  // for staged apps, so skip it.
  const cohort = !!currentCohort;
  document.getElementById("btn-open-topology").disabled = false;
  document.getElementById("btn-export-app").disabled = false;
  document.getElementById("btn-flag-loe").disabled = cohort;
  const covEl = document.getElementById("coverage-banner");
  if (cohort) { covEl.style.display = "none"; } else { loadCoverage(appId, covEl); }

  const frame = document.getElementById("card-frame");
  frame.style.height = "420px";
  frame.onload = () => autosizeFrame(frame);
  frame.srcdoc = "<p style='font-family:sans-serif;padding:14px;color:#c9d1d9;background:#0d1117;margin:0;'>Loading card…</p>";
  try {
    const html = await apiGetText(cardEndpoint(appId));
    frame.srcdoc = html ?? "<p style='font-family:sans-serif;padding:14px;color:#c9d1d9;background:#0d1117;margin:0;'>No card for this application.</p>";
  } catch (e) {
    frame.srcdoc = `<p style='font-family:sans-serif;padding:14px;color:#ffb3b5;background:#0d1117;margin:0;'>Failed to load card: ${escapeHtml(e.message)}</p>`;
  }
}

document.getElementById("btn-refresh-queue").addEventListener("click", loadQueue);
// Filtering changes which rows exist, so snap back to page 1 to avoid landing on
// an out-of-range page.
document.getElementById("queue-search").addEventListener("input", () => { queuePage = 0; renderQueue(); });
document.getElementById("queue-risk-filter").addEventListener("change", () => { queuePage = 0; renderQueue(); });

// ---------- cohort switcher (base run vs read-only evaluated cohorts) ----------
function applyCohortModeUI() {
  const cohort = !!currentCohort;
  const banner = document.getElementById("cohort-banner");
  if (cohort) {
    banner.style.display = "block";
    banner.innerHTML =
      `<b>Cohort preview — ${escapeHtml(currentCohort)}.</b> Read-only scoring of an ingested dataset. ` +
      `Scores are the <b>pre-fusion</b> <code>hybrid_anomaly_score</code> (higher = more anomalous), ` +
      `bucketed by within-cohort percentile — not the committed fused risk. The <b>3D identity ring</b> ` +
      `works; export, ego-graph and Flag-for-LOE need the cohort ingested (committed) first.`;
  } else {
    banner.style.display = "none";
  }
  // Export works in cohort mode (staged bundle), so keep "Export all flagged"
  // visible; relabel it for clarity.
  document.getElementById("btn-export-bulk").textContent = cohort ? "⤓ Export all (cohort)" : "⤓ Export all flagged";
  // "Remove cohort" only applies to an evaluated cohort (never the base run).
  document.getElementById("btn-remove-cohort").style.display = cohort ? "" : "none";
  // Only the training-feeding actions need the cohort committed.
  const hint = cohort ? "Ingest (commit) this cohort to enable — it feeds training" : "";
  ["btn-label-selected", "btn-flag-loe-selected"].forEach((id) => {
    document.getElementById(id).title = hint;
  });
  updateSelectionUI();
}

async function loadCohorts() {
  const sel = document.getElementById("cohort-select");
  const base = '<option value="">Primary dataset · 15k scored applications</option>';
  try {
    const data = await apiGet("/v3/monitoring/cohorts");
    const opts = data.cohorts.map((c) => {
      const extra = c.ring_available ? "" : ", no rings — re-evaluate";
      return `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} · ${c.n_rows ?? "?"} rows${extra}</option>`;
    }).join("");
    sel.innerHTML = base + opts;
  } catch (e) {
    sel.innerHTML = base;
  }
  sel.value = currentCohort || "";
}

document.getElementById("cohort-select").addEventListener("change", (e) => {
  currentCohort = e.target.value || null;
  document.getElementById("detail-section").style.display = "none";
  selectedAppId = null;
  loadQueue();
});

// Remove an evaluated cohort (demo reset): drops its staged preview files so it
// leaves the dropdown, then falls back to the base run. Base data + the
// downloadable sample CSV are untouched.
document.getElementById("btn-remove-cohort").addEventListener("click", async () => {
  if (!currentCohort) return;
  const name = currentCohort;
  if (!confirm(
    `Remove the evaluated cohort "${name}"?\n\n` +
    `⚠ This DISCARDS all of the cohort's outputs on the server:\n` +
    `   • its explanation / reviewer cards\n` +
    `   • its 3D identity rings and ego-graphs\n` +
    `   • its pre-fusion scores and evidence\n` +
    `   • its uploaded CSV\n\n` +
    `The base 15k data and the downloadable sample CSV are NOT touched.\n` +
    `To see this cohort again you must re-upload and re-evaluate the CSV.`
  )) return;
  try {
    const res = await apiPost(`/v3/monitoring/cohort/${encodeURIComponent(name)}/delete`, {});
    toast(`Removed cohort "${name}" (${res.removed.length} file(s)).`, "success");
    currentCohort = null;
    document.getElementById("detail-section").style.display = "none";
    selectedAppId = null;
    await loadCohorts();
    loadQueue();
  } catch (e) {
    toast(`Remove failed: ${e.message}`, "error");
  }
});

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

// Export endpoints switch with the active dataset: cohort mode bundles the
// staged (pre-fusion) evidence for the cohort apps.
const cohortSeg = () => `/v3/monitoring/cohort/${encodeURIComponent(currentCohort)}`;

document.getElementById("btn-export-app").addEventListener("click", () => {
  if (!selectedAppId) { toast("Select an application first.", "error"); return; }
  downloadUrl(currentCohort
    ? `${cohortSeg()}/${encodeURIComponent(selectedAppId)}/export`
    : `/v3/monitoring/${encodeURIComponent(selectedAppId)}/export`);
  toast(`Preparing export for ${selectedAppId}…`);
});

document.getElementById("btn-export-bulk").addEventListener("click", () => {
  if (currentCohort) {
    toast(`Bundling all of cohort "${currentCohort}" — this can take a moment…`);
    downloadUrl(`${cohortSeg()}/export-bulk`);
  } else {
    toast("Bundling all flagged applications — this can take a moment…");
    downloadUrl(`/v3/monitoring/export/bulk`);
  }
});

document.getElementById("btn-export-selected").addEventListener("click", () => {
  if (!selectedIds.size) { toast("Select one or more applications first.", "error"); return; }
  const ids = [...selectedIds].map(encodeURIComponent).join(",");
  downloadUrl(currentCohort
    ? `${cohortSeg()}/export-selected?ids=${ids}`
    : `/v3/monitoring/export/selected?ids=${ids}`);
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

// The topology endpoint for the active dataset. Both the 3D ring and the flat
// ego-graph render against the persisted staged graph in cohort mode, so both
// segments work there just like the base run.
function topoEndpoint(kind) {
  const k = kind === "ring" ? "ring" : "topology";
  return currentCohort
    ? `/v3/monitoring/cohort/${encodeURIComponent(currentCohort)}/${encodeURIComponent(selectedAppId)}/${k}`
    : `/v3/monitoring/${encodeURIComponent(selectedAppId)}/${k}`;
}

async function loadTopoFrame() {
  if (!selectedAppId) return;
  const frame = document.getElementById("topo-frame");
  frame.srcdoc = "<p style='font-family:sans-serif;padding:16px;color:#c9d1d9;background:#0d1117;margin:0;'>Rendering…</p>";
  try {
    const html = await apiGetText(topoEndpoint(topoKind));
    frame.srcdoc = html ?? "<p style='font-family:sans-serif;padding:16px;color:#c9d1d9;background:#0d1117;margin:0;'>No typed edges for this application — nothing to draw.</p>";
  } catch (e) {
    frame.srcdoc = `<p style='font-family:sans-serif;padding:16px;color:#ffb3b5;background:#0d1117;margin:0;'>Failed: ${escapeHtml(e.message)}</p>`;
  }
}

function openTopo(kind) {
  if (!selectedAppId) return;
  document.getElementById("topo-app-id").textContent = selectedAppId;
  document.getElementById("seg-ego").disabled = false;   // ego works in cohort mode too
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
  window.open(`${API_BASE}${topoEndpoint(topoKind)}`, "_blank");
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
// The "center" application_id posted with the pattern. For a single-card flag
// it's that app; for a bulk flag it's the first selected id (the rest ride along
// as the ring's other nodes). Decouples the LOE modal from selectedAppId so the
// same form serves both the per-card button and the toolbar's bulk button.
let loeCenterId = null;

// Programmatically switch tabs by re-using the nav button's own click handler
// (which also lazy-loads that view). Used after a bulk LOE flag to drop the
// reviewer straight into the Pattern queue where the new candidate now shows.
function activateView(name) {
  const btn = document.querySelector(`nav.tabs button[data-view="${name}"]`);
  if (btn) btn.click();
}

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

// ids: array of application IDs to flag as one candidate ring. Single-card flag
// passes [selectedAppId]; the toolbar's bulk flag passes every checkbox-selected
// id. The reviewer can still edit the ID list, fraud type, and shared-edge in the
// modal before recording.
function openLoe(ids) {
  const list = (ids || []).map(String).filter(Boolean);
  if (!list.length) return;
  loeCenterId = list[0];
  const bulk = list.length > 1;
  document.getElementById("loe-app-id").textContent = bulk ? `${list.length} applications` : list[0];
  document.getElementById("loe-nodes").value = list.join(", ");
  document.getElementById("loe-notes").value = "";
  document.getElementById("loe-by").value = "";
  document.getElementById("loe-edge").value = "shares_ip";
  loeType = "IP_CLUSTER";
  document.querySelectorAll("#loe-type-chips .chip").forEach((c) =>
    c.classList.toggle("on", c.dataset.type === "IP_CLUSTER"));
  loeModal.classList.add("open");
  // Check the center id against already-flagged clusters (soft, IP-only).
  loadCoverage(loeCenterId, document.getElementById("loe-coverage"));
}

function closeLoe() { loeModal.classList.remove("open"); }

document.getElementById("btn-flag-loe").addEventListener("click", () => {
  if (selectedAppId) openLoe([selectedAppId]);
});
document.getElementById("btn-flag-loe-selected").addEventListener("click", () => {
  if (!selectedIds.size) { toast("Select one or more applications first.", "error"); return; }
  openLoe([...selectedIds]);
});
document.getElementById("btn-loe-close").addEventListener("click", closeLoe);
document.getElementById("btn-loe-cancel").addEventListener("click", closeLoe);
loeModal.addEventListener("click", (e) => { if (e.target === loeModal) closeLoe(); });

document.getElementById("btn-loe-submit").addEventListener("click", async () => {
  if (!loeCenterId) return;
  const notes = document.getElementById("loe-notes").value.trim();
  const nodesRaw = document.getElementById("loe-nodes").value.trim();
  const edgeType = document.getElementById("loe-edge").value;
  const confirmed_by = document.getElementById("loe-by").value.trim();

  if (!confirmed_by) { toast("Enter your reviewer name.", "error"); return; }
  if (!nodesRaw) { toast("List at least this application's ID.", "error"); return; }
  if (loeType === "OTHER" && !notes) { toast("Describe the pattern when the type is OTHER.", "error"); return; }

  const nodes = nodesRaw.split(",").map((s) => s.trim()).filter(Boolean);
  const subgraph = { nodes, edges: [{ type: edgeType }] };

  try {
    const res = await apiPost("/v3/supervisor/patterns/confirm", {
      application_id: loeCenterId, fraud_type: loeType, subgraph, confirmed_by, notes,
    });
    toast(`Pattern ${res.pattern_id} recorded (${nodes.length} member${nodes.length === 1 ? "" : "s"}) — now awaiting confirmation.`, "success");
    closeLoe();
    // Drop the reviewer into the Pattern queue so the new candidate is visibly
    // there (activateView reloads the list); clear the queue selection since
    // those rows are now flagged.
    selectedIds.clear();
    updateSelectionUI();
    activateView("patterns");
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

// ---------- flagged history (persistent all-sessions directory) ----------
function flagStateBadge(state, inExposure) {
  const s = String(state || "").toUpperCase();
  const cls = s === "PROMOTED" ? "green" : s === "REJECTED" ? "grey" : "amber";
  const label = s === "CONFIRMED" ? "pending" : s.toLowerCase();
  let html = `<span class="badge ${cls}">${escapeHtml(label)}</span>`;
  if (inExposure) html += ` <span class="badge exposure">in LOE exposure</span>`;
  return html;
}

async function loadFlaggedHistory() {
  const body = document.getElementById("flagged-history-body");
  body.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const data = await apiGet("/v3/supervisor/patterns/all");
    if (!data.patterns.length) {
      body.innerHTML = `<div class="empty-state">Nothing flagged yet — flag a ring from the Review queue.</div>`;
      return;
    }
    body.innerHTML = data.patterns.map((p) => {
      const sg = p.subgraph || {};
      const members = sg.nodes || sg.member_ids || [];
      const inExp = !!(p.exposure && p.exposure.appended);
      const when = p.updated_at || p.created_at || "";
      const whenTxt = when ? new Date(when).toLocaleString() : "—";
      const clusterTxt = (p.exposure && p.exposure.cluster_id != null)
        ? ` · exposure cluster #${p.exposure.cluster_id}` : "";
      return `<div class="flag-hist-item" data-pattern-id="${escapeHtml(p.pattern_id)}" data-state="${escapeHtml(String(p.state || ""))}">
        <div class="fh-head">
          <input type="checkbox" class="fh-check" value="${escapeHtml(p.pattern_id)}">
          <span class="fh-id">${escapeHtml(p.pattern_id)}</span>
          <span>${escapeHtml(p.fraud_type || "?")}</span>
          ${flagStateBadge(p.state, inExp)}
        </div>
        <div class="fh-meta">by ${escapeHtml(p.confirmed_by || "?")} · ${escapeHtml(whenTxt)} · ${members.length} member${members.length === 1 ? "" : "s"}${clusterTxt}${p.notes ? " · " + escapeHtml(p.notes) : ""}</div>
        <div class="fh-members">${members.map(escapeHtml).join(", ")}</div>
      </div>`;
    }).join("");
    body.querySelectorAll(".fh-check").forEach((chk) => {
      chk.addEventListener("change", () => {
        if (chk.checked) flaggedSelected.add(chk.value); else flaggedSelected.delete(chk.value);
        updateFlaggedSelectionUI();
      });
    });
    flaggedSelected.clear();
    updateFlaggedSelectionUI();
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Could not load flagged history: ${escapeHtml(e.message)}</div>`;
  }
}

// Selection + hard-delete for the flagged-history directory. Delete removes the
// store RECORD only — the backend reports which deleted ids were PROMOTED so we
// can warn that their exposure/training effect persists until a rebuild.
const flaggedSelected = new Set();

function updateFlaggedSelectionUI() {
  const n = flaggedSelected.size;
  document.getElementById("fh-count").textContent = `${n} selected`;
  document.getElementById("btn-delete-flagged").disabled = n === 0;
  const all = document.getElementById("fh-select-all");
  const chks = document.querySelectorAll("#flagged-history-body .fh-check");
  all.checked = chks.length > 0 && [...chks].every((c) => c.checked);
}

document.getElementById("fh-select-all").addEventListener("change", (e) => {
  const on = e.target.checked;
  document.querySelectorAll("#flagged-history-body .fh-check").forEach((chk) => {
    chk.checked = on;
    if (on) flaggedSelected.add(chk.value); else flaggedSelected.delete(chk.value);
  });
  updateFlaggedSelectionUI();
});

document.getElementById("btn-delete-flagged").addEventListener("click", async () => {
  const ids = [...flaggedSelected];
  if (!ids.length) return;
  // Warn harder when any selected pattern is PROMOTED (already in exposure).
  const promoted = ids.filter((id) => {
    const el = document.querySelector(`.flag-hist-item[data-pattern-id="${id}"]`);
    return el && el.dataset.state === "PROMOTED";
  });
  let msg = `Delete ${ids.length} flagged pattern(s) from the history? This removes the record only.`;
  if (promoted.length) {
    msg += `\n\n⚠ ${promoted.length} of these are PROMOTED — their ring may already be in the ` +
           `topology-exposure set and the current checkpoint. Deleting the record does NOT ` +
           `un-train the model or remove the exposure cluster (that needs a rebuild/retrain).`;
  }
  if (!confirm(msg)) return;
  try {
    const res = await apiPost("/v3/supervisor/patterns/delete", { pattern_ids: ids });
    let t = `Deleted ${res.removed.length} pattern(s).`;
    if (res.removed_promoted.length) t += ` ${res.removed_promoted.length} were promoted (exposure effect persists until rebuild).`;
    if (res.not_found.length) t += ` ${res.not_found.length} not found.`;
    toast(t, "success");
    flaggedSelected.clear();
    loadFlaggedHistory();
    loadPatterns();  // a deleted CONFIRMED pattern also leaves the candidates list
  } catch (e) {
    toast(`Delete failed: ${e.message}`, "error");
  }
});

document.getElementById("btn-refresh-flagged-history").addEventListener("click", loadFlaggedHistory);

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

// Sample CSV is served statically by nginx from the frontend/ dir. It carries the
// full raw schema with fresh application_ids and a planted shared-IP ring, so the
// reviewer can upload → evaluate → see the cohort preview + rings immediately.
document.getElementById("btn-download-template").addEventListener("click", (e) => {
  e.stopPropagation();
  const a = document.createElement("a");
  a.href = "sample_cohort.csv"; a.download = "sample_cohort.csv";
  document.body.appendChild(a); a.click(); a.remove();
});

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
    // The cohort is now previewable — refresh the Review-queue dataset dropdown.
    loadCohorts();
    toast("Cohort evaluated — pick it in the Review queue's Dataset dropdown to review it.", "success");
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

document.getElementById("btn-upload-ckpt").addEventListener("click", async () => {
  const fileInput = document.getElementById("ckpt-file");
  const out = document.getElementById("ckpt-upload-status");
  const file = fileInput.files[0];
  if (!file) { toast("Choose a .pth checkpoint file first.", "error"); return; }
  if (!file.name.endsWith(".pth")) { toast("File must be a .pth checkpoint.", "error"); return; }
  if (!confirm(`Upload ${file.name} (${(file.size / 1048576).toFixed(1)} MB)? ` +
      `If validation passes, it becomes the live model.`)) return;

  const form = new FormData();
  form.append("file", file);
  form.append("cycle", document.getElementById("ckpt-cycle").value.trim() || "unknown");
  form.append("source_ref", document.getElementById("ckpt-source").value.trim() || "console-upload");

  out.style.display = "block";
  out.textContent = `Uploading ${file.name}…`;
  try {
    const res = await fetch(API_BASE + "/v3/training/upload-checkpoint", {
      method: "POST",
      body: form,   // multipart — browser sets the boundary header itself
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
    out.textContent = JSON.stringify(payload, null, 2);
    toast("Checkpoint uploaded — validating.", "success");
    if (payload.job_id) {
      document.getElementById("poll-job-id").value = payload.job_id;
      pollJob(payload.job_id, out);
    }
  } catch (e) {
    out.textContent = `Failed: ${e.message}`;
    toast(`Upload failed: ${e.message}`, "error");
  }
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
loadCohorts();
loadQueue();
