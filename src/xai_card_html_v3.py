"""
xai_card_html_v3.py

Renders the evidence-first explanation cards (outputs/explanation_cards_v3.json)
as self-contained, interactive, offline HTML reviewer cards — one per top-N
application plus an index. Presentation layer only: consumes the JSON the XAI
layer already produced (plus risk_scores_v3.csv for neighbour risk colouring);
computes nothing new, invents no thresholds.

Two-pane layout per card:
  * left  — interactive: an identity ego-network (the applicant + its shared-IP
            co-applications) and a per-detector "signal drivers" panel;
  * right — the explanation: risk placement, ranked reason codes, per-field
            declared-vs-model-expected breakdown (progressive disclosure), and
            the recommended action.

Every displayed number traces to a field in the card JSON. No CDN, no network:
each file embeds its own CSS/JS and is openable directly in a browser.

Reads:  outputs/explanation_cards_v3.json, outputs/risk_scores_v3.csv
Writes: outputs/cards/card_<rank>_<application_id>.html, outputs/cards/index.html

Run:
  python -m src.xai_card_html_v3            # top 50 cards
  python -m src.xai_card_html_v3 --top-k 100
"""

import argparse
import html
import json
from pathlib import Path

import pandas as pd

CARDS_JSON = Path("outputs/explanation_cards_v3.json")
RISK_CSV   = Path("outputs/risk_scores_v3.csv")
OUT_DIR    = Path("outputs/cards")

# risk → colour bucket (semantic, separate from any brand accent)
RISK_LOW, RISK_MED, RISK_HIGH = "#2c7da0", "#f4a261", "#e5383b"
DETECTOR_ORDER = [
    ("subspace",         "Tabular subspace", "#e5383b"),
    ("dense_relational", "Shared-identity dense-block", "#f72585"),
    ("hybrid",           "Relational RGCN", "#4cc9f0"),
]
GROUP_LABELS = {"financial": "Financial", "identity": "Identity", "network": "Network"}
EDGE_LABELS = {
    "shares_ip": "IP address", "shares_mobile": "mobile number",
    "shares_father_name": "father's name", "shares_mother_name": "mother's name",
    "shares_pincode": "pincode",
}


def _risk_color(v: float) -> str:
    return RISK_HIGH if v >= 0.66 else RISK_MED if v >= 0.33 else RISK_LOW


# key -> (short label, colour) for the model-traceability pills. Same keys and
# colours as DETECTOR_ORDER so pills match the fusion-composition footer.
DETECTOR_PILL = {k: (lbl, col) for k, lbl, col in DETECTOR_ORDER}
# Deep SAD (V4.2) is SUPPLEMENTARY, not a fusion driver — deliberately NOT added
# to DETECTOR_ORDER (that list drives the fusion-composition footer, which must
# stay exactly the 3 locked fusion inputs). It still gets its own pill colour so
# its evidence line can be visually distinguished from the fused-score drivers.
DETECTOR_PILL["deepsad"] = ("Deep SAD (supplementary)", "#9d4edd")


def _model_pill(key: str | None) -> str:
    """Small colour-coded pill naming the source model of an evidence line."""
    if key not in DETECTOR_PILL:
        return ""
    lbl, col = DETECTOR_PILL[key]
    return (f"<span class='mpill' style='color:{col};border-color:{col}55;"
            f"background:{col}1a' title='source model: {lbl}'>{lbl}</span>")


def _model_trace(card: dict) -> str:
    """Render the 'How this score was produced' traceability section from the
    pre-computed evidence.model_trace block. Adds no numbers of its own.

    Max fusion (changed 2026-07-22): exactly one detector's own normalised
    value IS the fused score's source — there is no "% share of a blend" any
    more, so each bar shows the detector's own value (0-1) with a WON DRIVER
    badge on whichever one is the max, plus how far it led the runner-up."""
    mt = card.get("evidence", {}).get("model_trace")
    if not mt or not mt.get("models"):
        return ""
    rows = []
    for m in mt["models"]:
        lbl, col = DETECTOR_PILL.get(m["key"], (m.get("name", m["key"]), "#8b949e"))
        normalized = float(m.get("normalized", 0.0)) * 100
        is_driver = bool(m.get("drove_score"))
        margin = m.get("margin_over_next")
        driver_badge = (f" <span class='flagchip' style='background:{col}22;color:{col};"
                        f"border-color:{col}55'>DRIVER"
                        + (f" · led by {margin:.2f}" if margin is not None else "") + "</span>") if is_driver else ""
        fired = m.get("fired_triggers", [])
        if fired:
            meta_tail = " · fired " + ", ".join(f"<span class='num'>{_esc(t)}</span>" for t in fired)
        elif m.get("note"):
            meta_tail = f" · <span style='color:var(--faint)'>{_esc(m['note'])}</span>"
        else:
            meta_tail = " · <span style='color:var(--faint)'>no trigger fired for this application</span>"
        rows.append(f"""
          <div class="mtrace">
            <div class="mtrace-h">
              <span class="mpill" style="color:{col};border-color:{col}55;background:{col}1a">{_esc(lbl)}</span>
              <span class="mtrace-name">{_esc(m['name'])}</span>
              <span class="mtrace-share num" style="color:{col}">{normalized:.0f}</span>{driver_badge}
            </div>
            <div class="mtrace-track"><div style="width:{max(0.0, min(100.0, normalized)):.0f}%;background:{col}"></div></div>
            <div class="mtrace-what">{_esc(m['what'])}</div>
            <div class="mtrace-meta"><b>Reads:</b> {_esc(m['reads'])}{meta_tail}</div>
          </div>""")
    return (
        "<div class='section-t'>How this score was produced — model traceability</div>"
        "<div class='mtrace-intro'>Max fusion of three independent models — the fused score IS "
        "the single highest normalised detector value below, marked DRIVER; the others are shown "
        f"for context (<code>{_esc(mt.get('fusion_formula', ''))}</code>).</div>"
        + "".join(rows)
    )


def _is_suspicious(card: dict) -> bool:
    """A card is rendered only for flagged applications — one that crossed an
    EVT threshold, carries a self-training trigger, or holds a non-negative
    label. Unflagged applications get no card (keeps the batch small even when
    frauds are many)."""
    ev = card.get("evidence", {})
    if ev.get("evt_crossings"):
        return True
    if card.get("triggers"):
        return True
    if ev.get("label_source", "negative") != "negative":
        return True
    return False


def _esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------------------
# Shared static assets (one copy per file — self-contained, offline)
# ---------------------------------------------------------------------------

CSS = """
:root{--ground:#0d1117;--panel:#161b22;--panel2:#1c2230;--raise:#20283a;
--border:rgba(255,255,255,.09);--border2:rgba(255,255,255,.14);
--fg:#e6edf3;--fg2:#c9d1d9;--muted:#8b949e;--faint:#6b7280;--cyan:#4cc9f0;--ip:#f72585;
--low:#2c7da0;--med:#f4a261;--high:#e5383b;--ok:#4ade80;
--mono:ui-monospace,"SF Mono","Cascadia Mono","Consolas",monospace;
--sans:"Inter","Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--fg);font-family:var(--sans);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.wrap{max-width:1180px;margin:0 auto;padding:22px}
.topbar{display:flex;flex-wrap:wrap;align-items:center;gap:14px;padding:0 4px 18px;border-bottom:1px solid var(--border);margin-bottom:20px}
.appid{font-family:var(--mono);font-size:15px;letter-spacing:.3px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
.spacer{flex:1}
.status{display:inline-flex;align-items:center;gap:7px;padding:5px 11px;border-radius:999px;font-size:12px;font-weight:600}
.status .dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.grid{display:grid;grid-template-columns:minmax(0,.82fr) minmax(0,1fr);gap:20px}
@media(max-width:880px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.card-b{padding:16px}
.tabs{display:flex;gap:4px;padding:8px 8px 0}
.tab{flex:1;text-align:center;padding:9px;font-size:12.5px;font-weight:600;color:var(--muted);background:transparent;border:0;border-radius:9px 9px 0 0;cursor:pointer;transition:.15s}
.tab:hover{color:var(--fg2)}.tab.on{color:var(--fg);background:var(--panel2)}
.pane{display:none;padding:16px}.pane.on{display:block}
canvas.net{width:100%;height:320px;display:block;background:radial-gradient(120% 120% at 50% 30%,#131a26 0%,var(--panel) 70%);border-radius:10px;cursor:grab}
.tip{position:absolute;pointer-events:none;background:#0b0f16;border:1px solid var(--border2);border-radius:8px;padding:8px 10px;font-size:12px;opacity:0;transition:.1s;box-shadow:0 6px 20px rgba(0,0,0,.5);z-index:5;white-space:nowrap}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:11.5px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dotr{width:9px;height:9px;border-radius:50%;display:inline-block}
.swatch{width:20px;height:3px;border-radius:2px;display:inline-block}
.sig{margin-bottom:15px}
.sig-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px}
.sig-name{font-size:12.5px;color:var(--fg2)}.sig-name b{color:var(--fg);font-weight:600}
.sig-val{font-family:var(--mono);font-size:12px;color:var(--muted)}
.track{height:9px;border-radius:6px;background:rgba(255,255,255,.06);overflow:hidden;position:relative}
.fill{height:100%;border-radius:6px}
.thr{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--fg2);opacity:.85}
.sig-note{font-size:11px;color:var(--faint);margin-top:4px}
.flagchip{font-size:10px;font-weight:700;letter-spacing:.05em;padding:2px 7px;border-radius:5px;background:rgba(247,37,133,.18);color:#ff6db0;border:1px solid rgba(247,37,133,.35)}
.riskhead{display:flex;align-items:center;gap:18px;padding:4px 2px 16px}
.gauge{width:92px;height:92px;border-radius:50%;flex:0 0 auto;display:grid;place-items:center;position:relative}
.gauge::before{content:"";position:absolute;inset:8px;border-radius:50%;background:var(--panel)}
.gauge b{position:relative;font-family:var(--mono);font-size:22px;font-weight:700}
.rh-txt .big{font-size:15px;font-weight:600}.rh-txt .sub{color:var(--muted);font-size:12.5px;margin-top:2px}
.rh-txt .sub b{color:var(--fg2)}
.section-t{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin:18px 2px 10px;font-weight:700}
.reason{display:flex;gap:11px;padding:11px 12px;border-radius:10px;background:var(--panel2);border:1px solid var(--border);margin-bottom:8px}
.reason .rk{font-family:var(--mono);font-size:12px;color:var(--faint);padding-top:1px}
.reason .rt{font-size:13px;font-weight:600}.reason .rd{font-size:12px;color:var(--muted);margin-top:3px}
.reason .rd .num{color:var(--fg2)}
.field{border:1px solid var(--border);border-radius:10px;margin-bottom:8px;overflow:hidden;background:var(--panel2)}
.field-h{display:flex;align-items:center;gap:12px;padding:11px 13px;cursor:pointer;user-select:none}
.field-h:hover{background:var(--raise)}
.stripe{width:3px;align-self:stretch;border-radius:3px;flex:0 0 auto}
.field-name{flex:1;font-size:13px;font-weight:600}
.field-tag{font-size:10.5px;color:var(--muted);font-family:var(--mono)}
.chev{color:var(--faint);transition:.2s;font-size:11px}.field.open .chev{transform:rotate(90deg)}
.field-b{display:none;padding:2px 14px 14px 28px;border-top:1px solid var(--border)}.field.open .field-b{display:block}
.cmp-row{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:12px}
.cmp-lab{width:78px;color:var(--muted);flex:0 0 auto}
.cmp-bar{flex:1;height:22px;border-radius:5px;background:rgba(255,255,255,.05);position:relative;overflow:hidden}
.cmp-bar i{position:absolute;top:0;bottom:0;border-radius:4px}
.cmp-mid{position:absolute;top:-3px;bottom:-3px;left:50%;width:1px;background:var(--border2)}
.cmp-v{font-family:var(--mono);font-size:12px;width:64px;text-align:right;flex:0 0 auto}
.why{font-size:12.5px;color:var(--fg2);background:rgba(76,201,240,.06);border-left:2px solid var(--cyan);padding:9px 12px;border-radius:0 8px 8px 0;margin-top:10px}
.why b{color:var(--cyan);font-weight:600}
.action{margin-top:18px;padding:14px 15px;border-radius:12px}
.action .at{font-size:11px;letter-spacing:.1em;text-transform:uppercase;font-weight:700}
.action .ab{font-size:13px;margin-top:6px}
.action .btns{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.btn{padding:8px 14px;border-radius:8px;font-size:12.5px;font-weight:600;cursor:pointer;border:1px solid var(--border2);background:var(--raise);color:var(--fg)}
.btn:hover{border-color:var(--fg2)}
.btn.danger{background:rgba(229,56,59,.2);border-color:rgba(229,56,59,.45);color:#ffb3b5}
.btn.ok{background:rgba(74,222,128,.14);border-color:rgba(74,222,128,.4);color:#a7f3c0}
.footnote{font-size:11px;color:var(--faint);margin-top:18px;padding-top:12px;border-top:1px solid var(--border);line-height:1.6}
.footnote code{font-family:var(--mono);color:var(--muted)}
.rvform{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px;align-items:flex-end}
.rvform label{font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);display:flex;flex-direction:column;gap:4px}
.rvform input,.rvform select{background:var(--ground);border:1px solid var(--border2);color:var(--fg);border-radius:7px;padding:7px 9px;font-size:12.5px;font-family:var(--sans)}
.rvform input:focus,.rvform select:focus{outline:none;border-color:var(--cyan)}
.rvform input.name{width:130px}.rvform input.cycle{width:100px}.rvform input.notes{flex:1;min-width:150px}
#rv-status{font-size:12.5px;margin-top:12px;min-height:18px;opacity:0;transition:.15s;font-weight:600}
.rvform-t{width:100%;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);font-weight:700;margin-bottom:2px}
/* index */
.idx-row{display:grid;grid-template-columns:44px 1fr auto auto;gap:14px;align-items:center;padding:12px 14px;border:1px solid var(--border);border-radius:11px;margin-bottom:8px;background:var(--panel);transition:.12s}
.idx-row:hover{border-color:var(--border2);background:var(--panel2)}
.idx-rank{font-family:var(--mono);color:var(--faint);font-size:13px}
.idx-id{font-family:var(--mono);font-size:13.5px}
.idx-reason{font-size:12px;color:var(--muted);margin-top:2px}
.idx-risk{font-family:var(--mono);font-weight:700;font-size:15px}
.idx-pill{font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px}
/* model-traceability pills + trace section */
.mpill{display:inline-block;font-size:9.5px;font-weight:700;letter-spacing:.03em;padding:1px 6px;border-radius:5px;border:1px solid;vertical-align:middle;font-family:var(--mono);white-space:nowrap}
.mtrace-intro{font-size:12px;color:var(--muted);margin:0 2px 12px}
.mtrace-intro code{font-family:var(--mono);color:var(--fg2);font-size:11px}
.mtrace{border:1px solid var(--border);border-radius:10px;background:var(--panel2);padding:11px 13px;margin-bottom:8px}
.mtrace-h{display:flex;align-items:center;gap:9px;margin-bottom:7px}
.mtrace-name{flex:1;font-size:12.5px;font-weight:600}
.mtrace-share{font-size:14px;font-weight:700}
.mtrace-track{height:6px;border-radius:4px;background:rgba(255,255,255,.06);overflow:hidden;margin-bottom:8px}
.mtrace-track>div{height:100%;border-radius:4px}
.mtrace-what{font-size:12px;color:var(--fg2);line-height:1.5}
.mtrace-meta{font-size:11px;color:var(--muted);margin-top:6px}
.mtrace-meta b{color:var(--fg2);font-weight:600}
.export-link{font-size:12px;color:var(--muted);display:inline-flex;align-items:center;gap:5px}
"""

# JS: tab switching + interactive ego-network. Nodes injected as a JSON blob.
JS_TEMPLATE = """
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  t.closest('.card').querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  t.closest('.card').querySelectorAll('.pane').forEach(x=>x.classList.remove('on'));
  t.classList.add('on');
  document.getElementById('pane-'+t.dataset.t).classList.add('on');
});
const NODES=__NODES__;
const cv=document.getElementById('net');
if(cv&&NODES.length){
  const ctx=cv.getContext('2d'),tip=document.getElementById('tip');
  const W=()=>cv.width/devicePixelRatio,H=()=>320;
  function fit(){const r=cv.getBoundingClientRect();cv.width=r.width*devicePixelRatio;cv.height=320*devicePixelRatio;
    ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);layout();draw();}
  function layout(){const cx=W()/2,cy=H()/2,R=Math.min(W(),H())*0.34,k=NODES.length-1;
    NODES[0].x=cx;NODES[0].y=cy;
    for(let i=1;i<NODES.length;i++){const a=-Math.PI/2+(i-1)*(2*Math.PI/Math.max(k,1));
      NODES[i].x=cx+R*Math.cos(a);NODES[i].y=cy+R*Math.sin(a);}}
  function draw(){ctx.clearRect(0,0,W(),H());
    ctx.lineWidth=2.5;ctx.strokeStyle='rgba(247,37,133,.55)';
    for(let i=1;i<NODES.length;i++){ctx.beginPath();ctx.moveTo(NODES[0].x,NODES[0].y);ctx.lineTo(NODES[i].x,NODES[i].y);ctx.stroke();}
    NODES.forEach(n=>{const r=n.me?18:12;
      ctx.beginPath();ctx.arc(n.x,n.y,r+4,0,7);ctx.fillStyle='rgba(13,17,23,.9)';ctx.fill();
      ctx.beginPath();ctx.arc(n.x,n.y,r,0,7);ctx.fillStyle=n.c;ctx.fill();
      ctx.lineWidth=n.me?2.5:1.2;ctx.strokeStyle=n.me?'#fff':'rgba(255,255,255,.25)';ctx.stroke();
      ctx.fillStyle='#c9d1d9';ctx.font='11px ui-monospace,monospace';ctx.textAlign='center';
      ctx.fillText(n.id,n.x,n.y+r+15);});}
  let drag=null;
  function at(e){const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    return NODES.find(n=>Math.hypot(n.x-mx,n.y-my)<(n.me?20:15));}
  cv.onmousedown=e=>{drag=at(e);cv.style.cursor=drag?'grabbing':'grab';};
  addEventListener('mouseup',()=>{drag=null;cv.style.cursor='grab';});
  cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    if(drag){drag.x=mx;drag.y=my;draw();return;}
    const n=at(e);
    if(n){tip.style.opacity=1;tip.style.left=(mx+14)+'px';tip.style.top=(my+8)+'px';
      tip.innerHTML='<b>'+n.id+'</b>'+(n.risk!=null?'<br>risk <span style=\\'font-family:monospace\\'>'+n.risk.toFixed(2)+'</span>':'')+(n.me?'<br><span style=\\'color:#ff6db0\\'>this application</span>':'<br>shares IP');}
    else tip.style.opacity=0;};
  cv.onmouseleave=()=>tip.style.opacity=0;
  fit();addEventListener('resize',fit);
}

// ---- supervisor review loop (POST to the live API; same-origin when served) ----
function _rvStatus(msg, ok){
  const el=document.getElementById('rv-status'); if(!el) return;
  el.textContent=msg; el.style.color=ok?'#a7f3c0':'#ffb3b5'; el.style.opacity=1;
}
function _rvName(){ const el=document.getElementById('rv-name'); return el?el.value.trim():''; }
async function _rvPost(url, body){
  if(location.protocol==='file:'){
    throw new Error('open this card via the API (…/card), not as a local file, to submit');
  }
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  let j={}; try{ j=await r.json(); }catch(e){}
  if(!r.ok) throw new Error(j.detail||('HTTP '+r.status));
  return j;
}
async function confirmFraud(appId){
  const by=_rvName(); if(!by){ _rvStatus('Enter your reviewer name first.', false); return; }
  const type=document.getElementById('rv-type').value;
  const notes=document.getElementById('rv-notes').value||'';
  const cycle=document.getElementById('rv-cycle').value||'';
  _rvStatus('Submitting…', true);
  try{
    const j=await _rvPost('/v3/supervisor/confirm-fraud',
      {application_id:appId, fraud_type:type, confirmed_by:by, notes:notes, cycle:cycle});
    _rvStatus('\\u2714 Confirmed as '+type+' — added to store ('+j.n_confirmed+' total).', true);
  }catch(e){ _rvStatus('\\u2716 '+e.message, false); }
}
async function clearApp(appId){
  const by=_rvName(); if(!by){ _rvStatus('Enter your reviewer name first.', false); return; }
  const notes=document.getElementById('rv-notes').value||'';
  _rvStatus('Submitting…', true);
  try{
    const j=await _rvPost('/v3/supervisor/mark-false-positive',
      {application_id:appId, confirmed_by:by, notes:notes});
    _rvStatus('\\u2714 Marked false positive ('+j.n_false_positives+' total).', true);
  }catch(e){ _rvStatus('\\u2716 '+e.message, false); }
}
async function clearLabel(appId){
  _rvStatus('Clearing…', true);
  try{
    const j=await _rvPost('/v3/supervisor/clear-label', {application_id:appId});
    _rvStatus('\\u21ba Label cleared — this application is back to unlabelled '+
      '(confirmed '+j.n_confirmed+', FP '+j.n_false_positives+').', true);
  }catch(e){ _rvStatus('\\u2716 '+e.message, false); }
}
"""


def _page(title: str, body: str, nodes_json: str = "[]") -> str:
    js = JS_TEMPLATE.replace("__NODES__", nodes_json)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>"
        f"{body}<script>{js}</script></body></html>"
    )


# ---------------------------------------------------------------------------
# Left pane — interactive
# ---------------------------------------------------------------------------

def _ego_nodes(card: dict, risk_map: dict) -> list[dict]:
    """Applicant + its shared-IP co-applications, coloured by risk."""
    app_id = card["application_id"]
    self_risk = float(card.get("risk_score_v3", 0.0))
    nodes = [{"id": _short(app_id), "risk": round(self_risk, 4),
              "c": _risk_color(self_risk), "me": True}]
    ip_conn = next((c for c in card["evidence"].get("graph_connections", [])
                    if c["edge_type"] == "shares_ip"), None)
    if ip_conn:
        for nid in ip_conn.get("sample_ids", [])[:6]:
            r = risk_map.get(nid)
            nodes.append({"id": _short(nid),
                          "risk": round(float(r), 4) if r is not None else None,
                          "c": _risk_color(float(r)) if r is not None else "#6b7280",
                          "me": False})
    return nodes


def _short(app_id: str) -> str:
    s = str(app_id)
    return s if len(s) <= 12 else s[:2] + "…" + s[-6:]


def _left_pane(card: dict, ring_href: str | None = None) -> str:
    ev = card["evidence"]
    ip_conn = next((c for c in ev.get("graph_connections", [])
                    if c["edge_type"] == "shares_ip"), None)
    ip_count = ip_conn["count"] if ip_conn else 0
    ip_pct = ip_conn.get("percentile") if ip_conn else None
    net_note = (
        f"This application shares one IP address with <b style='color:var(--fg2)'>{ip_count} "
        f"other application(s)</b>"
        + (f" — more connected than <span class='num' style='color:var(--fg2)'>{ip_pct:.1f}%</span> "
           "of applicants" if ip_pct is not None else "")
        + ". Drag to reposition · hover a node for its risk."
        if ip_count else
        "No shared-IP co-applications found. Suspicion rests on the feature-level evidence."
    )
    ring_link = (
        f"<div style='margin-top:12px'><a href='{_esc(ring_href)}' target='_blank' "
        f"style='display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:600;"
        f"padding:7px 12px;border-radius:8px;border:1px solid var(--border2);background:var(--raise);"
        f"color:var(--cyan)'>Examine full ring in 3D ↗</a></div>"
        if ring_href else ""
    )

    # signal drivers
    sig_html = _signal_bars(card)

    return f"""
    <div class="card">
      <div class="tabs">
        <button class="tab on" data-t="net">Identity network</button>
        <button class="tab" data-t="sig">Signal drivers</button>
      </div>
      <div class="pane on" id="pane-net">
        <div style="position:relative">
          <canvas class="net" id="net" width="720" height="320"></canvas>
          <div class="tip" id="tip"></div>
        </div>
        <div class="legend">
          <span><i class="swatch" style="background:#f72585"></i> shares IP</span>
          <span><i class="dotr" style="background:#e5383b"></i> high</span>
          <span><i class="dotr" style="background:#f4a261"></i> medium</span>
          <span><i class="dotr" style="background:#2c7da0"></i> low</span>
        </div>
        <p style="font-size:12px;color:var(--muted);margin:12px 2px 0">{net_note}</p>
        {ring_link}
      </div>
      <div class="pane" id="pane-sig">{sig_html}</div>
    </div>"""


def _signal_bars(card: dict) -> str:
    ev = card["evidence"]
    rows = []

    # 1) subspace groups (percentile + EVT threshold marker if crossed)
    for g, d in ev.get("subspace_groups", {}).items():
        pct = d.get("percentile")
        if pct is None:
            continue
        crossed = d.get("crossed")
        thr_pct = 100.0 * float(d["threshold"]) if False else None  # thresholds are score-space
        fill = f"linear-gradient(90deg,#e5383b,#ff6d70)" if crossed else \
               f"linear-gradient(90deg,#f4a261,#ffbf85)"
        chip = " &nbsp;<span class='flagchip'>THRESHOLD CROSSED</span>" if crossed else ""
        note = (f"observed <span class='num'>{d['score']:.3f}</span> vs EVT threshold "
                f"<span class='num'>{d['threshold']:.3f}</span>{chip}") if "threshold" in d else \
               "below its EVT threshold"
        rows.append(_bar(f"{GROUP_LABELS.get(g, g)} subspace detector {_model_pill('subspace')}",
                         f"{pct:.1f}<span style='color:var(--faint)'>pct</span>",
                         pct, fill, note))

    # 2) dense-block-relational (mobile/IP/pincode, IP-priority weighted)
    dib = ev.get("dense_block_relational", {})
    if dib.get("score", 0.0) > 0.0 and dib.get("percentile") is not None:
        by_rel = dib.get("by_relation", {})
        top_rel = max(by_rel.items(), key=lambda kv: kv[1])[0] if by_rel else None
        rel_label = {"mobile": "mobile number", "ip": "IP address", "pincode": "pincode"}.get(top_rel, "identity value")
        rows.append(_bar(f"Shared-identity dense-block {_model_pill('dense_relational')}",
                         f"{dib['percentile']:.1f}<span style='color:var(--faint)'>pct</span>",
                         dib["percentile"], "linear-gradient(90deg,#b5187f,#f72585)",
                         f"member of a dense shared-{rel_label} cluster"))

    # 2b) Deep SAD center-distance — SUPPLEMENTARY, not a fusion input. Only
    # shown when notably elevated (matches the narrative's 75th-pct cutoff).
    dsad = ev.get("deepsad") or {}
    if dsad.get("percentile") is not None and dsad["percentile"] >= 75.0:
        rows.append(_bar(f"Deep SAD center-distance {_model_pill('deepsad')}",
                         f"{dsad['percentile']:.1f}<span style='color:var(--faint)'>pct</span>",
                         dsad["percentile"], "linear-gradient(90deg,#5a189a,#9d4edd)",
                         "supplementary signal — informative context, does not drive the fused score"))

    # 3) fusion composition footer — max fusion (2026-07-22): shows each
    # detector's own normalised value, DRIVER marks the one that won.
    fc = ev.get("fusion_contributions", {})
    comp = ""
    if fc:
        items = [(lbl, fc[k]["normalized"] * 100, col, fc[k].get("is_driver"))
                 for k, lbl, col in DETECTOR_ORDER if k in fc]
        parts = " · ".join(
            f"<span style='color:{c}'>{lbl} {v:.0f}{' <b>(DRIVER)</b>' if drv else ''}</span>"
            for lbl, v, c, drv in items
        )
        comp = (f"<div class='sig-note' style='margin-top:18px;padding-top:12px;"
                f"border-top:1px solid var(--border)'><b style='color:var(--muted)'>Fusion "
                f"composition:</b> {parts}. &nbsp;risk = minmax(max(subspace, dense-relational, hybrid)).</div>")

    intro = ("<p style='font-size:12px;color:var(--muted);margin:0 0 16px'>What is pushing this "
             "application's score, by detector. The marker on a track is its EVT flag threshold.</p>")
    return intro + "".join(rows) + comp


def _bar(name: str, val: str, pct: float, fill: str, note: str) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    return f"""
      <div class="sig">
        <div class="sig-top"><span class="sig-name"><b>{name}</b></span>
          <span class="sig-val">{val}</span></div>
        <div class="track"><div class="fill" style="width:{pct:.1f}%;background:{fill}"></div></div>
        <div class="sig-note">{note}</div>
      </div>"""


# ---------------------------------------------------------------------------
# Right pane — explanation
# ---------------------------------------------------------------------------

def _reason_codes(card: dict) -> str:
    ev = card["evidence"]
    rows, rank = [], 1
    for c in ev.get("evt_crossings", []):
        rows.append(
            f"<div class='reason'><div class='rk'>{rank:02d}</div><div class='rc'>"
            f"<div class='rt'>{_esc(c['label']).capitalize()} {_model_pill(c.get('source_model'))}</div>"
            f"<div class='rd'>Crossed its extreme-value threshold — observed "
            f"<span class='num'>{c['observed']:.3f}</span> vs threshold "
            f"<span class='num'>{c['threshold']:.3f}</span>.</div></div></div>")
        rank += 1
    ip_conn = next((c for c in ev.get("graph_connections", [])
                    if c["edge_type"] == "shares_ip" and c.get("count", 0) > 0), None)
    if ip_conn:
        pct = ip_conn.get("percentile")
        pstr = (f" — more connected than <span class='num'>{pct:.1f}%</span> of applicants"
                if pct is not None else "")
        rows.append(
            f"<div class='reason'><div class='rk'>{rank:02d}</div><div class='rc'>"
            f"<div class='rt'>Shared-IP concentration {_model_pill('dense_relational')}</div>"
            f"<div class='rd'>Shares one IP with <span class='num'>{ip_conn['count']}</span> "
            f"other application(s){pstr}.</div></div></div>")
        rank += 1
    if not rows:
        rows.append("<div class='reason'><div class='rk'>—</div><div class='rc'>"
                    "<div class='rt'>No extreme-value threshold crossed</div>"
                    "<div class='rd'>Card provided for ranking context.</div></div></div>")
    return "".join(rows)


def _clamp_half(v: float, lim: float = 1.5) -> float:
    """Map a signed scaled value to a 0..48% half-bar width."""
    return min(abs(float(v)), lim) / lim * 48.0


def _binary_field_body(f: dict) -> str:
    """Network-consistency body for a binary flag: declared value vs. the
    co-applicant distribution that grounds the model's expectation."""
    ctx = f.get("network_context", {})
    declared = f.get("declared_label", "—")
    m   = ctx.get("n_linked")
    maj = ctx.get("majority_label")
    mc  = ctx.get("majority_count")
    pct = ctx.get("majority_pct")
    edge = ctx.get("context_edge")
    pctw = max(0.0, min(100.0, float(pct or 0)))
    dec_line = (f"<div class='cmp-row'><span class='cmp-lab'>declared</span>"
                f"<span class='cmp-v' style='color:#ffbf85'>{_esc(declared)}</span></div>")
    net_line = (f"<div class='cmp-row'><span class='cmp-lab'>co-applicants</span>"
                f"<div class='cmp-bar'><i style='left:0;width:{pctw:.0f}%;background:{RISK_HIGH}'></i></div>"
                f"<span class='cmp-v'>{mc}/{m}</span></div>") if m else ""
    via = f", linked mainly by {_esc(edge)}," if edge else ""
    why = (f"<div class='why'><b>What this means:</b> this applicant declared "
           f"<b>{_esc(declared)}</b>, but of <span class='num'>{m}</span> co-applicant(s) linked "
           f"to this application{via} <span class='num'>{mc}</span> "
           f"(<span class='num'>{pct:.0f}%</span>) declared <b>{_esc(maj)}</b> — an inconsistency "
           f"with its network that the model weighted as a top signal.</div>") if m else ""
    return dec_line + net_line + why


def _field_accordions(card: dict) -> str:
    out = []
    for i, f in enumerate(card.get("top_feature_errors", [])[:5]):
        ep = f.get("error_percentile")
        stripe = RISK_HIGH if (ep or 0) >= 99.5 else RISK_MED if (ep or 0) >= 90 else RISK_LOW
        kind = f.get("kind", "continuous")
        val = f.get("value")
        exp = f.get("expected")
        cmp_html, why = "", ""
        if kind == "binary":
            tag = f"mismatch {'99.9%+' if ep and ep >= 99.95 else f'{ep:.1f}%'}" if ep is not None else "network"
            cmp_html = _binary_field_body(f)
        else:
            tag = f"miss {'99.9%+' if ep and ep >= 99.95 else f'{ep:.1f}%'}" if ep is not None else ""
        if kind != "binary" and val is not None and exp is not None:
            vw, ew = _clamp_half(val), _clamp_half(exp)
            vside = "left:50%" if val >= 0 else f"right:50%;left:auto"
            eside = "left:50%" if exp >= 0 else f"right:50%;left:auto"
            vcol = RISK_HIGH if abs(val) >= abs(exp) else RISK_MED
            cmp_html = f"""
            <div class="cmp-row"><span class="cmp-lab">declared</span>
              <div class="cmp-bar"><span class="cmp-mid"></span>
                <i style="width:{vw:.0f}%;{vside};background:{vcol}"></i></div>
              <span class="cmp-v" style="color:#ffbf85">{val:+.3f}</span></div>
            <div class="cmp-row"><span class="cmp-lab">expected</span>
              <div class="cmp-bar"><span class="cmp-mid"></span>
                <i style="width:{ew:.0f}%;{eside};background:var(--low)"></i></div>
              <span class="cmp-v" style="color:var(--low)">{exp:+.3f}</span></div>"""
            vp = f.get("value_percentile")
            med = f.get("population_median")
            stand = ""
            if vp is not None:
                stand = (f"higher than <span class='num'>{vp:.1f}%</span>" if vp >= 50
                         else f"lower than <span class='num'>{100 - vp:.1f}%</span>") + " of applicants"
            direction = "above" if (val > exp) else "below"
            # "stand" (value_percentile) is only known when population stats were
            # passed to _top_features — absent for a cohort preview (no committed
            # population to rank against). Without it, drop the "declared value is
            # {stand}, while" lead-in entirely rather than leave a grammatical gap.
            lead = (f"the declared value is {stand}"
                    + (f" (population median {med:+.3f})" if med is not None else "")
                    + ", while ") if stand else ""
            why = (f"<div class='why'><b>What this means:</b> {lead}the model — reading the rest "
                   f"of the record and the application's network "
                   f"context — expected about <span class='num'>{exp:+.3f}</span>; the declared value is "
                   f"<b>{direction}</b> expectation"
                   + (f", and this miss is larger than <span class='num'>{ep:.1f}%</span> of all "
                      f"applications on this field" if ep is not None else "") + ".</div>")
        open_cls = " open" if i == 0 else ""
        out.append(f"""
        <div class="field{open_cls}">
          <div class="field-h" onclick="this.parentNode.classList.toggle('open')">
            <span class="stripe" style="background:{stripe}"></span>
            <span class="field-name">{_esc(f.get('feature_label', f.get('feature')))}</span>
            {_model_pill(f.get('source_model', 'hybrid'))}
            <span class="field-tag">{tag}</span><span class="chev">▶</span>
          </div>
          <div class="field-b">{cmp_html}{why}</div>
        </div>""")
    return "".join(out)


FRAUD_TYPES = ["IP_CLUSTER", "FEE_INFLATION", "INCOME_VIOLATION",
               "NAME_COLLISION", "CROSS_CHANNEL", "OTHER"]


def _suggest_fraud_type(card: dict) -> str:
    """Pre-select the most likely fraud type from the card's own evidence, so the
    reviewer usually just clicks Confirm. Heuristic only — fully overridable."""
    ev = card["evidence"]
    if ev.get("dense_block_relational", {}).get("score", 0.0) > 0.0:
        return "IP_CLUSTER"
    ip_conn = next((c for c in ev.get("graph_connections", [])
                    if c["edge_type"] == "shares_ip" and c.get("count", 0) > 0), None)
    if ip_conn:
        return "IP_CLUSTER"
    groups = ev.get("subspace_groups", {})
    if groups.get("financial", {}).get("crossed"):
        return "INCOME_VIOLATION"
    if any(c["edge_type"] in ("shares_father_name", "shares_mother_name")
           for c in ev.get("graph_connections", []) if c.get("count", 0) > 0):
        return "NAME_COLLISION"
    return "OTHER"


def _action(card: dict) -> str:
    ev = card["evidence"]
    app_id = _esc(card["application_id"])
    label_src = ev.get("label_source", "negative")
    triggers = card.get("triggers", [])
    crossed = bool(ev.get("evt_crossings"))
    if label_src == "confirmed":
        at, col = "Confirmed fraud", "#e5383b"
        ab = "Keep disbursement on hold and coordinate with the investigation team."
    elif triggers:
        at, col = "Recommended action", "#e5383b"
        ab = ("Hold disbursement and request supporting documents (fee receipt, admission letter, "
              "income certificate). Review IP-linked applications together before approval.")
    elif crossed:
        at, col = "Secondary verification", "#f4a261"
        ab = ("Not auto-flagged (signal agreement below the promotion requirement), but a crossed "
              "threshold warrants secondary verification before disbursement.")
    else:
        at, col = "No hold required", "#4ade80"
        ab = "No statistical flag crossed — card provided for ranking context."
    bg = f"linear-gradient(180deg,{col}1a,{col}08)"
    default_type = _suggest_fraud_type(card)
    opts = "".join(
        f"<option value='{t}'{' selected' if t == default_type else ''}>{t.replace('_', ' ').title()}</option>"
        for t in FRAUD_TYPES
    )
    return f"""
      <div class="action" style="background:{bg};border:1px solid {col}47">
        <div class="at" style="color:{col}">{at}</div>
        <div class="ab">{ab}</div>

        <div class="rvform">
          <span class="rvform-t">Reviewer decision — writes to the confirmed-fraud store</span>
          <label>Reviewer<input id="rv-name" class="name" type="text" placeholder="your name"></label>
          <label>Fraud type<select id="rv-type">{opts}</select></label>
          <label>Cycle<input id="rv-cycle" class="cycle" type="text" placeholder="2025-26"></label>
          <label>Notes<input id="rv-notes" class="notes" type="text" placeholder="optional"></label>
        </div>
        <div class="btns">
          <button class="btn danger" onclick="confirmFraud('{app_id}')">⚑ Confirm fraud</button>
          <button class="btn ok" onclick="clearApp('{app_id}')">✓ Mark false positive</button>
          <button class="btn" onclick="clearLabel('{app_id}')">↺ Undo label</button>
        </div>
        <div id="rv-status"></div>
      </div>"""


def _right_pane_preview(card: dict) -> str:
    """Right pane for a pre-fusion cohort PREVIEW card. Reuses _reason_codes and
    _field_accordions verbatim (both already degrade gracefully on empty
    evt_crossings/graph_connections) — omits _action (reviewer decision form),
    since confirm-fraud requires the app to exist in the committed features
    file (confirmed_fraud_store.add_confirmed reads engineered_features_v3.csv),
    which a staged-but-uncommitted cohort app is not in."""
    ev = card["evidence"]
    score = float(card.get("risk_score_v3", 0.0))
    pct = ev.get("risk_percentile")
    rank = ev.get("risk_rank")
    n = ev.get("population_size")
    gcol = _risk_color(score)
    sub = (f"Anomaly score <b class='num'>{score:.4f}</b> — higher than "
           f"<b class='num'>{pct:.2f}%</b> of <b class='num'>{n}</b> cohort application(s)"
           + (f" · <b>rank {rank}</b>" if rank else "")) if pct is not None and n else \
          f"Anomaly score <b class='num'>{score:.4f}</b> (higher = more anomalous)"
    headline = ("Highest-risk application in this cohort" if rank == 1
                else "Elevated anomaly score" if score >= 0.5 else "Ranking context")
    commit_note = (
        "<div class='action' style='background:linear-gradient(180deg,#4cc9f01a,#4cc9f008);"
        "border:1px solid #4cc9f047'>"
        "<div class='at' style='color:#4cc9f0'>Pre-fusion preview</div>"
        "<div class='ab'>This score is the raw hybrid-detector output for this uploaded cohort, "
        "not the committed fused risk — subspace IF, dense-block, and EVT triggers only exist "
        "after a committed pipeline run. Labeling, export, and the ego-graph unlock once this "
        "cohort is committed (admin → Decide → Merge + incremental/full retrain).</div></div>")
    return f"""
    <div class="card"><div class="card-b">
      <div class="riskhead">
        <div class="gauge" style="background:conic-gradient({gcol} {score*100:.0f}%,rgba(255,255,255,.07) 0)">
          <b>{score:.2f}</b></div>
        <div class="rh-txt"><div class="big">{headline}</div><div class="sub">{sub}</div></div>
      </div>
      <div class="section-t">Why it flagged — ranked reason codes</div>
      {_reason_codes(card)}
      <div class="section-t">What's happening in each field — declared vs. model-expected</div>
      {_field_accordions(card)}
      {commit_note}
      <div class="footnote">Pre-fusion preview card: the raw hybrid-detector reconstruction error
        for this uploaded cohort, computed read-only against the current checkpoint. No hand-set
        thresholds. Commit this cohort for the full fused-risk card with EVT triggers and
        model-traceability breakdown.</div>
    </div></div>"""


def _right_pane(card: dict) -> str:
    ev = card["evidence"]
    score = float(card.get("risk_score_v3", 0.0))
    pct = ev.get("risk_percentile")
    rank = ev.get("risk_rank")
    n = ev.get("population_size")
    gcol = _risk_color(score)
    sub = (f"Risk <b class='num'>{score:.4f}</b> — higher than "
           f"<b class='num'>{pct:.2f}%</b> of <b class='num'>{n:,}</b> scored applications · "
           f"<b>rank {rank}</b>") if pct is not None and n else \
          f"Anomaly score <b class='num'>{score:.4f}</b> (higher = more anomalous)"
    headline = ("Highest-risk application in the batch" if rank == 1
                else "Elevated fraud risk" if score >= 0.5 else "Ranking context")
    return f"""
    <div class="card"><div class="card-b">
      <div class="riskhead">
        <div class="gauge" style="background:conic-gradient({gcol} {score*100:.0f}%,rgba(255,255,255,.07) 0)">
          <b>{score:.2f}</b></div>
        <div class="rh-txt"><div class="big">{headline}</div><div class="sub">{sub}</div></div>
      </div>
      <div class="section-t">Why it flagged — ranked reason codes</div>
      {_reason_codes(card)}
      <div class="section-t">What's happening in each field — declared vs. model-expected</div>
      {_field_accordions(card)}
      {_model_trace(card)}
      {_action(card)}
      <div class="footnote">Every number on this card traces to
        <code>explanation_cards_v3.json</code> — no hand-set thresholds; the only numeric gates
        quoted are EVT-derived. Signal bars show each detector's population percentile; the fusion
        composition shows each detector's own normalised value, with the single argmax DRIVER marked
        (max fusion, changed 2026-07-22 — see README changelog).</div>
    </div></div>"""


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def _card_page(card: dict, rank: int, risk_map: dict, ring_href: str | None = None) -> str:
    app_id = card["application_id"]
    score = float(card.get("risk_score_v3", 0.0))
    status = card.get("review_status", "Pending Review")
    scol = _risk_color(score)
    body = f"""
    <div class="wrap">
      <div class="topbar">
        <div><div class="eyebrow">Scholarship fraud review · NIC v3 · card {rank}</div>
          <div class="appid">{_esc(app_id)}</div></div>
        <div class="spacer"></div>
        <a href="index.html" style="font-size:12px;color:var(--muted)">← all cards</a>
        <a class="export-link" href="/v3/monitoring/{_esc(app_id)}/export" download
           title="Download scorecard CSV + this card + evidence JSON">⤓ Export bundle</a>
        <div class="status" style="color:{scol};background:{scol}22;border:1px solid {scol}55">
          <span class="dot"></span>{_esc(status)}</div>
      </div>
      <div class="grid">{_left_pane(card, ring_href)}{_right_pane(card)}</div>
    </div>"""
    nodes_json = json.dumps(_ego_nodes(card, risk_map))
    return _page(f"Reviewer card — {app_id}", body, nodes_json)


# ---------------------------------------------------------------------------
# Per-application Plotly ego-ring (deep-dive; shared plotly.min.js, offline)
# ---------------------------------------------------------------------------

def _ego_figure(app_id: str, app_ids, id_to_idx: dict, nidx: dict, rel_id: dict, risk_map: dict):
    """Build the Plotly 3D ego-ring figure for one application, or None if the
    application is not in the graph. Reuses graph_viz_v3's styling. This is the
    single expensive step — called per batch entry OR lazily per API request."""
    from src.graph_viz_v3 import _figure_for_ring
    import numpy as np

    t = id_to_idx.get(app_id)
    if t is None:
        return None
    node_set = [t]
    seen = {t}
    for e in nidx.get(t, []):
        if e["neighbor_idx"] not in seen:
            seen.add(e["neighbor_idx"]); node_set.append(e["neighbor_idx"])
    local = {g: i for i, g in enumerate(node_set)}
    edges = []
    for g in node_set:
        for e in nidx.get(g, []):
            v = e["neighbor_idx"]
            if v in local:
                edges.append((local[g], local[v], rel_id.get(e["edge_type"], 0)))
    scores = np.array([float(risk_map.get(app_ids[g], 0.0)) for g in node_set], dtype=float)
    sg = {"node_ids": node_set, "scores": scores, "edges": edges}
    fig = _figure_for_ring(sg, app_ids, 1)
    n_edges = len(set((min(a, b), max(a, b)) for a, b, _ in edges))
    fig.update_layout(title=dict(text=(
        f"<b>Identity ring — {app_id}</b>"
        f"<span style='color:#8b949e'>    {len(node_set)} nodes · {n_edges} edges · "
        f"risk {float(scores[0]):.3f}</span>")))
    return fig


# mtime-keyed cache (added 2026-07-22, same pattern as confirmed_fraud_graph_store.py's
# _ip_cache) — this is the .pt GRAPH FALLBACK path only (used when Postgres is
# unavailable; the ring's primary path is already PG-indexed and needs none of
# this). Rebuilding the whole graph's neighbor index from scratch on every
# fallback ring request is the kind of full-population rescan the rest of this
# change is trying to eliminate; keyed by (graph path, mtime, ids path, mtime)
# so both the default committed graph and any staged cohort graph are cached
# independently and safely invalidate when their underlying file changes.
_GRAPH_CTX_CACHE: dict = {}


def _graph_ctx(graph_pt: "Path | None" = None, nodeorder_csv: "Path | None" = None):
    """Load the graph neighbour index + id mapping once (shared by batch + API),
    cached by file identity + mtime so a repeated fallback request doesn't
    rebuild the whole graph's neighbor index from scratch.

    Defaults to the committed graph + features. Pass graph_pt + nodeorder_csv to
    render against a STAGED cohort graph instead (the merged base+cohort graph
    snapshotted by evaluate-dataset), so a read-only cohort preview can draw rings
    for apps that aren't in the committed graph. nodeorder_csv must carry the
    application_id column in the SAME row order the staged graph was built from."""
    from src.config_v3 import EDGE_TYPES
    from src.graph_viz_v3 import FINAL_CSV
    from src.xai_layer_v3 import _build_neighbor_index

    graph_pt = Path(graph_pt) if graph_pt else Path("data/processed/identity_graph_v3.pt")
    ids_csv  = Path(nodeorder_csv) if nodeorder_csv else FINAL_CSV
    if not graph_pt.exists() or not ids_csv.exists():
        return None

    cache_key = (str(graph_pt), graph_pt.stat().st_mtime, str(ids_csv), ids_csv.stat().st_mtime)
    cached = _GRAPH_CTX_CACHE.get(str(graph_pt))
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    app_ids = pd.read_csv(ids_csv)["application_id"].values
    id_to_idx = {a: i for i, a in enumerate(app_ids)}
    nidx = _build_neighbor_index(graph_pt)
    rel_id = {name: i for i, name in enumerate(EDGE_TYPES)}
    ctx = (app_ids, id_to_idx, nidx, rel_id)
    _GRAPH_CTX_CACHE[str(graph_pt)] = (cache_key, ctx)
    return ctx


def _ring_fig_from_pg(app_id: str, risk_map: dict | None):
    """Cut-over path: build the ring figure from indexed Postgres queries
    (ego_neighbors + induced_subgraph_edges) — no .pt graph load. Node and
    edge SETS are identical to the graph path (Gate 2 + the render-parity
    check); node ordering is sorted, so layout is deterministic."""
    import numpy as np
    from src.config_v3 import EDGE_TYPES
    from src.db import reads as db_reads
    from src.graph_viz_v3 import _figure_for_ring

    if not db_reads.identity_row_exists(str(app_id)):
        return None  # unknown application — same as "not in the graph"
    nbrs = db_reads.ego_neighbors(str(app_id))
    neighbor_ids = sorted(set().union(*nbrs.values())) if nbrs else []
    node_ids = [str(app_id)] + [n for n in neighbor_ids if n != str(app_id)]
    # isolated-but-known apps render a 1-node ring, like the graph path
    local = {a: i for i, a in enumerate(node_ids)}
    rel_id = {name: i for i, name in enumerate(EDGE_TYPES)}
    edges = []
    for a, b, rel in db_reads.induced_subgraph_edges(node_ids):
        if a in local and b in local:
            edges.append((local[a], local[b], rel_id.get(rel, 0)))
    if risk_map is None:
        risk_map = db_reads.risk_scores_for(node_ids)
    scores = np.array([float(risk_map.get(a, 0.0)) for a in node_ids], dtype=float)
    sg = {"node_ids": list(range(len(node_ids))), "scores": scores, "edges": edges}
    fig = _figure_for_ring(sg, np.array(node_ids), 1)
    n_edges = len(set((min(a, b), max(a, b)) for a, b, _ in edges))
    fig.update_layout(title=dict(text=(
        f"<b>Identity ring — {app_id}</b>"
        f"<span style='color:#8b949e'>    {len(node_ids)} nodes · {n_edges} edges · "
        f"risk {float(scores[0]):.3f}</span>")))
    return fig


def build_ring_html(app_id: str, risk_map: dict | None = None,
                    graph_pt: "Path | None" = None,
                    nodeorder_csv: "Path | None" = None) -> str | None:
    """LAZY: render one Plotly 3D ego-ring as a standalone HTML string, computed
    on demand (this is what the API's /ring endpoint calls when a link is
    clicked). plotly.js is embedded inline so the response is self-contained.
    Returns None if the application is not in the graph. Pass graph_pt +
    nodeorder_csv to render against a staged cohort graph (see _graph_ctx).

    Committed-graph requests are served from Postgres (indexed 1-hop +
    induced-edge queries) when NIC_READS_FROM_PG allows, falling back to the
    .pt graph on any failure. Staged-cohort requests (graph_pt passed) always
    use the staged bundle."""
    if graph_pt is None:
        try:
            from src.db.reads import reads_from_pg
            if reads_from_pg():
                fig = _ring_fig_from_pg(app_id, risk_map)
                if fig is not None:
                    return fig.to_html(include_plotlyjs=True, full_html=True)
                return None
        except Exception as e:  # noqa: BLE001 — PG outage must not kill rings
            print(f"[xai_card_html] WARNING: PG ring path failed, falling back to graph: {e}")

    ctx = _graph_ctx(graph_pt, nodeorder_csv)
    if ctx is None:
        return None
    app_ids, id_to_idx, nidx, rel_id = ctx
    if risk_map is None:
        risk_map = {}
        if RISK_CSV.exists():
            rdf = pd.read_csv(RISK_CSV)
            risk_map = dict(zip(rdf["application_id"], rdf["risk_score_v3"]))
    fig = _ego_figure(app_id, app_ids, id_to_idx, nidx, rel_id, risk_map)
    if fig is None:
        return None
    return fig.to_html(include_plotlyjs=True, full_html=True)


def build_staged_card_html(name: str, app_id: str) -> str | None:
    """PREVIEW reviewer card for a staged cohort application (scored read-only
    by evaluate-dataset) — reuses the SAME chrome, tabs, identity-network mini
    view, signal-driver rendering, and field accordions as the committed-card
    path (_left_pane/_right_pane_preview/_page), fed by a pseudo evidence dict
    built from what a pre-fusion cohort actually has: the hybrid detector's
    per-feature error/predicted vectors and, when the cohort's graph bundle was
    persisted (Evaluate ran with a graph), REAL shares_ip neighbours from it.

    Genuinely absent pre-commit (never fabricated): subspace_groups,
    dense_block_relational, fusion_contributions, evt_crossings — these require a
    committed pipeline run. _reason_codes/_signal_bars/_model_trace already
    degrade gracefully on empty inputs, so omitting them is a truthful gap,
    not a rendering failure. The reviewer-decision form is omitted (not
    disabled-and-shown) because confirm-fraud reads the COMMITTED features
    file and would 422 on an uncommitted app_id.

    Returns None if the app isn't in the cohort's staged scores."""
    import json as _json
    from src.xai_layer_v3 import _degree_counts_from_index, _top_features

    scores_path = Path(f"outputs/staged_scores_{name}.csv")
    feats_path  = Path(f"outputs/staged_features_{name}.csv")
    if not scores_path.exists():
        return None
    sdf = pd.read_csv(scores_path)
    sdf["application_id"] = sdf["application_id"].astype(str)
    match = sdf[sdf["application_id"] == str(app_id)]
    if match.empty:
        return None
    row = match.iloc[0]

    per_feat  = _json.loads(row["per_feature_error_json"])
    predicted = (_json.loads(row["per_feature_predicted_json"])
                 if "per_feature_predicted_json" in sdf.columns else None)
    actual_vals: dict = {}
    if feats_path.exists():
        fdf = pd.read_csv(feats_path)
        fm  = fdf[fdf["application_id"].astype(str) == str(app_id)]
        if not fm.empty:
            actual_vals = fm.iloc[0].to_dict()

    # Real identity-network preview: shares_ip neighbours from the cohort's
    # persisted graph bundle (same mechanism the 3D ring uses). Absent only
    # if Evaluate ran before graph persistence was added (stale cohort) —
    # degrades to "no shared-IP co-applications found", not a crash.
    graph_pt  = Path(f"outputs/staged_graph_{name}.pt")
    nodeorder = Path(f"outputs/staged_nodeorder_{name}.csv")
    graph_connections: list[dict] = []
    neighbor_idxs = None
    risk_map = dict(zip(sdf["application_id"], sdf["hybrid_anomaly_score"]))
    if graph_pt.exists() and nodeorder.exists():
        ctx = _graph_ctx(graph_pt, nodeorder)
        if ctx is not None:
            g_app_ids, id_to_idx, nidx, _rel_id = ctx
            t = id_to_idx.get(str(app_id))
            if t is not None:
                ip_neighbors = [e["neighbor_idx"] for e in nidx.get(t, [])
                                if e["edge_type"] == "shares_ip"]
                if ip_neighbors:
                    from src.config_v3 import EDGE_TYPES as _ALL_EDGE_TYPES
                    degree = _degree_counts_from_index(nidx, len(g_app_ids), _ALL_EDGE_TYPES)["shares_ip"]
                    this_degree = len(ip_neighbors)
                    pct = 100.0 * (degree < this_degree).sum() / max(len(degree), 1)
                    graph_connections.append({
                        "edge_type": "shares_ip", "count": this_degree,
                        "percentile": round(float(pct), 1),
                        "sample_ids": [str(g_app_ids[i]) for i in ip_neighbors[:6]],
                    })
                    neighbor_idxs = ip_neighbors

    top_feats = _top_features(per_feat, actual_vals, k=6, predicted=predicted,
                              neighbor_idxs=neighbor_idxs)
    score = float(row["hybrid_anomaly_score"])

    # Within-cohort ranking — the only "population" a pre-fusion preview has.
    risk_rank = int((sdf["hybrid_anomaly_score"] > score).sum()) + 1
    risk_percentile = 100.0 * (sdf["hybrid_anomaly_score"] < score).sum() / max(len(sdf), 1)

    pseudo_card = {
        "application_id": app_id,
        "risk_score_v3": score,
        "triggers": [],
        "top_feature_errors": top_feats,
        "evidence": {
            "subspace_groups": {}, "dense_block_relational": {}, "fusion_contributions": {},
            "graph_connections": graph_connections, "evt_crossings": [],
            "risk_percentile": risk_percentile, "risk_rank": risk_rank,
            "population_size": len(sdf), "label_source": "preview",
        },
    }

    ring_href = f"/v3/monitoring/cohort/{name}/{_esc(app_id)}/ring"
    body = f"""
    <div class="wrap">
      <div class="topbar">
        <div><div class="eyebrow">Scholarship fraud review · NIC v3 · cohort preview</div>
          <div class="appid">{_esc(app_id)}</div></div>
        <div class="spacer"></div>
        <div class="status" style="color:#f4a261;background:#f4a26122;border:1px solid #f4a26155">
          <span class="dot"></span>PREVIEW · pre-fusion</div>
      </div>
      <div class="grid">{_left_pane(pseudo_card, ring_href)}{_right_pane_preview(pseudo_card)}</div>
    </div>"""
    nodes_json = json.dumps(_ego_nodes(pseudo_card, risk_map))
    return _page(f"Reviewer card (preview) — {app_id}", body, nodes_json)


def generate_ego_rings(cards: list[dict], risk_map: dict) -> dict[str, str]:
    """BATCH (offline, opt-in): pre-render one Plotly ego-ring file per card,
    with include_plotlyjs="directory" so plotly.min.js is written once and each
    ring file is tens of KB. Only used for the standalone offline export
    (ring_mode="file"); the API path uses build_ring_html instead. Returns
    app_id -> local filename."""
    ctx = _graph_ctx()
    if ctx is None:
        print("[xai_card] graph/features not found — skipping Plotly ego-rings.")
        return {}
    app_ids, id_to_idx, nidx, rel_id = ctx

    hrefs: dict[str, str] = {}
    for rank, card in enumerate(cards, start=1):
        app_id = card["application_id"]
        fig = _ego_figure(app_id, app_ids, id_to_idx, nidx, rel_id, risk_map)
        if fig is None:
            continue
        fn = f"ring_{app_id}.html"
        fig.write_html(str(OUT_DIR / fn), include_plotlyjs="directory", full_html=True)
        hrefs[app_id] = fn
        print(f"[xai_card]   ring #{rank:3d}  {app_id}  -> {OUT_DIR / fn}")
    return hrefs


def build_card_html(app_id: str, rank: int = 1) -> str | None:
    """LAZY: render one lightweight reviewer card as HTML for the API's /card
    endpoint. First checks the pre-computed explanation_cards_v3.json (the
    top-N reviewer-queue batch, fast path — just a JSON scan). If the
    application isn't in that batch (added 2026-07-22 — see README changelog),
    falls back to build_card_for_app(), which assembles a card on demand from
    the cached population context: O(1) relative to population size after the
    first call in this process, not a full-population rescan. This is what
    gives every application a card, not just the top 500, without needing to
    pre-render and store one for the whole scored population. The 3D link
    points at the API's /ring route (also lazy). Returns None only if the
    application isn't in the scored population at all."""
    card = None
    if CARDS_JSON.exists():
        cards = json.loads(CARDS_JSON.read_text())
        card = next((c for c in cards if str(c["application_id"]) == str(app_id)), None)
    if card is None:
        from src.xai_layer_v3 import build_card_for_app
        card = build_card_for_app(app_id)
    if card is None:
        return None
    risk_map = {}
    if RISK_CSV.exists():
        rdf = pd.read_csv(RISK_CSV)
        risk_map = dict(zip(rdf["application_id"], rdf["risk_score_v3"]))
    ring_href = f"/v3/monitoring/{app_id}/ring"   # lazy: computed only when clicked
    return _card_page(card, rank, risk_map, ring_href)


def _index_page(cards: list[dict], filenames: list[str]) -> str:
    rows = []
    for i, (c, fn) in enumerate(zip(cards, filenames), start=1):
        ev = c["evidence"]
        score = float(c.get("risk_score_v3", 0.0))
        scol = _risk_color(score)
        crossings = ev.get("evt_crossings", [])
        reason = (", ".join(x["label"] for x in crossings[:2]) if crossings
                  else "ranking context")
        status = c.get("review_status", "Pending Review").split("—")[0].strip()
        rows.append(f"""
        <a class="idx-row" href="{fn}">
          <span class="idx-rank">{i:02d}</span>
          <span><span class="idx-id">{_esc(c['application_id'])}</span>
            <div class="idx-reason">{_esc(reason)}</div></span>
          <span class="idx-pill" style="color:{scol};background:{scol}22">{status}</span>
          <span class="idx-risk" style="color:{scol}">{score:.3f}</span>
        </a>""")
    body = f"""
    <div class="wrap">
      <div class="topbar"><div>
        <div class="eyebrow">Scholarship fraud review · NIC v3</div>
        <div class="appid" style="font-size:18px">Explanation cards — top {len(cards)}</div>
      </div></div>
      {''.join(rows)}
      <div class="footnote">Ranked by fused risk. Each row opens an interactive evidence card.
        Generated from <code>outputs/explanation_cards_v3.json</code>.</div>
    </div>"""
    return _page("Explanation cards — NIC v3", body)


def render_cards(top_k: int | None = None, suspicious_only: bool = True,
                 ring_mode: str = "api") -> int:
    """Render the lightweight reviewer cards + index into outputs/cards/.

    suspicious_only : only flagged applications get a card (default) — see
                      _is_suspicious. Keeps the batch small even with many frauds.
    ring_mode       : how the "Examine in 3D" link behaves —
        "api"  (default, for MLflow/serving) — link points at the API /ring
               route; Plotly is computed lazily, only when a reviewer clicks.
               No Plotly compute here → cheap.
        "file" (offline standalone) — pre-render a Plotly ring file per card
               (expensive); link is the local file.
        "none" — no 3D link at all (pure simplistic).
    Returns the number of cards written.
    """
    if not CARDS_JSON.exists():
        raise FileNotFoundError(f"{CARDS_JSON} not found — run the XAI layer first (src.xai_layer_v3).")
    cards = json.loads(CARDS_JSON.read_text())
    if suspicious_only:
        cards = [c for c in cards if _is_suspicious(c)]
    if top_k is not None:
        cards = cards[:top_k]

    risk_map = {}
    if RISK_CSV.exists():
        rdf = pd.read_csv(RISK_CSV)
        risk_map = dict(zip(rdf["application_id"], rdf["risk_score_v3"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plotly files pre-rendered only in the offline "file" mode.
    ring_files = generate_ego_rings(cards, risk_map) if ring_mode == "file" else {}

    def _ring_href(app_id: str) -> str | None:
        if ring_mode == "api":
            return f"/v3/monitoring/{app_id}/ring"     # lazy, computed on click
        if ring_mode == "file":
            return ring_files.get(app_id)
        return None                                    # "none"

    filenames = []
    for rank, card in enumerate(cards, start=1):
        fn = f"card_{rank:03d}_{card['application_id']}.html"
        (OUT_DIR / fn).write_text(
            _card_page(card, rank, risk_map, _ring_href(card["application_id"])), encoding="utf-8")
        filenames.append(fn)

    (OUT_DIR / "index.html").write_text(_index_page(cards, filenames), encoding="utf-8")
    print(f"[xai_card] Wrote {len(cards)} suspicious card(s) + index -> {OUT_DIR}  "
          f"(ring_mode={ring_mode})")
    return len(cards)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=None, help="cap number of cards (default: all suspicious)")
    ap.add_argument("--all", action="store_true", help="render every card, not just suspicious ones")
    ap.add_argument("--ring-mode", choices=["api", "file", "none"], default="api",
                    help="api=lazy link (default), file=pre-render Plotly offline, none=no 3D link")
    args = ap.parse_args()
    render_cards(top_k=args.top_k, suspicious_only=not args.all, ring_mode=args.ring_mode)
