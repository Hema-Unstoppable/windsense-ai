/* ═══════════════════════════════════════════════════════════════════
   WindSense AI — Live Data Layer
   Connects dashboard.html to the FastAPI backend. If the API is
   reachable it renders REAL ML output; if not, it leaves the built-in
   demo content in place and shows a "Demo data" badge.

   It augments the existing dashboard script by overriding a few global
   functions (showApp, showScreen, renderCharts) — no markup changes.
   ═══════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  // ── Config ────────────────────────────────────────────────────────
  const API_ROOT = window.WS_API_ROOT || "http://localhost:8000";
  const API = API_ROOT + "/api";

  // ── State ─────────────────────────────────────────────────────────
  let LIVE = false;
  let FLEET = null;          // /dashboard/fleet_overview
  let TURBINES = [];         // /turbines
  let MODEL = null;          // /ml/model
  let selectedTurbineId = null;
  const liveCharts = {};

  // keep the demo chart renderer so we can fall back to it
  const _demoRenderCharts = window.renderCharts;
  const _demoBuildHeatmap = window.buildHeatmap;

  // resolve the Supabase client from the inline script's global binding
  function getSB() {
    try { if (typeof sb !== "undefined" && sb) return sb; } catch (e) { /* TDZ */ }
    return window.sb || null;
  }

  // ── Auth headers ──────────────────────────────────────────────────
  // Sends the logged-in Supabase identity so the backend serves only this
  // tenant's data: the email (email_header mode) + bearer token (jwt mode).
  async function authHeader() {
    const h = {};
    try {
      const client = getSB();
      if (client && client.auth) {
        const { data } = await client.auth.getSession();
        if (data && data.session) {
          h.Authorization = "Bearer " + data.session.access_token;
          const email = data.session.user && data.session.user.email;
          if (email) h["X-WS-User-Email"] = email;
        }
      }
    } catch (e) { /* ignore */ }
    return h;
  }

  async function apiGet(path) {
    const r = await fetch(API + path, { headers: await authHeader() });
    if (!r.ok) throw new Error("GET " + path + " → " + r.status);
    return r.json();
  }
  async function apiPost(path, body) {
    const r = await fetch(API + path, {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, await authHeader()),
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error("POST " + path + " → " + r.status);
    return r.json();
  }

  // ── Helpers ───────────────────────────────────────────────────────
  const fmtEur = (n) => "€" + Math.round(n || 0).toLocaleString("en-IE");
  const badgeLetter = (rc) => ({ CRITICAL: "C", HIGH: "H", MEDIUM: "M", LOW: "L" }[rc] || "L");
  const cellState = (rc) => ({ CRITICAL: "crit", HIGH: "warn", MEDIUM: "warn", LOW: "ok" }[rc] || "offline");
  const compColor = (s) => (s >= 70 ? "var(--green)" : s >= 45 ? "var(--amber)" : "var(--red)");
  const sevClass = (rc) => ({ CRITICAL: "crit", HIGH: "warn", MEDIUM: "warn", LOW: "ok" }[rc] || "ok");
  const statusWord = (rc) => ({ CRITICAL: "Critical", HIGH: "Warning", MEDIUM: "Warning", LOW: "Operational" }[rc] || "Operational");

  function setLivePill(live, note) {
    const p = document.querySelector(".live-pill");
    if (!p) return;
    if (live) {
      p.style.background = "var(--green-dim)";
      p.style.color = "var(--green)";
      p.innerHTML = '<div class="live-dot"></div>Live data · API connected';
    } else {
      p.style.background = "rgba(217,119,6,.1)";
      p.style.color = "var(--amber)";
      p.style.border = "1px solid rgba(217,119,6,.25)";
      p.innerHTML = '<div class="live-dot" style="background:#D97706"></div>Demo data · ' + (note || "API offline");
    }
  }

  function toast(msg) {
    let t = document.getElementById("ws-toast");
    if (!t) {
      t = document.createElement("div");
      t.id = "ws-toast";
      t.style.cssText =
        "position:fixed;bottom:24px;right:24px;z-index:2000;background:var(--navy);color:#fff;" +
        "padding:12px 18px;border-radius:8px;font-size:13px;font-weight:600;box-shadow:0 8px 30px rgba(0,0,0,.3);" +
        "border:1px solid rgba(13,179,158,.4);transition:opacity .3s;font-family:var(--f-main)";
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.style.opacity = "1";
    clearTimeout(t._timer);
    t._timer = setTimeout(() => (t.style.opacity = "0"), 2600);
  }

  // ══════════════════════════════════════════════════════════════════
  //  BOOT — override showApp to attempt a live connection
  // ══════════════════════════════════════════════════════════════════
  window.showApp = async function () {
    const overlay = document.getElementById("loading-overlay");
    try {
      const h = await fetch(API + "/health");
      if (!h.ok) throw new Error("health " + h.status);
      LIVE = true;
      setLivePill(true);
      await loadLiveData();
    } catch (e) {
      console.warn("[WindSense] API not reachable — showing demo data.", e.message);
      LIVE = false;
      setLivePill(false);
      if (_demoBuildHeatmap) _demoBuildHeatmap();   // demo heatmap
    }
    if (overlay) {
      overlay.classList.add("hidden");
      setTimeout(() => (overlay.style.display = "none"), 400);
    }
    // honour ?screen= from sidebar links on other pages (e.g. validation.html)
    try {
      const scr = new URLSearchParams(location.search).get("screen");
      if (scr) {
        const link = document.querySelector('.sb-nav a[onclick*="' + scr + '"]');
        if (window.showScreen) window.showScreen(scr, link);
      }
    } catch (e) { /* ignore */ }
  };

  async function loadLiveData() {
    [FLEET, TURBINES, MODEL] = await Promise.all([
      apiGet("/dashboard/fleet_overview?top=20"),
      apiGet("/turbines"),
      apiGet("/ml/model").catch(() => null),
    ]);
    renderFleetKPIs(FLEET.kpis);
    renderFleetAlerts(FLEET.risk_ranking);
    renderHeatmap(TURBINES);
    renderRiskQueue(FLEET.risk_ranking, FLEET.kpis);
    renderPredictionsKPIs();
    renderPredictionsList();
    renderCerts();
    wireTopbar();
    updateBreadcrumb("fleet");
    // auto-select first turbine for the SHAP panel
    if (TURBINES.length) selectTurbine(TURBINES[0].id);
  }

  // ── SCREEN 1 · Fleet KPIs ─────────────────────────────────────────
  function renderFleetKPIs(k) {
    const strip = document.querySelector("#screen-fleet .kpi-strip");
    if (!strip) return;
    strip.innerHTML = `
      <div class="kpi-card kpi-border-green"><div class="kpi-label">Fleet Health Score</div>
        <div class="kpi-val">${Math.round(k.fleet_health_score)}%</div>
        <div class="kpi-change up">${k.turbines_monitored} turbines monitored</div></div>
      <div class="kpi-card kpi-border-red"><div class="kpi-label">Critical Alerts</div>
        <div class="kpi-val">${k.critical_alerts}</div>
        <div class="kpi-change dn">Immediate action required</div></div>
      <div class="kpi-card kpi-border-amber"><div class="kpi-label">Active Alerts</div>
        <div class="kpi-val">${k.total_active_alerts}</div>
        <div class="kpi-change warn">Across the fleet</div></div>
      <div class="kpi-card kpi-border-teal"><div class="kpi-label">Turbines Operating</div>
        <div class="kpi-val">${k.turbines_operating}</div>
        <div class="kpi-change up">${k.turbines_offline} offline · ${k.fleet_availability}% avail.</div></div>
      <div class="kpi-card"><div class="kpi-label">Financial Exposure</div>
        <div class="kpi-val">${fmtEur(k.financial_exposure_eur)}</div>
        <div class="kpi-change warn">&Sigma; Expected Loss (&euro;)</div></div>`;
  }

  function renderFleetAlerts(rows) {
    const list = document.querySelector("#screen-fleet .alert-list");
    if (!list) return;
    const top = rows.slice(0, 6);
    list.innerHTML = top.map((r) => `
      <div class="alert-row" onclick="WS.openTurbine(${r.turbine_id})">
        <div class="rbadge ${badgeLetter(r.risk_class)}">${badgeLetter(r.risk_class)}</div>
        <div class="alert-info">
          <div class="alert-title">${r.component || "Anomaly"} risk · ${r.turbine_name}</div>
          <div class="alert-sub">${r.site_name} · fault prob ${(r.fault_probability * 100).toFixed(0)}% · RUL ${Math.round(r.rul_days || 0)}d</div>
        </div>
        <div class="alert-rpn"><span class="rpn-val">${Math.round(r.rpn)}</span><span class="rpn-lbl">RPN</span></div>
      </div>`).join("");
  }

  // ── SCREEN 1 · Heatmap (real turbines) ────────────────────────────
  function renderHeatmap(turbines) {
    const grid = document.getElementById("heatmap");
    if (!grid) return;
    grid.innerHTML = "";
    let crit = 0;
    turbines.forEach((t) => {
      const st = cellState(t.risk_class);
      if (st === "crit") crit++;
      const cell = document.createElement("div");
      cell.className = "hcell " + st;
      cell.textContent = t.name.replace("WT-", "T");
      cell.title = `${t.name} (${t.oem} ${t.model}) — ${t.risk_class || "n/a"} · health ${t.health_score ?? "?"}`;
      cell.onclick = () => openTurbine(t.id);
      grid.appendChild(cell);
    });
    const badge = document.getElementById("badge-alerts");
    if (badge) badge.textContent = crit;
  }

  // ── SCREEN 3 · Risk queue ─────────────────────────────────────────
  function renderRiskQueue(rows, kpis) {
    const tbody = document.querySelector("#screen-riskqueue tbody");
    if (tbody) {
      tbody.innerHTML = rows.map((r) => {
        const ap = r.action_priority || statusWord(r.risk_class);
        const apCol = ap === "High" ? "var(--red)" : ap === "Medium" ? "var(--amber)" : "var(--green)";
        const cb = r.cost_breakdown || {};
        const tip = cb.total ? `Consequence €${(cb.total).toLocaleString()} = parts €${cb.parts.toLocaleString()} + labour €${cb.labour.toLocaleString()} + crane €${cb.crane.toLocaleString()} + lost revenue €${cb.lost_revenue.toLocaleString()} (${cb.downtime_days}d). EEL = P(${(r.fault_probability*100).toFixed(0)}%) × consequence.` : "";
        const rulCol = (r.rul_days||0) <= 7 ? "var(--red)" : (r.rul_days||0) <= 21 ? "var(--amber)" : "var(--g500)";
        return `<tr>
          <td style="font-family:var(--f-mono);color:var(--g400)">#${r.rank}</td>
          <td class="tip" data-tip="${tip}" style="font-family:var(--f-mono);font-weight:700;color:var(--navy);cursor:help">${fmtEur(r.expected_loss_eur != null ? r.expected_loss_eur : r.financial_exposure_eur)}</td>
          <td><span class="sdot" style="color:${apCol}"><span style="width:7px;height:7px;border-radius:50%;background:${apCol};display:inline-block;margin-right:5px"></span>${ap}</span></td>
          <td><strong>${r.turbine_name}</strong> <span style="font-family:var(--f-mono);font-size:10px;color:var(--g400)">${turbineMeta(r.turbine_id)}</span></td>
          <td>${r.component || "—"}</td>
          <td style="font-family:var(--f-mono);font-size:12px">${(r.fault_probability * 100).toFixed(0)}%</td>
          <td style="font-family:var(--f-mono);font-size:12px;color:${rulCol};font-weight:600">${Math.round(r.rul_days || 0)}d</td>
          <td style="font-family:var(--f-mono);font-size:10px;color:var(--g400)">${r.iso14224_code || "—"}</td>
          <td><button class="cert-btn" style="padding:5px 10px;font-size:11px" onclick="WS.openTurbine(${r.turbine_id})">View</button></td>
        </tr>`;
      }).join("");
    }
    const strip = document.querySelector("#screen-riskqueue .kpi-strip");
    if (strip && kpis) {
      const totalEel = rows.reduce((s, r) => s + (r.expected_loss_eur != null ? r.expected_loss_eur : (r.financial_exposure_eur||0)), 0);
      const apHigh = rows.filter((r) => (r.action_priority||"") === "High").length;
      const apMed = rows.filter((r) => (r.action_priority||"") === "Medium").length;
      const apLow = rows.filter((r) => (r.action_priority||"") === "Low").length;
      const topEel = rows.length ? (rows[0].expected_loss_eur != null ? rows[0].expected_loss_eur : rows[0].financial_exposure_eur) : 0;
      strip.innerHTML = `
        <div class="kpi-card kpi-border-green"><div class="kpi-label">Fleet Expected Loss (€)</div><div class="kpi-val">${fmtEur(totalEel)}</div><div class="kpi-change dn">Σ derived EEL</div></div>
        <div class="kpi-card kpi-border-red"><div class="kpi-label">Action Priority · High</div><div class="kpi-val">${apHigh}</div><div class="kpi-change dn">AIAG-VDA</div></div>
        <div class="kpi-card kpi-border-amber"><div class="kpi-label">Action Priority · Medium</div><div class="kpi-val">${apMed}</div><div class="kpi-change warn">Schedule</div></div>
        <div class="kpi-card kpi-border-teal"><div class="kpi-label">Action Priority · Low</div><div class="kpi-val">${apLow}</div><div class="kpi-change up">Monitor</div></div>
        <div class="kpi-card"><div class="kpi-label">Top single exposure</div><div class="kpi-val">${fmtEur(topEel)}</div><div class="kpi-change dn">${rows.length?rows[0].turbine_name:''}</div></div>`;
    }
  }

  function turbineMeta(id) {
    const t = TURBINES.find((x) => x.id === id);
    return t ? `${t.oem} ${t.model}` : "—";
  }

  // ── SCREEN 2 · Predictions list + KPIs ────────────────────────────
  function renderPredictionsKPIs() {
    const strip = document.querySelector("#screen-predictions .kpi-strip");
    if (!strip) return;
    const acc = MODEL && MODEL.loaded_model && MODEL.loaded_model.metrics
      ? (MODEL.loaded_model.metrics.roc_auc_macro_ovr || MODEL.loaded_model.metrics.accuracy) : null;
    const ver = MODEL && MODEL.loaded_model ? MODEL.loaded_model.version : "—";
    const ranked = FLEET.risk_ranking;
    const active = ranked.filter((r) => r.risk_class !== "LOW").length;
    const avgRul = ranked.length ? Math.round(ranked.reduce((s, r) => s + (r.rul_days || 0), 0) / ranked.length) : 0;
    const highConf = ranked.filter((r) => r.fault_probability >= 0.7).length;
    strip.innerHTML = `
      <div class="kpi-card kpi-border-teal"><div class="kpi-label">Model ROC-AUC</div><div class="kpi-val">${acc ? (acc * 100).toFixed(1) + "%" : "—"}</div><div class="kpi-change up">XGBoost</div></div>
      <div class="kpi-card kpi-border-red"><div class="kpi-label">Active Predictions</div><div class="kpi-val">${active}</div><div class="kpi-change warn">${highConf} high confidence</div></div>
      <div class="kpi-card kpi-border-amber"><div class="kpi-label">Avg. RUL</div><div class="kpi-val">${avgRul}d</div><div class="kpi-change up">Across flagged units</div></div>
      <div class="kpi-card"><div class="kpi-label">Turbines Scored</div><div class="kpi-val">${ranked.length}</div><div class="kpi-change up">Whole fleet</div></div>
      <div class="kpi-card kpi-border-green"><div class="kpi-label">Model Version</div><div class="kpi-val" style="font-size:16px">${ver}</div><div class="kpi-change up">Active</div></div>`;
  }

  function renderPredictionsList() {
    const container = document.querySelector('#screen-predictions div[style*="overflow-y"]');
    if (!container) return;
    const ranked = FLEET.risk_ranking;
    container.innerHTML = ranked.map((r) => {
      const L = badgeLetter(r.risk_class);
      return `<div class="pred-card" id="predcard-${r.turbine_id}" onclick="WS.selectTurbine(${r.turbine_id})">
        <div class="pred-head">
          <div class="rbadge ${L}">${L}</div>
          <div><div class="pred-name">${r.turbine_name} · ${r.component || "Anomaly"}</div>
               <div class="pred-rul">${turbineMeta(r.turbine_id)} · RUL: ${Math.round(r.rul_days || 0)} days</div></div>
          <span class="pred-conf" title="Calibrated fault probability">${(r.fault_probability * 100).toFixed(0)}% calib.</span>
        </div>
        <div class="pred-body" style="display:none"></div>
      </div>`;
    }).join("");
  }

  // ── SCREEN 2 · select a turbine → health_summary → SHAP + components
  async function selectTurbine(turbineId) {
    selectedTurbineId = turbineId;
    document.querySelectorAll("#screen-predictions .pred-card").forEach((c) => {
      c.classList.remove("selected");
      const b = c.querySelector(".pred-body");
      if (b) b.style.display = "none";
    });
    const card = document.getElementById("predcard-" + turbineId);
    if (card) { card.classList.add("selected"); }

    let hs;
    try { hs = await apiGet(`/turbines/${turbineId}/health_summary`); }
    catch (e) { console.warn(e); return; }

    if (card) {
      const b = card.querySelector(".pred-body");
      if (b) {
        b.style.display = "block";
        b.innerHTML = `<div style="font-size:11px;color:var(--g500);font-family:var(--f-mono)">` +
          `Health ${Math.round(hs.overall_health_score)}/100 · ${hs.predicted_component || "Normal"} · ` +
          `conf ${(hs.confidence * 100).toFixed(0)}% · SHAP breakdown shown in panel →</div>`;
      }
    }

    // SHAP panel
    const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setText("shap-title", `SHAP Explainability · ${hs.turbine_name}`);
    // calibrated probability, transparently showing the raw (pre-calibration) value
    const calP = Math.round((hs.fault_probability || 0) * 100);
    let probLine;
    if (hs.calibrated && hs.raw_fault_probability != null) {
      const rawP = Math.round(hs.raw_fault_probability * 100);
      probLine = `${hs.predicted_component || "Normal"} · fault prob ${calP}% (calibrated, raw ${rawP}%)`;
    } else {
      probLine = `${hs.predicted_component || "Normal"} · fault prob ${calP}%`;
    }
    setText("shap-sub", probLine);
    setText("shap-narrative", hs.narrative);

    // calibration badge in the SHAP card header
    const titleEl = document.getElementById("shap-title");
    if (titleEl) {
      let badge = document.getElementById("calib-badge");
      if (!badge) {
        badge = document.createElement("span");
        badge.id = "calib-badge";
        badge.style.cssText = "margin-left:8px;font-family:var(--f-mono);font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;vertical-align:middle";
        titleEl.appendChild(badge);
      }
      if (hs.calibrated) {
        badge.textContent = "✓ CALIBRATED";
        badge.style.background = "rgba(22,163,74,.15)"; badge.style.color = "var(--green)";
        badge.title = "Probability calibrated via isotonic regression so it reflects real failure frequency.";
      } else {
        badge.textContent = "RAW PROB";
        badge.style.background = "rgba(217,119,6,.15)"; badge.style.color = "var(--amber)";
        badge.title = "Uncalibrated probability — may be over-confident.";
      }
    }

    const narrative = document.getElementById("shap-narrative");
    if (narrative) {
      const body = narrative.parentElement;
      body.querySelectorAll(".shap-row, .shap-note").forEach((n) => n.remove());
      hs.shap_explanation.forEach((f) => {
        const row = document.createElement("div");
        row.className = "shap-row";
        const w = Math.min(100, f.pct * 2);
        // amber + ⚠ when the driver is NOT a physically expected signal for the mode
        const unexpected = f.expected === false;
        const fill = unexpected ? "#D97706" : "linear-gradient(90deg,var(--teal),var(--teal-b))";
        const flag = unexpected ? ' <span title="Not a physically expected driver for this failure mode" style="color:#D97706">&#9888;</span>' : "";
        row.innerHTML = `<div class="shap-lbl">${f.feature}${flag}</div>
          <div class="shap-track"><div class="shap-fill" style="width:${w}%;background:${fill}"></div></div>
          <div class="shap-pct">${f.pct.toFixed(0)}%</div>`;
        body.appendChild(row);
      });
      // decision-support recommendation (traceable, edge-case aware)
      const rec = hs.recommendation;
      if (rec) {
        const uCol = rec.urgency === "Immediate" || rec.urgency === "High" ? "#f87171"
                   : rec.urgency === "Medium" ? "#fbbf24" : "#4ade80";
        const flags = [];
        if (rec.sensor_fault_suspected) flags.push('<span style="color:#fbbf24">&#9888; possible sensor fault</span>');
        if (rec.model_conflict) flags.push('<span style="color:#fbbf24">&#9888; signals disagree</span>');
        const note = document.createElement("div");
        note.className = "shap-note";
        note.style.cssText = "margin-top:14px;padding:14px;border-radius:8px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-left:3px solid " + uCol;
        note.innerHTML =
          `<div style="display:flex;justify-content:space-between;align-items:center">
             <span style="font-family:var(--f-mono);font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:rgba(255,255,255,.4)">Decision Support · ${rec.classification}</span>
             <span style="font-family:var(--f-mono);font-size:11px;font-weight:700;color:${uCol}">${rec.urgency.toUpperCase()}</span>
           </div>
           <div style="font-size:14px;font-weight:700;color:#fff;margin:6px 0 4px">${rec.action_label} — ${rec.headline}</div>
           ${flags.length?`<div style="font-size:12px;margin-bottom:6px">${flags.join(' · ')}</div>`:''}
           <ul style="margin:6px 0 0 16px;padding:0;font-size:12px;color:rgba(255,255,255,.65);line-height:1.6">
             ${rec.rationale.map(r=>`<li>${r}</li>`).join('')}
           </ul>
           <div style="margin-top:10px;font-size:10px;color:rgba(255,255,255,.4);font-style:italic">${rec.disclaimer}</div>`;
        body.appendChild(note);
      }

      // explainability integrity banner
      const ex = hs.explainability;
      if (ex) {
        const note = document.createElement("div");
        note.className = "shap-note";
        const score = ex.alignment_score != null ? Math.round(ex.alignment_score * 100) + "%" : "n/a";
        const col = ex.alignment_score == null ? "var(--g400)"
                  : ex.alignment_score >= 0.6 ? "var(--green)"
                  : ex.alignment_score >= 0.3 ? "var(--amber)" : "var(--red)";
        let extra = "";
        if (ex.unexpected_top) extra += `<div style="margin-top:6px;color:#fbbf24">&#9888; ${ex.unexpected_top}</div>`;
        if (ex.caveat) extra += `<div style="margin-top:6px;color:rgba(255,255,255,.55)">${ex.caveat}</div>`;
        note.style.cssText = "margin-top:14px;padding:12px;border-radius:8px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);font-size:12px;line-height:1.6;color:rgba(255,255,255,.7)";
        note.innerHTML = `<span style="font-family:var(--f-mono);font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:rgba(255,255,255,.4)">Physical alignment</span>
          <span style="float:right;font-weight:700;color:${col}">${score}</span>
          <div style="margin-top:4px">${ex.verdict}</div>${extra}`;
        body.appendChild(note);
      }
    }

    // Component health card
    setText("comp-title", `Component Health · ${hs.turbine_name} · ${hs.oem || ""} ${hs.model || ""}`);
    const compTitle = document.getElementById("comp-title");
    if (compTitle) {
      const body = compTitle.closest(".card").querySelector(".card-body");
      const overall = Math.round(hs.overall_health_score);
      body.innerHTML = hs.component_health.map((c) => {
        const col = compColor(c.health_score);
        return `<div class="comp-row"><div class="comp-name">${c.component}</div>
          <div class="comp-track"><div class="comp-fill" style="width:${c.health_score}%;background:${col}"></div></div>
          <div class="comp-score" style="color:${col}">${Math.round(c.health_score)}</div></div>`;
      }).join("") +
        `<div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--g100)">
          <div style="font-size:11px;color:var(--g400);font-family:var(--f-mono);margin-bottom:6px">Overall Health Score</div>
          <div style="display:flex;align-items:center;gap:12px">
            <div style="font-family:var(--f-mono);font-size:28px;font-weight:700;color:${compColor(overall)}">${overall}</div>
            <div style="font-size:12px;color:var(--g500)">${hs.classification}<br/>RUL ${Math.round(hs.rul_days || 0)} days · ${hs.predicted_component || "Normal"}</div>
          </div></div>`;
    }
  }

  function openTurbine(turbineId) {
    const link = document.querySelector('.sb-nav a[onclick*="predictions"]');
    window.showScreen("predictions", link);
    selectTurbine(turbineId);
  }

  // ── SCREEN 4 · Certificates ───────────────────────────────────────
  async function renderCerts() {
    const container = document.querySelector('#screen-certs div[style*="padding:16px"]');
    if (!container) return;
    container.innerHTML = '<div style="font-size:13px;color:var(--g400);padding:8px">Generating certificates…</div>';
    try {
      const certs = await Promise.all(
        TURBINES.map((t) => apiGet(`/reports/asset_health_certificate?turbine_id=${t.id}`).catch(() => null))
      );
      container.innerHTML = certs.filter(Boolean).map((c) => {
        const cls = c.overall_health_score >= 85 ? "" : c.overall_health_score >= 65 ? "amber" : "red";
        const comps = ["Gearbox", "Generator", "Main Bearing"];
        const rows = comps.map((name) => {
          const s = c.component_scores[name] ?? 0;
          return `<div class="comp-row"><div class="comp-name">${name}</div>
            <div class="comp-track"><div class="comp-fill" style="width:${s}%;background:${compColor(s)}"></div></div>
            <div class="comp-score" style="color:${compColor(s)}">${Math.round(s)}</div></div>`;
        }).join("");
        const hash8 = (c.content_hash || "").slice(0, 12);
        return `<div class="cert-card">
          <div class="cert-header"><div class="cert-score ${cls}">${Math.round(c.overall_health_score)}</div>
            <div class="cert-info"><strong>${c.turbine_name} · ${c.oem} ${c.model} · ${c.site_name}</strong>
            <span>${c.classification} · ${c.certificate_ref}</span></div></div>
          ${rows}
          <div style="margin-top:8px;font-size:10px;color:var(--g400);font-family:var(--f-mono)">
            Engineering decision-support · coverage ${c.data_coverage_pct ?? "—"}% · 🔒 sealed ${hash8}…
          </div>
          <button class="cert-btn" onclick="WS.downloadCert('${c.certificate_ref}', ${c.turbine_id})">
            <svg width="12" height="12" fill="none" viewBox="0 0 12 12"><path d="M6 1v7M3 5l3 3 3-3M2 10h8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>Download PDF</button>
        </div>`;
      }).join("");
    } catch (e) {
      container.innerHTML = '<div style="font-size:13px;color:var(--red);padding:8px">Could not load certificates.</div>';
    }
  }

  async function downloadCert(ref, turbineId) {
    toast("Generating certificate PDF…");
    try {
      const r = await fetch(`${API}/reports/asset_health_certificate.pdf?turbine_id=${turbineId}`,
                            { headers: await authHeader() });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = (ref || "asset_health_certificate") + ".pdf";
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      toast("Downloaded " + (ref || "certificate") + ".pdf");
    } catch (e) {
      toast("PDF download failed: " + e.message);
    }
  }

  // ── SCREEN 3 · SCADA charts (real timeseries) ─────────────────────
  async function renderChartsLive() {
    if (!FLEET || !FLEET.risk_ranking.length) return;
    const tid = selectedTurbineId || FLEET.risk_ranking[0].turbine_id;
    const tname = (TURBINES.find((t) => t.id === tid) || {}).name || "turbine";
    let ts;
    try {
      ts = await apiGet(`/turbines/${tid}/timeseries?limit=1500&signals=power_kw,wind_speed,gearbox_oil_temp,gen_bearing_de_temp,rotor_rpm`);
    } catch (e) { console.warn(e); return; }

    const pts = ts.points;
    const labels = pts.map((p) => new Date(p.ts).toLocaleDateString("en-IE", { month: "short", day: "numeric" }));
    const get = (k) => pts.map((p) => p.values[k]);
    const oil = get("gearbox_oil_temp");
    const oilBase = oil.filter((v) => v != null);
    const mean = oilBase.length ? oilBase.reduce((a, b) => a + b, 0) / oilBase.length : 0;

    const def = {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { mode: "index", intersect: false } },
      scales: {
        x: { grid: { color: "rgba(0,0,0,.05)" }, ticks: { font: { size: 10 }, maxTicksLimit: 8, color: "#94A3B8" } },
        y: { grid: { color: "rgba(0,0,0,.05)" }, ticks: { font: { size: 10 }, color: "#94A3B8" } },
      },
      elements: { point: { radius: 0, hoverRadius: 4 } },
    };
    const mk = (id, cfg) => { if (liveCharts[id]) liveCharts[id].destroy(); liveCharts[id] = new Chart(document.getElementById(id), cfg); };

    setCardTitle("chart-oil-temp", `Gearbox Oil Temperature · ${tname}`, "Real SCADA · 10-min resolution");
    mk("chart-oil-temp", {
      type: "line",
      data: { labels, datasets: [
        { label: "Oil Temp (°C)", data: oil, borderColor: "#DC2626", backgroundColor: "rgba(220,38,38,.08)", fill: true, tension: .3, borderWidth: 2 },
        { label: "Baseline", data: labels.map(() => mean.toFixed(1)), borderColor: "rgba(148,163,184,.5)", borderDash: [4, 4], borderWidth: 1, fill: false },
      ] },
      options: Object.assign({}, def, { scales: Object.assign({}, def.scales, { y: Object.assign({}, def.scales.y, { title: { display: true, text: "°C" } }) }) }),
    });

    setCardTitle("chart-vibration", `Generator Bearing Temp · ${tname}`, "Drive-end bearing · real SCADA");
    mk("chart-vibration", {
      type: "line",
      data: { labels, datasets: [{ label: "Gen Bearing DE (°C)", data: get("gen_bearing_de_temp"), borderColor: "#D97706", backgroundColor: "rgba(217,119,6,.08)", fill: true, tension: .3, borderWidth: 2 }] },
      options: Object.assign({}, def, { scales: Object.assign({}, def.scales, { y: Object.assign({}, def.scales.y, { title: { display: true, text: "°C" } }) }) }),
    });

    setCardTitle("chart-power", `Power Curve · ${tname}`, "Power vs wind speed · real SCADA");
    const scatter = pts.map((p) => ({ x: p.values.wind_speed, y: p.values.power_kw })).filter((d) => d.x != null && d.y != null);
    mk("chart-power", {
      type: "scatter",
      data: { datasets: [{ label: "Power", data: scatter, backgroundColor: "rgba(13,179,158,.45)", pointRadius: 3 }] },
      options: Object.assign({}, def, {
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (d) => `Wind ${d.parsed.x} m/s · ${Math.round(d.parsed.y)} kW` } } },
        scales: { x: Object.assign({}, def.scales.x, { title: { display: true, text: "Wind Speed (m/s)" } }), y: Object.assign({}, def.scales.y, { title: { display: true, text: "Power (kW)" } }) },
      }),
    });

    // Availability per day from status_type_id (0/1 = operating)
    setCardTitle("chart-availability", `Availability · ${tname}`, "Daily, derived from SCADA status");
    const byDay = {};
    pts.forEach((p) => {
      const d = new Date(p.ts).toLocaleDateString("en-IE", { month: "short", day: "numeric" });
      byDay[d] = byDay[d] || { up: 0, n: 0 };
      byDay[d].n++;
      if (p.status_type_id === 0 || p.status_type_id === 1) byDay[d].up++;
    });
    const days = Object.keys(byDay).slice(-14);
    const avail = days.map((d) => +(100 * byDay[d].up / byDay[d].n).toFixed(1));
    mk("chart-availability", {
      type: "bar",
      data: { labels: days, datasets: [{ label: "Availability %", data: avail, backgroundColor: avail.map((v) => v >= 95 ? "rgba(22,163,74,.6)" : "rgba(217,119,6,.6)"), borderRadius: 4 }] },
      options: Object.assign({}, def, { scales: { x: def.scales.x, y: Object.assign({}, def.scales.y, { min: 0, max: 100, title: { display: true, text: "%" } }) } }),
    });
  }

  function setCardTitle(canvasId, title, sub) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const card = canvas.closest(".card");
    if (!card) return;
    const t = card.querySelector(".card-title"); if (t) t.textContent = title;
    const s = card.querySelector(".card-sub"); if (s) s.textContent = sub;
  }

  // override renderCharts: live when connected, else demo
  window.renderCharts = function () {
    if (LIVE) renderChartsLive();
    else if (_demoRenderCharts) _demoRenderCharts();
  };

  // ── Topbar wiring (Run ML Analysis) ───────────────────────────────
  function wireTopbar() {
    const btn = document.querySelector(".topbar-btn.primary");
    if (btn) {
      btn.onclick = async function () {
        const orig = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = "Running inference…";
        try {
          const res = await apiPost("/ml/run_inference", { store: true });
          toast(`Inference complete · ${res.predictions_written} predictions · model ${res.model_version}`);
          await loadLiveData();
        } catch (e) {
          toast("Inference failed: " + e.message);
        } finally {
          btn.disabled = false;
          btn.innerHTML = orig;
        }
      };
    }
  }

  // ── Breadcrumb with live counts ───────────────────────────────────
  function updateBreadcrumb(screen) {
    const bc = document.getElementById("page-breadcrumb");
    if (!bc || !FLEET) return;
    const site = TURBINES[0] ? TURBINES[0].site_name : "Fleet";
    bc.textContent = `WindSense AI · ${site} · ${FLEET.kpis.turbines_monitored} turbines · LIVE`;
  }

  // override showScreen to refresh live charts + breadcrumb
  const _origShowScreen = window.showScreen;
  window.showScreen = function (id, linkEl) {
    if (_origShowScreen) _origShowScreen(id, linkEl);
    if (!LIVE) return;
    if (id === "scada") setTimeout(renderChartsLive, 60);
    updateBreadcrumb(id);
  };

  // ── Public hooks for inline onclick handlers ──────────────────────
  window.WS = {
    openTurbine: openTurbine,
    selectTurbine: selectTurbine,
    downloadCert: downloadCert,
    refresh: loadLiveData,
    apiRoot: API_ROOT,
  };

  console.log("[WindSense] live layer loaded. API:", API);
})();
