"""
═══════════════════════════════════════════════════════════════════════
 ASSET HEALTH CERTIFICATE — PDF renderer (fpdf2)
═══════════════════════════════════════════════════════════════════════
Produces a downloadable, forensic-grade PDF:
  asset details · data provenance · model traceability · component scores +
  RUL · limitations & known miss-rate · named certifying engineer · validity
  · and a SHA-256 content seal that makes the document tamper-evident.

Framed as an ENGINEERING DECISION-SUPPORT assessment (not an insurance
instrument), per the liability review.
"""
from __future__ import annotations

from datetime import datetime
from fpdf import FPDF

TEAL = (10, 140, 124)
NAVY = (9, 25, 42)
GREY = (100, 116, 139)
LIGHT = (241, 245, 249)
RED = (220, 38, 38)
AMBER = (217, 119, 6)
GREEN = (22, 163, 74)


def _a(text) -> str:
    """ASCII-sanitise (core PDF fonts are latin-1 only)."""
    if text is None:
        return ""
    s = str(text)
    repl = {"€": "EUR ", "—": " - ", "–": "-", "→": "->", "≈": "~", "×": "x",
            "’": "'", "“": '"', "”": '"', "·": "-", "Σ": "Sum ", "≤": "<=", "≥": ">=",
            "⚠": "!", "✓": "OK", "°": "deg"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


class _PDF(FPDF):
    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 26, "F")
        self.set_xy(12, 7)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 15)
        self.cell(120, 6, "WindSense AI", ln=0)
        self.set_font("Helvetica", "", 9)
        self.set_xy(12, 15)
        self.set_text_color(13, 179, 158)
        self.cell(120, 5, "Asset Health Assessment - Engineering Decision Support", ln=0)
        self.set_xy(-70, 9)
        self.set_text_color(180, 190, 200)
        self.set_font("Helvetica", "", 8)
        self.cell(58, 5, _a(self._ref or ""), align="R", ln=0)
        self.set_y(32)

    def footer(self):
        self.set_y(-16)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GREY)
        self.multi_cell(0, 3.5,
            _a(f"Tamper-evident SHA-256 seal: {self._hash}\n"
               f"Generated {datetime.utcnow():%Y-%m-%d %H:%M UTC} - Decision support only; "
               f"not an insurance or lending instrument. Verify the seal against the source data + "
               f"model version + SHAP vector."), align="C")


def _section(pdf, title):
    pdf.ln(2)
    pdf.set_fill_color(*LIGHT)
    pdf.set_text_color(*NAVY)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, _a("  " + title), ln=1, fill=True)
    pdf.ln(1)


def _kv(pdf, k, v, kw=55):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(kw, 5.5, _a(k), ln=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.5, _a(v), ln=1)


def render_certificate_pdf(cert: dict, sustainability: dict, financial: dict,
                           compliance: list) -> bytes:
    t = cert["turbine"]
    site = cert["site"]
    pred = cert["prediction"]
    pdf = _PDF(format="A4")
    pdf._ref = cert["certificate_ref"]
    pdf._hash = cert.get("content_hash", "")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_left_margin(12); pdf.set_right_margin(12)

    # ── Decision-support banner ──
    pdf.set_fill_color(255, 247, 237)
    pdf.set_draw_color(*AMBER)
    pdf.set_text_color(146, 64, 14)
    pdf.set_font("Helvetica", "B", 8)
    pdf.multi_cell(0, 4.5, _a(
        "ENGINEERING DECISION-SUPPORT ASSESSMENT - This is not an insurance instrument, "
        "credit/lending document, or guarantee of future performance. Requires review and "
        "sign-off by a certified reliability engineer before any action."), border=1, fill=True)

    # ── Overall result ──
    _section(pdf, "Overall Assessment")
    score = int(cert["overall_health_score"])
    col = GREEN if score >= 85 else AMBER if score >= 65 else RED
    pdf.set_font("Helvetica", "B", 26); pdf.set_text_color(*col)
    pdf.cell(28, 12, str(score), ln=0)
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(30, 41, 59)
    pdf.set_xy(40, pdf.get_y() + 1)
    pdf.multi_cell(0, 5, _a(f"Overall health score (0-100)\nClassification: {cert['classification']}  "
                            f"|  Risk class: {cert['risk_class']}"))
    pdf.ln(2)

    # ── Asset details ──
    _section(pdf, "1. Asset Details")
    _kv(pdf, "Turbine", f"{t.name} (asset ref {t.external_ref})")
    _kv(pdf, "OEM / Model", f"{t.oem or 'n/a'} {t.model or ''}")
    _kv(pdf, "Rated power", f"{int(t.rated_power_kw or 0)} kW")
    _kv(pdf, "Site / Location", site.name if site else "n/a")
    _kv(pdf, "Commissioned", str(t.commissioned_on) if t.commissioned_on else "Not recorded (recommend adding to asset master)")
    _kv(pdf, "Assessment date", f"{cert['issued_at']:%Y-%m-%d}")
    _kv(pdf, "Valid until", f"{cert['valid_until']:%Y-%m-%d} (90-day validity)")

    # ── Data provenance ──
    _section(pdf, "2. Data Provenance")
    _kv(pdf, "SCADA coverage", f"{cert.get('data_coverage_pct', 0)}% of assessed window has complete key signals")
    _kv(pdf, "Rows assessed", f"{cert.get('rows_assessed', 0):,} records (10-minute averages)")
    _kv(pdf, "Sampling note", "10-min averaged SCADA hides high-frequency early signatures; "
                              "high-frequency/vibration data recommended for earlier warning.")

    # ── Model traceability ──
    _section(pdf, "3. Model Traceability")
    _kv(pdf, "Model version", pred.model_version or "n/a")
    _kv(pdf, "Training window", cert.get("model_training_window", "n/a"))
    _kv(pdf, "Calibration", cert.get("calibration_status", "n/a"))
    _kv(pdf, "Fault probability", f"{(pred.fault_probability or 0)*100:.0f}% (calibrated), component: {pred.predicted_component}")

    # ── Component scores + RUL ──
    _section(pdf, "4. Component Scores & Remaining Useful Life")
    pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*GREY)
    pdf.cell(60, 6, "Component", border="B"); pdf.cell(35, 6, "Health (0-100)", border="B")
    pdf.cell(0, 6, "RUL estimate (with range)", border="B", ln=1)
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(30, 41, 59)
    for comp, s in cert["component_scores"].items():
        rul = cert["rul_estimates"].get(comp, 0)
        lo, hi = int(rul * 0.7), int(rul * 1.3)
        sc = GREEN if s >= 70 else AMBER if s >= 45 else RED
        pdf.cell(60, 5.5, _a(comp))
        pdf.set_text_color(*sc); pdf.cell(35, 5.5, str(int(s)))
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 5.5, _a(f"~{int(rul)} days  (range {lo}-{hi} d)"), ln=1)

    # ── Narrative ──
    _section(pdf, "5. Plain-language Assessment")
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5, _a(cert["narrative"]))

    # ── Limitations & miss-rate ──
    _section(pdf, "6. Limitations & Known Miss-Rate")
    pdf.set_font("Helvetica", "", 8.5); pdf.set_text_color(120, 50, 10)
    pdf.multi_cell(0, 4.6, _a(cert.get("limitations", "")))

    # ── Certifying engineer ──
    _section(pdf, "7. Certifying Engineer")
    _kv(pdf, "Engineer", cert.get("certifying_engineer", "Pending sign-off"))
    pdf.ln(6)
    pdf.set_draw_color(*GREY); pdf.set_text_color(*GREY); pdf.set_font("Helvetica", "", 8)
    pdf.cell(80, 5, "", border="B"); pdf.cell(20, 5, ""); pdf.cell(60, 5, "", border="B", ln=1)
    pdf.cell(80, 4, "Signature"); pdf.cell(20, 4, ""); pdf.cell(60, 4, "Date / License no.", ln=1)

    out = pdf.output()
    return bytes(out)
