"""
═══════════════════════════════════════════════════════════════════════
 FMECA RISK ENGINE  (the WindSense core IP)
═══════════════════════════════════════════════════════════════════════
Turns raw ML probabilities into prioritised maintenance decisions using a
Failure-Mode, Effects & Criticality Analysis (FMECA) weighting scheme.

Risk score formula (matches the UI):

    risk_score = ML_fault_probability
                 × FMECA_consequence_weight
                 × (1 − maintenance_coverage)

The Risk Priority Number (RPN) additionally folds in detectability:

    RPN = probability(1-10) × consequence(1-10) × detectability(1-10)
"""
from __future__ import annotations

# Fault class index → component (matches the trained classifier targets)
CLASS_TO_COMPONENT = {
    0: "Normal",
    1: "Generator",
    2: "Gearbox",
    3: "Hydraulic",
    4: "Transformer",
}
COMPONENT_TO_CLASS = {v: k for k, v in CLASS_TO_COMPONENT.items()}

# FMECA consequence weights (0..1) — severity if the failure occurs.
# Derived from RCM/FMECA practice for wind turbine subsystems.
CONSEQUENCE_WEIGHT = {
    "Gearbox":     0.90,   # very high — €300-400k, 5-14 day downtime
    "Generator":   0.80,
    "Main Bearing":0.85,
    "Transformer": 0.75,
    "Hydraulic":   0.55,
    "Pitch":       0.60,
    "Yaw":         0.50,
    "Blades":      0.70,
    "Normal":      0.00,
}

# Typical detectability lead-time (days of warning the model provides).
# Higher lead time = more detectable = LOWER risk contribution.
DETECTION_LEAD_DAYS = {
    "Gearbox":     30,
    "Generator":   28,
    "Hydraulic":   20,
    "Transformer": 25,
    "Normal":      90,
}

# Indicative repair / downtime economics per component (EUR), planned cost
# and the unplanned premium used for financial-exposure estimates.
COMPONENT_ECONOMICS = {
    "Gearbox":     {"planned": 120_000, "unplanned": 400_000, "downtime_days": 14},
    "Generator":   {"planned": 60_000,  "unplanned": 180_000, "downtime_days": 10},
    "Hydraulic":   {"planned": 8_000,   "unplanned": 35_000,  "downtime_days": 4},
    "Transformer": {"planned": 45_000,  "unplanned": 150_000, "downtime_days": 9},
    "Normal":      {"planned": 0,       "unplanned": 0,       "downtime_days": 0},
}


# ── Itemised consequence cost model (auditable, line-by-line) ─────────
# parts + labour + crane/vessel mobilisation, with written severity criteria
# and ISO 14224 / IEC wind-taxonomy alignment so data transfers between fleets.
CONSEQUENCE_ITEMS = {
    "Gearbox":     {"parts": 230_000, "labour": 22_000, "crane": 120_000, "downtime_days": 14, "severity": 9},
    "Generator":   {"parts": 90_000,  "labour": 18_000, "crane": 120_000, "downtime_days": 10, "severity": 8},
    "Main Bearing":{"parts": 110_000, "labour": 20_000, "crane": 120_000, "downtime_days": 12, "severity": 8},
    "Transformer": {"parts": 95_000,  "labour": 15_000, "crane": 0,       "downtime_days": 9,  "severity": 7},
    "Hydraulic":   {"parts": 12_000,  "labour": 6_000,  "crane": 0,       "downtime_days": 4,  "severity": 5},
    "Pitch":       {"parts": 18_000,  "labour": 8_000,  "crane": 0,       "downtime_days": 5,  "severity": 6},
    "Yaw":         {"parts": 15_000,  "labour": 7_000,  "crane": 0,       "downtime_days": 4,  "severity": 5},
    "Blades":      {"parts": 60_000,  "labour": 30_000, "crane": 120_000, "downtime_days": 10, "severity": 7},
    "Normal":      {"parts": 0,       "labour": 0,      "crane": 0,       "downtime_days": 0,  "severity": 1},
}

# Written severity criteria — every score means something concrete.
SEVERITY_CRITERIA = {
    "Gearbox":     "S9 — full gearbox replacement, ~14 days downtime, main crane required (~€370k consequence).",
    "Generator":   "S8 — generator swap, ~10 days downtime, crane required (~€230k consequence).",
    "Main Bearing":"S8 — main bearing exchange, ~12 days downtime, crane required (~€250k consequence).",
    "Transformer": "S7 — transformer repair/replace, ~9 days downtime, no crane (~€110k consequence).",
    "Hydraulic":   "S5 — hydraulic group repair, ~4 days downtime, internal crew (~€18k consequence).",
    "Pitch":       "S6 — pitch actuator service, ~5 days downtime (~€26k consequence).",
    "Yaw":         "S5 — yaw drive service, ~4 days downtime (~€22k consequence).",
    "Blades":      "S7 — blade repair, ~10 days downtime, crane required (~€90k consequence).",
}

# ISO 14224 / IEC 61400-style taxonomy so failure naming transfers across fleets.
ISO14224_TAXONOMY = {
    "Gearbox":     {"equipment_class": "Wind turbine", "subunit": "Drivetrain / Gearbox", "code": "WT-DT-GB"},
    "Generator":   {"equipment_class": "Wind turbine", "subunit": "Electrical / Generator", "code": "WT-EL-GEN"},
    "Main Bearing":{"equipment_class": "Wind turbine", "subunit": "Drivetrain / Main bearing", "code": "WT-DT-MB"},
    "Transformer": {"equipment_class": "Wind turbine", "subunit": "Electrical / Transformer", "code": "WT-EL-TR"},
    "Hydraulic":   {"equipment_class": "Wind turbine", "subunit": "Hydraulic system", "code": "WT-HY"},
    "Pitch":       {"equipment_class": "Wind turbine", "subunit": "Pitch system", "code": "WT-PI"},
    "Yaw":         {"equipment_class": "Wind turbine", "subunit": "Yaw system", "code": "WT-YA"},
    "Blades":      {"equipment_class": "Wind turbine", "subunit": "Rotor / Blades", "code": "WT-RO-BL"},
}


def consequence_weight(component: str) -> float:
    return CONSEQUENCE_WEIGHT.get(component, 0.5)


def consequence_cost(component: str, rated_power_kw: float = 2000.0,
                     power_price_eur_mwh: float = 80.0,
                     capacity_factor: float = 0.35) -> dict:
    """
    Itemised consequence cost in € — every line defensible to an auditor.
      total = parts + labour + crane/vessel + lost_revenue
      lost_revenue = capacity(MW) × CF × downtime_hours × power_price(€/MWh)
    """
    it = CONSEQUENCE_ITEMS.get(component, CONSEQUENCE_ITEMS["Hydraulic"])
    downtime_h = it["downtime_days"] * 24
    lost_revenue = (rated_power_kw / 1000.0) * capacity_factor * downtime_h * power_price_eur_mwh
    total = it["parts"] + it["labour"] + it["crane"] + lost_revenue
    return {
        "parts": it["parts"], "labour": it["labour"], "crane": it["crane"],
        "lost_revenue": round(lost_revenue), "downtime_days": it["downtime_days"],
        "total": round(total), "severity": it["severity"],
        "lost_revenue_formula": f"{rated_power_kw/1000:.1f}MW × {capacity_factor:.0%} CF × "
                                f"{downtime_h}h × €{power_price_eur_mwh}/MWh",
    }


def expected_economic_loss(fault_probability: float, component: str,
                           rated_power_kw: float = 2000.0,
                           power_price_eur_mwh: float = 80.0) -> dict:
    """EEL = P(failure within RUL window) × itemised consequence cost (€)."""
    cost = consequence_cost(component, rated_power_kw, power_price_eur_mwh)
    eel = round(float(fault_probability) * cost["total"])
    return {"expected_loss_eur": eel, "fault_probability": round(float(fault_probability), 3),
            "consequence": cost}


def action_priority(fault_probability: float, component: str, rul_days: float | None) -> str:
    """
    AIAG-VDA style Action Priority (High/Medium/Low) — the modern replacement
    for raw S×O×D multiplication. Combines severity, occurrence (probability)
    and detection (RUL vs available lead time).
    """
    S = CONSEQUENCE_ITEMS.get(component, {}).get("severity", 5)
    high_occ = fault_probability >= 0.5
    med_occ = fault_probability >= 0.25
    imminent = (rul_days is not None) and (rul_days <= 14)
    if S >= 8 and (high_occ or imminent):
        return "High"
    if S >= 8 and med_occ:
        return "High"
    if S >= 6 and high_occ:
        return "High"
    if S >= 5 and (med_occ or imminent):
        return "Medium"
    if med_occ:
        return "Medium"
    return "Low"


def risk_score(fault_probability: float, component: str,
               maintenance_coverage: float = 0.0) -> float:
    """Core WindSense risk score, 0..1."""
    cw = consequence_weight(component)
    return float(fault_probability) * cw * (1.0 - float(maintenance_coverage))


def detectability_factor(component: str, rul_days: float | None) -> float:
    """0..1 — lower = easier to detect in time (less risky)."""
    lead = DETECTION_LEAD_DAYS.get(component, 30)
    if rul_days is None:
        return 0.5
    # If RUL is much shorter than typical lead time → harder to act → higher.
    ratio = max(0.0, min(1.5, lead / max(rul_days, 1.0)))
    return min(1.0, 0.3 + 0.47 * ratio)


def rpn(fault_probability: float, component: str, rul_days: float | None) -> float:
    """Risk Priority Number on a 1..1000 style scale (here scaled 0..100)."""
    prob_score = 1 + 9 * fault_probability                  # 1..10
    cons_score = 1 + 9 * consequence_weight(component)       # 1..10
    det_score = 1 + 9 * detectability_factor(component, rul_days)  # 1..10
    raw = prob_score * cons_score * det_score                # 1..1000
    return round(raw / 10.0, 1)                              # 0..100


def risk_class(score: float) -> str:
    """Map a 0..1 risk score to a class band."""
    if score >= 0.65:
        return "CRITICAL"
    if score >= 0.40:
        return "HIGH"
    if score >= 0.20:
        return "MEDIUM"
    return "LOW"


def status_from_class(rc: str) -> str:
    return {"CRITICAL": "Critical", "HIGH": "Warning",
            "MEDIUM": "Warning", "LOW": "Operational"}.get(rc, "Operational")


def financial_exposure(component: str, fault_probability: float,
                       rated_power_kw: float = 2000.0,
                       power_price_eur_mwh: float = 80.0) -> float:
    """
    Expected Economic Loss (€) = P(failure) × itemised consequence cost.
    Now derived line-by-line (parts + labour + crane + lost revenue) so the
    dashboard's Financial Exposure is a number you can defend, not invent.
    """
    return expected_economic_loss(fault_probability, component,
                                  rated_power_kw, power_price_eur_mwh)["expected_loss_eur"]


def health_score_from_probability(fault_probability: float) -> float:
    """Convert anomaly probability to a 0..100 health score."""
    return round(max(0.0, min(100.0, 100.0 * (1.0 - fault_probability))), 0)


def classification_from_health(score: float) -> str:
    if score >= 85:
        return "HEALTHY"
    if score >= 65:
        return "SERVICEABLE"
    if score >= 45:
        return "REQUIRES ATTENTION"
    return "CRITICAL - IMMEDIATE ACTION"


def rul_from_probability(fault_probability: float, component: str) -> float:
    """
    Heuristic RUL (days) fallback when a regression model is unavailable.
    Maps anomaly probability to a remaining-life band scaled by the
    component's typical detection lead time.
    """
    lead = DETECTION_LEAD_DAYS.get(component, 30)
    # high probability → short RUL
    return round(max(3.0, lead * (1.05 - fault_probability)), 0)
