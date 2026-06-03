"""
═══════════════════════════════════════════════════════════════════════
 MAINTENANCE DECISION-SUPPORT ENGINE
═══════════════════════════════════════════════════════════════════════
Turns ML outputs into a TRACEABLE recommendation, framed as decision
support (not an autonomous instruction) with a human in the loop — the
lower-liability posture the reviews require.

Handles the four edge cases explicitly, in priority order:
  1. Signals point to a faulty sensor   → CHECK INSTRUMENT (don't send a crane)
  2. Low prob but imminent + critical    → ESCALATE / INSPECT NOW (don't bury it)
  3. High prob, long life, low conseq.   → MONITOR (don't cry wolf)
  4. Models disagree                     → INSPECT to resolve + FLAG the doubt

Every output carries its rationale (the evidence + the rule that fired) and a
liability disclaimer. No silent single-signal picks.
"""
from __future__ import annotations

DISCLAIMER = ("Decision support only — not an autonomous instruction. Requires review "
              "and sign-off by a certified reliability engineer before any action.")


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def build_recommendation(fault_prob: float, rul_days: float | None, component: str,
                         severity: int, eel_eur: float, explain: dict | None = None) -> dict:
    """
    Returns a structured, traceable decision-support recommendation.
    `explain` is the annotate_shap() output (optional) — enables the
    sensor-fault and model-conflict edge cases.
    """
    rul = float(rul_days) if rul_days is not None else 60.0
    very_imminent = rul <= 7
    imminent = rul <= 14
    high_prob = fault_prob >= 0.5
    mod_prob = fault_prob >= 0.3
    low_prob = fault_prob < 0.3
    critical_part = severity >= 8
    low_consequence = severity <= 5

    # ── detect sensor-fault and model-conflict signatures from SHAP ──
    sensor_fault = False
    model_conflict = False
    top_feature = None
    if explain:
        feats = explain.get("features") or []
        align = explain.get("alignment_score")
        caveat = explain.get("caveat")
        if feats:
            top = feats[0]
            top_feature = top.get("feature")
            second = feats[1]["pct"] if len(feats) > 1 else 0
            isolated = top.get("pct", 0) >= 40 and top.get("pct", 0) >= 2 * (second or 0.1)
            single_raw_sensor = ("_margin" not in (top_feature or "").lower()) and (top.get("group") is not None)
            if isolated and single_raw_sensor and (align is not None and align < 0.35):
                sensor_fault = True
        # genuine conflict = signed SHAP shows a substantial OPPOSING signal
        # (some drivers push risk up, others pull it down) while the model is uncertain.
        contribs = [f.get("contribution", 0) or 0 for f in feats]
        pos = max([c for c in contribs if c > 0], default=0.0)
        neg = max([-c for c in contribs if c < 0], default=0.0)
        if pos > 0 and neg >= 0.6 * pos and 0.35 <= fault_prob <= 0.65:
            model_conflict = True

    rationale: list[str] = []

    # ── decision tree (priority order matters) ──
    if sensor_fault:
        action, urgency = "CHECK_INSTRUMENT", "Medium"
        headline = "Verify instrumentation before any mechanical action"
        rationale.append(
            f"The prediction is dominated by a single isolated sensor ({top_feature}) that is "
            f"not physically aligned with a {component} failure mode — a likely instrument or "
            f"calibration issue, not mechanical degradation.")
        rationale.append("Do NOT mobilise a crane on this signal alone; check/replace the sensor first.")

    elif low_prob and very_imminent and critical_part:
        action, urgency = "ESCALATE_INSPECT_NOW", "Immediate"
        headline = f"Escalate — inspect {component} now"
        rationale.append(
            f"Probability is only {_pct(fault_prob)}, but estimated lead time is very short "
            f"({int(rul)}d) on a critical component (severity {severity}/10). On critical parts, "
            f"imminence outweighs probability — do not bury this.")

    elif high_prob and not imminent and low_consequence:
        action, urgency = "MONITOR", "Low"
        headline = "Monitor — no action required yet"
        rationale.append(
            f"Elevated probability ({_pct(fault_prob)}) but long lead time ({int(rul)}d) and low "
            f"consequence (severity {severity}/10). Continue monitoring; raising a crisis now would "
            f"create alarm fatigue.")

    elif high_prob and imminent and critical_part:
        action, urgency = "PLAN_INTERVENTION", "High"
        headline = f"Plan {component} intervention (account for crane lead-time)"
        rationale.append(
            f"High probability ({_pct(fault_prob)}), short lead time ({int(rul)}d) and a critical "
            f"component (severity {severity}/10). Begin planning parts + crane/vessel now.")

    elif high_prob and very_imminent:
        action, urgency = "SCHEDULE_MAINTENANCE", "High"
        headline = f"Schedule {component} maintenance this week"
        rationale.append(f"High probability ({_pct(fault_prob)}) and short lead time ({int(rul)}d).")

    elif model_conflict:
        action, urgency = "INSPECT_TO_RESOLVE", "Medium"
        headline = "Signals disagree — inspect to resolve"
        rationale.append(
            "Different signal families disagree (e.g. thermal vs electrical). Conflict-resolution "
            "rule: schedule a targeted inspection rather than acting on either signal alone. The "
            "disagreement is flagged, not hidden.")

    elif high_prob:
        action, urgency = "SCHEDULE_MAINTENANCE", "Medium"
        headline = f"Schedule {component} inspection / maintenance"
        rationale.append(f"Probability {_pct(fault_prob)} with {int(rul)}d lead time.")

    elif mod_prob and critical_part:
        action, urgency = "INSPECT_SOON", "Medium"
        headline = f"Schedule {component} inspection"
        rationale.append(
            f"Moderate probability ({_pct(fault_prob)}) on a critical component (severity "
            f"{severity}/10) — an inspection is prudent even before high confidence.")

    elif imminent and critical_part:
        action, urgency = "ESCALATE_INSPECT_NOW", "High"
        headline = f"Inspect {component} — imminent on a critical part"
        rationale.append(f"Short lead time ({int(rul)}d) on a critical component (severity {severity}/10).")

    else:
        action, urgency = "MONITOR", "Low"
        headline = "Monitor — within normal limits"
        rationale.append(
            f"Probability {_pct(fault_prob)}, lead time {int(rul)}d, severity {severity}/10 — "
            f"no action indicated.")

    if model_conflict and action != "INSPECT_TO_RESOLVE":
        rationale.append("NOTE: signal families partially disagree — treat confidence with caution.")

    rationale.append(
        f"Evidence: calibrated P(fail)={_pct(fault_prob)}, RUL≈{int(rul)}d, "
        f"expected loss €{int(eel_eur):,}, severity {severity}/10.")

    return {
        "action": action,
        "action_label": action.replace("_", " ").title(),
        "urgency": urgency,
        "headline": headline,
        "rationale": rationale,
        "sensor_fault_suspected": sensor_fault,
        "model_conflict": model_conflict,
        "classification": "DECISION SUPPORT",
        "human_in_the_loop": True,
        "disclaimer": DISCLAIMER,
    }
