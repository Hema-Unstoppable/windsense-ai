"""
═══════════════════════════════════════════════════════════════════════
 EXPLAINABILITY INTEGRITY  —  make SHAP reflect physics, not coincidence
═══════════════════════════════════════════════════════════════════════
SHAP charts look convincing but can be physically nonsensical. This module
provides the guardrails the reviewers demanded:

  • PHYSICAL_DRIVERS — which canonical signals *should* drive each failure
    mode. Lets us flag cross-component drivers (e.g. a generator bearing
    signal "explaining" a gearbox fault).
  • COLLINEAR_GROUPS — thermal/electrical signals that move together, so
    we group their attribution instead of presenting an arbitrary split.
  • annotate_shap() — tags each SHAP feature {expected, group}, computes a
    physical-alignment score and the right caveats.

Used live by the API (turbine health summary) and by the validation suite
(perturbation + stability + alignment tests).
"""
from __future__ import annotations

# Which canonical-signal stems are the PHYSICALLY EXPECTED drivers per mode.
PHYSICAL_DRIVERS: dict[str, list[str]] = {
    "Gearbox":     ["gearbox", "high_speed_shaft", "rot_rpm", "gen_latest_period_rpm"],
    "Generator":   ["gen_bearing", "stator_winding", "gen_latest_period_rpm",
                    "aver_curr", "split_ring", "busbar"],
    "Hydraulic":   ["hydraulic", "pit_angle"],
    "Transformer": ["hv_transformer", "transformer", "aver_voltage", "igbt", "grid_reactive"],
    "Main Bearing":["main_bearing", "high_speed_shaft", "rot_rpm", "nose_cone"],
}

# Sensor families that are physically collinear → SHAP splits them arbitrarily.
COLLINEAR_GROUPS: dict[str, list[str]] = {
    "Thermal cluster (collinear)": [
        "amb_temp", "nac_temperature", "temp_oil_gearbox", "temp_gen_bearing",
        "temp_nose_cone", "temp_hub_controller", "temp_top_nacelle", "temp_choke",
        "temp_vcp", "temp_vcs", "temp_busbar", "temp_split_ring", "temp_in_stator",
        "temp_bearing_on_high_speed", "temp_hv_transformer", "temp_the_igbt", "temp_igbt",
    ],
    "Electrical cluster (collinear)": [
        "grid_power", "poss_active_power", "aver_curr", "aver_voltage",
        "grid_reactive", "poss_induc", "poss_cap", "grid_freq",
    ],
}

# Signals an operator can physically push UP to test causal response (perturbation).
# delta = a realistic step change; degradation should INCREASE predicted risk.
PERTURBATION_SIGNALS: dict[str, float] = {
    "Temp_oil_gearbox_C_avg": 12.0,
    "Temp_gen_bearing_Drive_End_C_avg": 12.0,
    "Temp_gen_bearing_Non_Drive_End_C_avg": 12.0,
    "Temp_hydraulic_oil_group_C": 12.0,
    "Temp_HV_transformer_phase_L1_C": 12.0,
    "gearbox_temp_margin": 10.0,
    "gen_bearing_DE_margin": 10.0,
    "hydraulic_temp_margin": 10.0,
}


def _matches(feature: str, stems: list[str]) -> bool:
    f = feature.lower()
    return any(s in f for s in stems)


def is_expected(feature: str, component: str) -> bool:
    """Is this feature a physically expected driver for the predicted mode?"""
    stems = PHYSICAL_DRIVERS.get(component)
    if not stems:
        return True            # Normal / unknown → don't flag
    # temperature margins are de-collinearised and broadly informative → allow
    if "_margin" in feature.lower() and _matches(feature, stems):
        return True
    return _matches(feature, stems)


def collinear_group(feature: str) -> str | None:
    for group, stems in COLLINEAR_GROUPS.items():
        # margins are de-collinearised, exclude them from the raw thermal cluster
        if "_margin" in feature.lower():
            continue
        if _matches(feature, stems):
            return group
    return None


def annotate_shap(shap_list: list[dict], component: str) -> dict:
    """
    Tag each SHAP feature with {expected, group}, and compute:
      • alignment_score : fraction of |contribution| from physically expected signals
      • collinearity_caveat : when ≥2 top drivers are collinear
      • unexpected_top : flag when the #1 driver is not a typical signal for the mode
      • verdict
    Returns {features, alignment_score, caveat, unexpected_top, verdict}.
    """
    if not shap_list:
        return {"features": [], "alignment_score": None, "caveat": None,
                "unexpected_top": None, "verdict": "No SHAP available."}

    feats = []
    total = sum(abs(f.get("contribution", 0)) for f in shap_list) or 1.0
    expected_abs = 0.0
    group_counts: dict[str, int] = {}
    for f in shap_list:
        exp = is_expected(f["feature"], component)
        grp = collinear_group(f["feature"])
        if exp:
            expected_abs += abs(f.get("contribution", 0))
        if grp:
            group_counts[grp] = group_counts.get(grp, 0) + 1
        feats.append({**f, "expected": exp, "group": grp})

    alignment = round(expected_abs / total, 3) if component in PHYSICAL_DRIVERS else None

    caveat = None
    for grp, n in group_counts.items():
        if n >= 2:
            caveat = (f"{n} of the top drivers belong to the {grp} — these signals move "
                      f"together, so SHAP splits their importance arbitrarily. Read them as a "
                      f"group, not as precise individual percentages.")
            break

    unexpected_top = None
    top = feats[0]
    if component in PHYSICAL_DRIVERS and not top["expected"]:
        unexpected_top = (f"Top driver '{top['feature']}' is not a typical {component} signal. "
                          f"Verify a real mechanical/thermal coupling or treat as possible "
                          f"spurious correlation before acting.")

    if alignment is None:
        verdict = "n/a"
    elif alignment >= 0.6:
        verdict = "ALIGNED — explanation is dominated by physically expected signals."
    elif alignment >= 0.3:
        verdict = "PARTIAL — some drivers are physically expected; review the rest."
    else:
        verdict = "WEAK — explanation is dominated by signals not physically tied to this mode."

    return {"features": feats, "alignment_score": alignment, "caveat": caveat,
            "unexpected_top": unexpected_top, "verdict": verdict}
