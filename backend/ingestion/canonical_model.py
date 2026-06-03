"""
═══════════════════════════════════════════════════════════════════════
 CANONICAL WIND DATA MODEL
═══════════════════════════════════════════════════════════════════════
The single, OEM-independent schema every site maps into. Onboarding a new
turbine/OEM means writing a mapping file (source tag -> canonical field) —
never new Python. This registry defines each canonical field's unit, valid
physical range, group, whether it is required for the ML model, and whether
it should be checked for a "frozen sensor".
"""
from __future__ import annotations

CANONICAL_FIELDS: dict[str, dict] = {
    # ── operational ──
    "Wind_speed_m/s_avg":        {"unit": "m/s", "min": 0, "max": 40,  "group": "operational", "required": True,  "frozen_check": True},
    "Est_windspeed_m/s_avg":     {"unit": "m/s", "min": 0, "max": 40,  "group": "operational", "required": False, "frozen_check": True},
    "Grid_power_kW_avg":         {"unit": "kW",  "min": -500, "max": 6000, "group": "power",   "required": True,  "frozen_check": True},
    "Rot_rpm_rpm_avg":           {"unit": "rpm", "min": 0, "max": 30,  "group": "operational", "required": False, "frozen_check": True},
    "gen_latest_period_rpm_avg": {"unit": "rpm", "min": 0, "max": 2200,"group": "operational", "required": False, "frozen_check": True},
    "Pit_angle_avg":             {"unit": "deg", "min": -10, "max": 100,"group": "mechanical",  "required": False, "frozen_check": False, "is_angle": True},
    "Nac_direction":             {"unit": "deg", "min": 0, "max": 360, "group": "mechanical",  "required": False, "frozen_check": False, "is_angle": True},
    # ── thermal ──
    "Temp_oil_gearbox_C_avg":              {"unit": "degC", "min": -40, "max": 120, "group": "thermal", "required": True,  "frozen_check": True},
    "Temp_bearing_on_high_speed_shaft_C_avg": {"unit": "degC", "min": -40, "max": 150, "group": "thermal", "required": False, "frozen_check": True},
    "Temp_gen_bearing_Drive_End_C_avg":    {"unit": "degC", "min": -40, "max": 150, "group": "thermal", "required": True,  "frozen_check": True},
    "Temp_gen_bearing_Non_Drive_End_C_avg":{"unit": "degC", "min": -40, "max": 150, "group": "thermal", "required": False, "frozen_check": True},
    "Nac_temperature_C_avg":               {"unit": "degC", "min": -40, "max": 80,  "group": "thermal", "required": True,  "frozen_check": True},
    "Amb_temp_C_avg":                      {"unit": "degC", "min": -50, "max": 60,  "group": "thermal", "required": False, "frozen_check": True},
    "Temp_hydraulic_oil_group_C":          {"unit": "degC", "min": -40, "max": 120, "group": "thermal", "required": False, "frozen_check": True},
    "Temp_HV_transformer_phase_L1_C":      {"unit": "degC", "min": -40, "max": 200, "group": "thermal", "required": False, "frozen_check": True},
    # ── electrical ──
    "Grid_freq_Hz_avg":          {"unit": "Hz",   "min": 45, "max": 65,   "group": "electrical", "required": False, "frozen_check": False},
    "Aver_voltage_ph1_V_avg":    {"unit": "V",    "min": 0,  "max": 800,  "group": "electrical", "required": False, "frozen_check": True},
    "Grid_reactive_power_kVAr_avg": {"unit": "kVAr", "min": -3000, "max": 3000, "group": "electrical", "required": False, "frozen_check": True},
}

REQUIRED_FIELDS = [k for k, v in CANONICAL_FIELDS.items() if v.get("required")]


# ── Unit conversions (source unit -> canonical unit) ─────────────────
CONVERSIONS = {
    ("degF", "degC"): lambda x: (x - 32.0) * 5.0 / 9.0,
    ("K", "degC"):    lambda x: x - 273.15,
    ("W", "kW"):      lambda x: x / 1000.0,
    ("MW", "kW"):     lambda x: x * 1000.0,
    ("VAr", "kVAr"):  lambda x: x / 1000.0,
    ("MVAr", "kVAr"): lambda x: x * 1000.0,
    ("kV", "V"):      lambda x: x * 1000.0,
    ("mph", "m/s"):   lambda x: x * 0.44704,
    ("km/h", "m/s"):  lambda x: x / 3.6,
}


def field_meta(canonical: str) -> dict | None:
    return CANONICAL_FIELDS.get(canonical)


def is_known(canonical: str) -> bool:
    return canonical in CANONICAL_FIELDS


def convert_units(series, from_unit: str, to_unit: str):
    """Convert a pandas Series from source unit to canonical unit (no-op if equal/unknown)."""
    if not from_unit or from_unit == to_unit:
        return series, False
    fn = CONVERSIONS.get((from_unit, to_unit))
    if fn is None:
        return series, False         # unknown conversion -> leave as-is, flagged elsewhere
    return series.map(lambda v: fn(v) if v is not None else v), True
