"""
═══════════════════════════════════════════════════════════════════════
 NORMALISATION ENGINE  (Layer 2 of the WindSense architecture)
═══════════════════════════════════════════════════════════════════════
Maps raw, OEM-specific SCADA tag names to a single CANONICAL schema, so
the ML models and the entire experience layer are identical regardless of
turbine make/model. This is what makes WindSense OEM-agnostic.

For Wind Farm A (CARE to Compare dataset) the raw tags are anonymised
(`sensor_0_avg`, `power_29_avg`, ...). This module maps them to the
human-readable canonical names used by the models and surfaced in the UI.

To onboard a new OEM/client you add one more mapping dict — the rest of
the platform is untouched.
"""
from __future__ import annotations

import pandas as pd

# ── Canonical hot-signal names (promoted to real DB columns) ──────────
HOT_SIGNALS = {
    "wind_speed":          "Wind_speed_m/s_avg",
    "power_kw":            "Grid_power_kW_avg",
    "rotor_rpm":           "Rot_rpm_rpm_avg",
    "gearbox_oil_temp":    "Temp_oil_gearbox_C_avg",
    "gen_bearing_de_temp": "Temp_gen_bearing_Drive_End_C_avg",
    "nacelle_temp":        "Nac_temperature_C_avg",
    "ambient_temp":        "Amb_temp_C_avg",
}

# ── Raw → canonical mapping for the Wind Farm A (CARE) anonymised feed ─
# Built from comma_feature_description_modified.csv. {raw_col: canonical}
WINDFARM_A_MAP: dict[str, str] = {
    "sensor_0_avg": "Amb_temp_C_avg",
    "sensor_1_avg": "Wind_abs_dir_avg",
    "sensor_2_avg": "Wind_rel_dir_avg",
    "wind_speed_3_avg": "Wind_speed_m/s_avg",
    "wind_speed_3_max": "Wind_speed_m/s_max",
    "wind_speed_3_min": "Wind_speed_m/s_min",
    "wind_speed_3_std": "Wind_speed_m/s_std",
    "wind_speed_4_avg": "Est_windspeed_m/s_avg",
    "sensor_5_avg": "Pit_angle_avg",
    "sensor_5_max": "Pit_angle_max",
    "sensor_5_min": "Pit_angle_min",
    "sensor_5_std": "Pit_angle_std",
    "sensor_6_avg": "Temp_hub_controller_C_avg",
    "sensor_7_avg": "Temp_top_nacelle_controller_C_avg",
    "sensor_8_avg": "Temp_choke_coils_on_the_VCS_section_C_avg",
    "sensor_9_avg": "Temp_VCP_board_C_avg",
    "sensor_10_avg": "Temp_VCS_cooling_water_C_avg",
    "sensor_11_avg": "Temp_bearing_on_high_speed_shaft_C_avg",
    "sensor_12_avg": "Temp_oil_gearbox_C_avg",
    "sensor_13_avg": "Temp_gen_bearing_Drive_End_C_avg",
    "sensor_14_avg": "Temp_gen_bearing_Non_Drive_End_C_avg",
    "sensor_15_avg": "Temp_in_stator_windings_ph1_C_avg",
    "sensor_16_avg": "Temp_in_stator_windings_ph2_C_avg",
    "sensor_17_avg": "Temp_in_stator_windings_ph3_C_avg",
    "sensor_18_avg": "gen_latest_period_rpm_avg",
    "sensor_18_max": "gen_latest_period_rpm_max",
    "sensor_18_min": "gen_latest_period_rpm_min",
    "sensor_18_std": "gen_latest_period_rpm_std",
    "sensor_19_avg": "Temp_split_ring_chamber_C_avg",
    "sensor_20_avg": "Temp_busbar_section_C_avg",
    "sensor_21_avg": "Temp_the_IGBT_driver_on_the_grid_side_inverter_C_avg",
    "sensor_22_avg": "Act_ph_displacement_avg",
    "sensor_23_avg": "Aver_curr_ph1_A_avg",
    "sensor_24_avg": "Aver_curr_ph2_A_avg",
    "sensor_25_avg": "Aver_curr_ph3_A_avg",
    "sensor_26_avg": "Grid_freq_Hz_avg",
    "reactive_power_27_avg": "Poss_cap_reactive_power_kVAr_avg",
    "reactive_power_27_max": "Poss_cap_reactive_power_kVAr_max",
    "reactive_power_27_min": "Poss_cap_reactive_power_kVAr_min",
    "reactive_power_27_std": "Poss_cap_reactive_power_kVAr_std",
    "reactive_power_28_avg": "Poss_induc_reactive_power_kVAr_avg",
    "reactive_power_28_max": "Poss_induc_reactive_power_kVAr_max",
    "reactive_power_28_min": "Poss_induc_reactive_power_kVAr_min",
    "reactive_power_28_std": "Poss_induc_reactive_power_kVAr_std",
    "power_29_avg": "Poss_active_power_kW_avg",
    "power_29_max": "Poss_active_power_kW_max",
    "power_29_min": "Poss_active_power_kW_min",
    "power_29_std": "Poss_active_power_kW_std",
    "power_30_avg": "Grid_power_kW_avg",
    "power_30_max": "Grid_power_kW_max",
    "power_30_min": "Grid_power_kW_min",
    "power_30_std": "Grid_power_kW_std",
    "sensor_31_avg": "Grid_reactive_power_kVAr_avg",
    "sensor_31_max": "Grid_reactive_power_kVAr_max",
    "sensor_31_min": "Grid_reactive_power_kVAr_min",
    "sensor_31_std": "Grid_reactive_power_kVAr_std",
    "sensor_32_avg": "Aver_voltage_ph1_V_avg",
    "sensor_33_avg": "Aver_voltage_ph2_V_avg",
    "sensor_34_avg": "Aver_voltage_ph3_V_avg",
    "sensor_35_avg": "Temp_IGBT_rotor_side_inverter_ph1_C_avg",
    "sensor_36_avg": "Temp_IGBT_rotor_side_inverter_ph2_C_avg",
    "sensor_37_avg": "Temp_the_IGBT_driver_on_the_rotor_side_inverter_phase_ph3_C",
    "sensor_38_avg": "Temp_HV_transformer_phase_L1_C",
    "sensor_39_avg": "Temp_HV_transformer_phase_L2_C",
    "sensor_40_avg": "Temp_HV_transformer_phase_L3_C",
    "sensor_41_avg": "Temp_hydraulic_oil_group_C",
    "sensor_42_avg": "Nac_direction",
    "sensor_43_avg": "Nac_temperature_C_avg",
    "sensor_44": "Active_power_disconnected_Wh",
    "sensor_45": "Actice_power_connected_in_delta_Wh",
    "sensor_46": "Active_cpower_connected_in_star_Wh",
    "sensor_47": "Reactive_power_gen_disconnected_VArh",
    "sensor_48": "Reactive_power_gen_connected_delta_VArh",
    "sensor_49": "Reactive_power_gen_connected_start_VArh",
    "sensor_50": "Tot_active_power_Wh",
    "sensor_51": "Tot_reactive_power_VArh",
    "sensor_52_avg": "Rot_rpm_rpm_avg",
    "sensor_52_max": "Rot_rpm_rpm_max",
    "sensor_52_min": "Rot_rpm_rpm_min",
    "sensor_52_std": "Rot_rpm_rpm_std",
    "sensor_53_avg": "Temp_nose_cone_C_avg",
}

# Registry of per-profile maps. Add new OEM/client feeds here.
PROFILES = {
    "windfarm_a": WINDFARM_A_MAP,
}

# Meta columns (never used as ML features)
META_COLS = {
    "time_stamp", "asset_id", "id", "train_test", "status_type_id",
    "event_id", "event_label", "event_description", "source_file",
}


def normalise_columns(df: pd.DataFrame, profile: str = "windfarm_a") -> pd.DataFrame:
    """Rename raw SCADA columns to canonical names for the given profile."""
    mapping = PROFILES.get(profile, {})
    present = {raw: canon for raw, canon in mapping.items() if raw in df.columns}
    return df.rename(columns=present)


def canonical_feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric canonical columns available as candidate ML features."""
    import numpy as np
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in META_COLS]
