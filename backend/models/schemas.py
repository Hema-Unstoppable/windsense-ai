"""
Pydantic response/request models — the JSON contracts a frontend engineer
binds to directly. Grouped by the 4 UI screens.
"""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


# ── shared ────────────────────────────────────────────────────────────
class DataSourceInfo(BaseModel):
    type: str                       # SCADA_LIVE | CSV_UPLOAD | ...
    label: str                      # human label for the icon tooltip
    icon: str                       # frontend icon key
    last_updated: datetime | None = None


# ════════════════════════════════════════════════════════════════════
#  SCREEN 1 — Fleet Overview / Reliability Dashboard
#  GET /api/dashboard/fleet_overview
# ════════════════════════════════════════════════════════════════════
class FleetKPIs(BaseModel):
    turbines_monitored: int
    turbines_operating: int
    turbines_offline: int
    fleet_health_score: float            # 0..100
    critical_alerts: int
    total_active_alerts: int
    predicted_failures_30d: int
    fleet_availability: float            # %
    financial_exposure_eur: float


class RiskRow(BaseModel):
    turbine_id: int
    turbine_name: str
    site_name: str
    risk_score: float
    rpn: float
    risk_class: str
    rank: int
    health_score: float
    rul_days: float | None
    component: str | None
    fault_probability: float
    financial_exposure_eur: float        # = Expected Economic Loss (€) — ranking metric
    status: str                          # Critical | Warning | Operational
    # auditable EEL detail
    expected_loss_eur: float | None = None
    cost_breakdown: dict | None = None   # {parts, labour, crane, lost_revenue, total, ...}
    action_priority: str | None = None   # High | Medium | Low (AIAG-VDA, replaces raw RPN)
    iso14224_code: str | None = None
    severity_criteria: str | None = None
    recommended_action: str | None = None  # decision-support action label


class AnomalyAlert(BaseModel):
    turbine_id: int
    turbine_name: str
    severity: str                        # CRITICAL | WARNING
    component: str | None
    message: str
    ml_confidence: float
    detected_ago: str                    # "2h ago"


class FleetOverviewResponse(BaseModel):
    kpis: FleetKPIs
    risk_ranking: list[RiskRow]          # top N by risk
    anomaly_alerts: list[AnomalyAlert]
    data_source: DataSourceInfo
    generated_at: datetime


# ════════════════════════════════════════════════════════════════════
#  SCREEN 2 — Turbine Health & Predictions
#  GET /api/turbines/{id}/health_summary
# ════════════════════════════════════════════════════════════════════
class ShapFeature(BaseModel):
    feature: str
    value: float | None = None
    contribution: float                  # signed SHAP contribution
    pct: float                           # normalised % of |contribution|
    expected: bool | None = None         # physically expected driver for the mode?
    group: str | None = None             # collinearity group (if any)


class ExplainabilityIntegrity(BaseModel):
    alignment_score: float | None = None # fraction of attribution from expected signals
    caveat: str | None = None            # collinearity caveat
    unexpected_top: str | None = None    # flag if #1 driver is physically implausible
    verdict: str


class MaintenanceRecommendation(BaseModel):
    action: str                          # MONITOR | INSPECT_SOON | ESCALATE_INSPECT_NOW | ...
    action_label: str
    urgency: str                         # Low | Medium | High | Immediate
    headline: str
    rationale: list[str]                 # traceable evidence + rule that fired
    sensor_fault_suspected: bool = False
    model_conflict: bool = False
    classification: str = "DECISION SUPPORT"
    human_in_the_loop: bool = True
    disclaimer: str


class FailureModePrediction(BaseModel):
    component: str                       # Gearbox / Generator / Hydraulic / Transformer
    predicted_class: int
    probability: float
    severity: str                        # Critical | High | Medium | Low


class ComponentHealth(BaseModel):
    component: str
    health_score: float                  # 0..100
    rul_days: float | None


class TurbineHealthSummary(BaseModel):
    turbine_id: int
    turbine_name: str
    oem: str | None
    model: str | None
    site_name: str
    overall_health_score: float
    classification: str
    fault_probability: float              # calibrated
    raw_fault_probability: float | None = None
    calibrated: bool | None = None
    predicted_component: str | None
    confidence: float
    rul_days: float | None
    failure_mode_predictions: list[FailureModePrediction]
    component_health: list[ComponentHealth]
    shap_explanation: list[ShapFeature]
    explainability: ExplainabilityIntegrity | None = None
    recommendation: MaintenanceRecommendation | None = None
    narrative: str
    model_version: str
    data_source: DataSourceInfo
    last_prediction_at: datetime | None


# ════════════════════════════════════════════════════════════════════
#  SCREEN 3 — Time-Series & Anomaly Explorer
#  GET /api/turbines/{id}/timeseries?from=&to=&signals=
# ════════════════════════════════════════════════════════════════════
class TimeseriesPoint(BaseModel):
    ts: datetime
    values: dict[str, float | None]      # {"power_kw":.., "wind_speed":..}
    status_type_id: int | None = None
    is_anomaly: bool = False             # inside a labeled/predicted anomaly window


class AnomalyMarker(BaseModel):
    start: datetime
    end: datetime | None
    label: str
    component: str | None
    source: str                          # actual_event | ml_predicted


class TimeseriesResponse(BaseModel):
    turbine_id: int
    turbine_name: str
    signals: list[str]
    points: list[TimeseriesPoint]
    anomaly_markers: list[AnomalyMarker]
    resolution_minutes: int
    data_source: DataSourceInfo


# ════════════════════════════════════════════════════════════════════
#  SCREEN 4 — Asset Health Certificate / Reports
#  GET /api/reports/asset_health_certificate?turbine_id=
# ════════════════════════════════════════════════════════════════════
class ComplianceItem(BaseModel):
    item: str
    status: str                          # ok | overdue | not_completed
    detail: str | None = None


class SustainabilityKPIs(BaseModel):
    co2_avoided_tonnes: float
    energy_production_gwh: float
    capacity_factor_pct: float
    fleet_uptime_pct: float


class FinancialImpact(BaseModel):
    avoided_downtime_cost_eur: float
    opex_per_mwh_eur: float
    avoided_unplanned_premium_eur: float
    total_exposure_eur: float


class AssetHealthCertificateResponse(BaseModel):
    certificate_ref: str
    turbine_id: int
    turbine_name: str
    oem: str | None
    model: str | None
    site_name: str
    issued_at: datetime
    valid_until: datetime
    overall_health_score: float
    risk_class: str
    classification: str
    component_scores: dict[str, float]
    rul_estimates: dict[str, float]
    narrative: str
    sustainability: SustainabilityKPIs
    financial_impact: FinancialImpact
    compliance: list[ComplianceItem]
    model_version: str
    data_source: DataSourceInfo
    # forensic / tamper-evidence
    content_hash: str | None = None
    data_coverage_pct: float | None = None
    model_training_window: str | None = None
    calibration_status: str | None = None
    certifying_engineer: str | None = None
    limitations: str | None = None


# ── ML ops ────────────────────────────────────────────────────────────
class RunInferenceRequest(BaseModel):
    turbine_id: int | None = None        # None = whole fleet
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    store: bool = True


class RunInferenceResponse(BaseModel):
    status: str
    model_version: str
    turbines_scored: int
    predictions_written: int
    risk_rankings_written: int
    elapsed_seconds: float
