"""
SCREEN 4 — Asset Health Certificate / Reports
  · JSON certificate (decision-support, with forensic fields + SHA-256 seal)
  · Downloadable PDF
  · Human-override logging + immutable audit log
"""
from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import get_db
from app.deps import get_current_user
from app.routers.dashboard import _data_source
from models.schemas import (
    AssetHealthCertificateResponse, SustainabilityKPIs, FinancialImpact, ComplianceItem,
)
from services import asset_health

router = APIRouter(prefix="/api/reports", tags=["screen 4 · asset health certificate"])


def _build_cert(db, user, turbine_id, persist, engineer=None):
    cert = asset_health.generate_certificate(db, user.id, turbine_id, persist=persist,
                                             certifying_engineer=engineer)
    if not cert:
        raise HTTPException(409, "No prediction available. Run inference first.")
    return cert


@router.get("/asset_health_certificate", response_model=AssetHealthCertificateResponse)
def asset_health_certificate(turbine_id: int = Query(...), persist: bool = Query(False),
                             db: Session = Depends(get_db), user=Depends(get_current_user)):
    cert = _build_cert(db, user, turbine_id, persist)
    t, site, pred = cert["turbine"], cert["site"], cert["prediction"]
    return AssetHealthCertificateResponse(
        certificate_ref=cert["certificate_ref"], turbine_id=t.id, turbine_name=t.name,
        oem=t.oem, model=t.model, site_name=site.name if site else "—",
        issued_at=cert["issued_at"], valid_until=cert["valid_until"],
        overall_health_score=cert["overall_health_score"], risk_class=cert["risk_class"],
        classification=cert["classification"], component_scores=cert["component_scores"],
        rul_estimates=cert["rul_estimates"], narrative=cert["narrative"],
        sustainability=SustainabilityKPIs(**asset_health.sustainability_kpis(db, user.id, t)),
        financial_impact=FinancialImpact(**asset_health.financial_impact(db, user.id, pred)),
        compliance=[ComplianceItem(**c) for c in asset_health.compliance_items()],
        model_version=pred.model_version, data_source=_data_source(db, user),
        content_hash=cert["content_hash"], data_coverage_pct=cert["data_coverage_pct"],
        model_training_window=cert["model_training_window"],
        calibration_status=cert["calibration_status"],
        certifying_engineer=cert["certifying_engineer"], limitations=cert["limitations"],
    )


@router.get("/asset_health_certificate.pdf")
def asset_health_certificate_pdf(turbine_id: int = Query(...),
                                 engineer: str | None = Query(None),
                                 db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Generate + download the certificate PDF (persists it and logs CERT_ISSUED)."""
    from services.certificate_pdf import render_certificate_pdf
    cert = _build_cert(db, user, turbine_id, persist=True, engineer=engineer)
    pdf_bytes = render_certificate_pdf(
        cert,
        asset_health.sustainability_kpis(db, user.id, cert["turbine"]),
        asset_health.financial_impact(db, user.id, cert["prediction"]),
        asset_health.compliance_items(),
    )
    fname = f"{cert['certificate_ref']}.pdf"
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.post("/certificate/override")
def log_override(payload: dict = Body(...), db: Session = Depends(get_db),
                 user=Depends(get_current_user)):
    """
    Record a human override (score change / sign-off / data exclusion) to the
    immutable audit log. Insurers assume the customer is adversarial — every
    manual change is recorded and never deleted.
    """
    from models import AuditLog
    event = payload.get("event_type", "SCORE_OVERRIDE")
    entry = AuditLog(
        user_id=user.id, event_type=event,
        turbine_id=payload.get("turbine_id"),
        certificate_ref=payload.get("certificate_ref"),
        actor=payload.get("actor") or user.email,
        detail={k: payload.get(k) for k in ("field", "old_value", "new_value", "reason", "excluded")},
    )
    db.add(entry)
    db.commit()
    return {"status": "logged", "event_type": event, "id": entry.id, "ts": entry.ts}


@router.get("/audit_log")
def audit_log(turbine_id: int | None = Query(None), limit: int = Query(100, ge=1, le=1000),
              db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Immutable audit trail — certificate issuance, overrides, sign-offs, exclusions."""
    from models import AuditLog
    q = select(AuditLog).where(AuditLog.user_id == user.id)
    if turbine_id:
        q = q.where(AuditLog.turbine_id == turbine_id)
    rows = db.execute(q.order_by(AuditLog.ts.desc()).limit(limit)).scalars().all()
    return [{
        "id": r.id, "ts": r.ts, "event_type": r.event_type, "turbine_id": r.turbine_id,
        "certificate_ref": r.certificate_ref, "actor": r.actor,
        "detail": r.detail, "content_hash": r.content_hash,
    } for r in rows]


@router.get("/certificates")
def list_certificates(db: Session = Depends(get_db), user=Depends(get_current_user)):
    from models import AssetHealthCertificate, Turbine
    rows = db.execute(
        select(AssetHealthCertificate)
        .where(AssetHealthCertificate.user_id == user.id)
        .order_by(AssetHealthCertificate.issued_at.desc())
    ).scalars().all()
    turbines = {t.id: t for t in db.execute(
        select(Turbine).where(Turbine.user_id == user.id)).scalars().all()}
    return [{
        "certificate_ref": c.certificate_ref,
        "turbine_id": c.turbine_id,
        "turbine_name": turbines[c.turbine_id].name if c.turbine_id in turbines else None,
        "overall_health_score": c.overall_health_score,
        "classification": c.classification, "risk_class": c.risk_class,
        "issued_at": c.issued_at, "valid_until": c.valid_until,
        "content_hash": c.content_hash,
    } for c in rows]
