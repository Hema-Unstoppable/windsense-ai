"""
Config-driven onboarding endpoints — list site mapping files, view a mapping,
and run mapping validation + a data-quality report from a data sample.
No per-site code: a new site is just a YAML/JSON mapping file.
"""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Depends, Query, HTTPException

from config import settings
from app.deps import get_current_user

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])

MAP_DIR = Path(__file__).resolve().parents[2] / "ingestion" / "mappings"


@router.get("/mappings")
def list_mappings(user=Depends(get_current_user)):
    from ingestion.config_ingest import load_mapping
    out = []
    for p in sorted(MAP_DIR.glob("*.y*ml")) + sorted(MAP_DIR.glob("*.json")):
        try:
            cfg = load_mapping(p)
            out.append({"file": p.name, "site": cfg.get("site"), "country": cfg.get("country"),
                        "n_mappings": len(cfg.get("mappings", [])),
                        "sample_rate_seconds": cfg.get("sample_rate_seconds"),
                        "timezone": cfg.get("timezone")})
        except Exception as e:
            out.append({"file": p.name, "error": str(e)})
    return out


@router.get("/mapping/{name}")
def get_mapping(name: str, user=Depends(get_current_user)):
    from ingestion.config_ingest import load_mapping
    p = MAP_DIR / name
    if not p.exists():
        raise HTTPException(404, "Mapping not found")
    cfg = load_mapping(p)
    return {"site": cfg.get("site"), "country": cfg.get("country"),
            "timezone": cfg.get("timezone"), "sample_rate_seconds": cfg.get("sample_rate_seconds"),
            "timestamp_column": cfg.get("timestamp_column"), "asset_id_column": cfg.get("asset_id_column"),
            "mappings": cfg.get("mappings", [])}


@router.post("/validate")
def validate(profile: str = Query("windfarm_a.yaml"), rows: int = Query(40000, ge=1000, le=300000),
             user=Depends(get_current_user)):
    """Run mapping validation + data-quality report on a sample of the site CSV."""
    import pandas as pd
    from ingestion.config_ingest import load_mapping, validate_mapping, data_quality_report

    p = MAP_DIR / profile
    if not p.exists():
        raise HTTPException(404, f"Mapping {profile} not found")
    cfg = load_mapping(p)
    try:
        df = pd.read_csv(settings.CSV_PATH, nrows=rows, low_memory=False)
    except Exception as e:
        raise HTTPException(409, f"Could not read site CSV: {e}")

    validation = validate_mapping(cfg, df.columns.tolist())
    dq = data_quality_report(df, cfg)
    return {
        "site": cfg.get("site"), "country": cfg.get("country"),
        "timezone": cfg.get("timezone"), "sample_rate_seconds": cfg.get("sample_rate_seconds"),
        "rows_sampled": len(df),
        "validation": validation,
        "data_quality": dq,
    }
