"""
═══════════════════════════════════════════════════════════════════════
 DATA INGESTION SERVICE
═══════════════════════════════════════════════════════════════════════
Streams the SCADA CSV into the database for a tenant:
  • creates user → site → turbines
  • builds events from the labeled event windows
  • loads scada_timeseries (hot columns + canonical signal JSON)

Memory-safe: the 824 MB CSV is read in chunks. `prediction`-split rows
(the held-out anomaly windows) are prioritised so inference is meaningful,
then topped up with `train` rows to MAX_SCADA_ROWS.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from ml.canonical import normalise_columns, META_COLS
from ml.feature_engineering import FAULT_KEYWORDS

# Showcase mixed-fleet OEMs (the OEM-agnostic story) keyed by asset_id order
OEM_PROFILES = [
    ("Vestas", "V90", 3000),
    ("GE", "2.5XL", 2500),
    ("Enercon", "E-82", 2300),
    ("Nordex", "N117", 2400),
    ("Siemens Gamesa", "SG 2.1-114", 2100),
]

COMPONENT_FROM_DESC = {
    "Generator bearing failure": "Generator",
    "Gearbox failure": "Gearbox",
    "Hydraulic group": "Hydraulic",
    "Transformer failure": "Transformer",
}


def get_or_create_first_user(db: Session) -> object:
    from models import User, Site
    user = db.execute(
        select(User).where(User.email == settings.FIRST_USER_EMAIL)
    ).scalar()
    if user:
        return user
    user = User(email=settings.FIRST_USER_EMAIL, display_name=settings.FIRST_USER_NAME,
                company=settings.FIRST_USER_NAME, plan="pilot")
    db.add(user)
    db.flush()
    site = Site(user_id=user.id, name=settings.FIRST_SITE_NAME,
                country=settings.FIRST_SITE_COUNTRY, data_source="CSV_UPLOAD")
    db.add(site)
    db.commit()
    return user


def _ensure_turbines(db: Session, user, site_id: int, asset_ids: list[int]) -> dict[int, int]:
    """Create turbines for each asset_id. Returns {asset_id: turbine_db_id}."""
    from models import Turbine
    mapping = {}
    for i, aid in enumerate(sorted(asset_ids)):
        existing = db.execute(
            select(Turbine).where(Turbine.user_id == user.id,
                                  Turbine.external_ref == str(aid))
        ).scalar()
        if existing:
            mapping[aid] = existing.id
            continue
        oem, model, rated = OEM_PROFILES[i % len(OEM_PROFILES)]
        t = Turbine(user_id=user.id, site_id=site_id, external_ref=str(aid),
                    name=f"WT-{int(aid):03d}", oem=oem, model=model,
                    rated_power_kw=rated, data_source="CSV_UPLOAD")
        db.add(t)
        db.flush()
        mapping[aid] = t.id
    db.commit()
    return mapping


def build_events(db: Session, user, turbine_map: dict[int, int]) -> int:
    """Cheap 5-column scan to construct event windows."""
    from models import Event
    cols = ["asset_id", "event_id", "event_label", "event_description", "time_stamp"]
    agg: dict = {}
    for ch in pd.read_csv(settings.CSV_PATH, usecols=cols,
                          parse_dates=["time_stamp"], chunksize=300_000):
        for eid, g in ch.groupby("event_id"):
            cur = agg.setdefault(eid, {
                "asset_id": int(g["asset_id"].iloc[0]),
                "label": str(g["event_label"].iloc[0]),
                "desc": (None if pd.isna(g["event_description"].iloc[0])
                         else str(g["event_description"].iloc[0])),
                "start": g["time_stamp"].min(), "end": g["time_stamp"].max(),
            })
            cur["start"] = min(cur["start"], g["time_stamp"].min())
            cur["end"] = max(cur["end"], g["time_stamp"].max())

    written = 0
    for eid, e in agg.items():
        tid = turbine_map.get(e["asset_id"])
        if not tid:
            continue
        db.add(Event(
            user_id=user.id, turbine_id=tid, external_event_id=str(eid),
            label=e["label"], description=e["desc"],
            component=COMPONENT_FROM_DESC.get(e["desc"]),
            start_time=e["start"].to_pydatetime(),
            end_time=e["end"].to_pydatetime(), data_source="CSV_UPLOAD",
        ))
        written += 1
    db.commit()
    return written


def _canonical_signals(row: pd.Series) -> dict:
    """All canonical numeric signals for one row, NaNs dropped, JSON-safe."""
    out = {}
    for k, v in row.items():
        if k in META_COLS or k in ("y_multi", "y_binary"):
            continue
        if isinstance(v, (int, float, np.floating, np.integer)) and not (
                isinstance(v, float) and math.isnan(v)):
            out[k] = float(v)
    return out


def ingest_scada(db: Session, user, turbine_map: dict[int, int],
                 max_rows: int | None = None) -> int:
    """Stream SCADA rows into scada_timeseries (prediction split prioritised)."""
    from models import ScadaTimeseries
    max_rows = settings.MAX_SCADA_ROWS if max_rows is None else max_rows
    cap = math.inf if max_rows in (0, None) else max_rows

    written = 0
    batch = []
    # Two passes by priority: prediction rows first, then train rows.
    for split in ("prediction", "train"):
        if written >= cap:
            break
        for ch in pd.read_csv(settings.CSV_PATH, parse_dates=["time_stamp"],
                              chunksize=100_000, low_memory=False):
            ch = ch[ch["train_test"] == split]
            if ch.empty:
                continue
            ch = normalise_columns(ch)
            for _, row in ch.iterrows():
                if written >= cap:
                    break
                tid = turbine_map.get(int(row["asset_id"]))
                if not tid:
                    continue
                sig = _canonical_signals(row)
                batch.append(ScadaTimeseries(
                    user_id=user.id, turbine_id=tid,
                    ts=row["time_stamp"].to_pydatetime(),
                    status_type_id=(int(row["status_type_id"])
                                    if not pd.isna(row.get("status_type_id")) else None),
                    data_source="CSV_UPLOAD",
                    wind_speed=sig.get("Wind_speed_m/s_avg"),
                    power_kw=sig.get("Grid_power_kW_avg"),
                    rotor_rpm=sig.get("Rot_rpm_rpm_avg"),
                    gearbox_oil_temp=sig.get("Temp_oil_gearbox_C_avg"),
                    gen_bearing_de_temp=sig.get("Temp_gen_bearing_Drive_End_C_avg"),
                    nacelle_temp=sig.get("Nac_temperature_C_avg"),
                    ambient_temp=sig.get("Amb_temp_C_avg"),
                    signals=sig,
                ))
                written += 1
                if len(batch) >= 5000:
                    db.bulk_save_objects(batch)
                    db.commit()
                    batch = []
                    print(f"[ingest] {written:,} rows ...")
            if written >= cap:
                break
    if batch:
        db.bulk_save_objects(batch)
        db.commit()
    print(f"[ingest] done — {written:,} SCADA rows")
    return written


def seed_everything(db: Session) -> dict:
    """Full end-to-end seed for the first user."""
    cols = ["asset_id"]
    asset_ids = set()
    for ch in pd.read_csv(settings.CSV_PATH, usecols=cols, chunksize=300_000):
        asset_ids.update(int(a) for a in ch["asset_id"].unique())

    user = get_or_create_first_user(db)
    from models import Site
    site = db.execute(select(Site).where(Site.user_id == user.id)).scalar()
    turbine_map = _ensure_turbines(db, user, site.id, list(asset_ids))
    n_events = build_events(db, user, turbine_map)
    n_scada = ingest_scada(db, user, turbine_map)
    return {
        "user_id": user.id, "turbines": len(turbine_map),
        "events": n_events, "scada_rows": n_scada,
    }
