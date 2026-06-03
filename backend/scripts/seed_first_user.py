"""
═══════════════════════════════════════════════════════════════════════
 END-TO-END SEED  for the first user (Wind Farm A)
═══════════════════════════════════════════════════════════════════════
Wires the CSV all the way through to live API data in ONE command:

    python -m backend.scripts.seed_first_user

Steps:
    1. create tables
    2. create user → site → turbines
    3. build events + ingest scada_timeseries (capped by MAX_SCADA_ROWS)
    4. (unless --skip-train) train the ML model
    5. run inference → write ml_predictions + risk_rankings
    6. generate an Asset Health Certificate per turbine

Flags:
    --skip-train     reuse an existing trained model (faster re-runs)
    --skip-ml        data only (no train, no inference)
"""
from __future__ import annotations

import argparse
import sys
import time

try:  # ensure box-drawing/arrow chars print on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import SessionLocal, Base, engine
import models  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-ml", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("\n══════════ WindSense AI — End-to-End Seed ══════════\n")

    # 1. tables
    print("[1/6] Creating tables ...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # 2-3. data
        from services.ingestion import seed_everything
        print("[2/6] + [3/6] Seeding user/site/turbines + events + SCADA ...")
        info = seed_everything(db)
        print(f"        → {info}")

        if args.skip_ml:
            _summary(t0, info, trained=False, scored=None)
            return

        # 4. train
        if not args.skip_train:
            print("[4/6] Training ML model (streaming balanced sample) ...")
            from ml.train_model import train
            summary = train()
            print(f"        → {summary['version']} | metrics={summary['metrics']}")
        else:
            print("[4/6] Skipping training (reusing existing model).")

        # 5. inference
        print("[5/6] Running inference (writing predictions + risk rankings) ...")
        from ml.inference import run_inference, reset_model_cache
        reset_model_cache()
        from models import User
        user = db.query(User).first()
        result = run_inference(db, user.id)
        print(f"        → {result}")

        # 6. certificates
        print("[6/6] Generating Asset Health Certificates ...")
        from services.asset_health import generate_certificate
        from models import Turbine
        turbines = db.query(Turbine).filter(Turbine.user_id == user.id).all()
        n_cert = 0
        for t in turbines:
            if generate_certificate(db, user.id, t.id, persist=True):
                n_cert += 1
        print(f"        → {n_cert} certificates generated")

        _summary(t0, info, trained=not args.skip_train, scored=result)
    finally:
        db.close()


def _summary(t0, info, trained, scored):
    print("\n══════════ SEED COMPLETE ══════════")
    print(f"  Turbines    : {info['turbines']}")
    print(f"  Events      : {info['events']}")
    print(f"  SCADA rows  : {info['scada_rows']:,}")
    if scored:
        print(f"  Predictions : {scored['predictions_written']}")
        print(f"  Risk ranked : {scored['risk_rankings_written']}")
        print(f"  Model       : {scored['model_version']}")
    print(f"  Elapsed     : {round(time.time() - t0, 1)}s")
    print("\nNext (from backend/):  uvicorn app.main:app --reload   ->   http://localhost:8000/docs\n")


if __name__ == "__main__":
    main()
