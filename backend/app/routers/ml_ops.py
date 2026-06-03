"""
ML operations — run inference, inspect the active model, trigger retraining.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import get_db
from app.deps import get_current_user
from models.schemas import RunInferenceRequest, RunInferenceResponse
from ml.inference import run_inference, load_model, reset_model_cache

router = APIRouter(prefix="/api/ml", tags=["ml ops"])


@router.post("/run_inference", response_model=RunInferenceResponse)
def api_run_inference(req: RunInferenceRequest,
                      db: Session = Depends(get_db), user=Depends(get_current_user)):
    result = run_inference(db, user.id, turbine_id=req.turbine_id, store=req.store)
    return RunInferenceResponse(**result)


@router.get("/model")
def active_model(db: Session = Depends(get_db), user=Depends(get_current_user)):
    from models import ModelRegistry
    reg = db.execute(
        select(ModelRegistry).where(ModelRegistry.is_active == True)  # noqa: E712
        .order_by(ModelRegistry.trained_at.desc()).limit(1)
    ).scalar()
    try:
        bundle = load_model()
        loaded = {"version": bundle["version"], "algorithm": bundle["algorithm"],
                  "n_features": len(bundle["features"]), "metrics": bundle.get("metrics")}
    except FileNotFoundError:
        loaded = None
    return {
        "loaded_model": loaded,
        "registry": None if not reg else {
            "version": reg.version, "algorithm": reg.algorithm,
            "trained_at": reg.trained_at, "n_train_rows": reg.n_train_rows,
            "metrics": reg.metrics, "n_features": len(reg.feature_list or []),
        },
    }


@router.post("/retrain")
def retrain(background: BackgroundTasks, algorithm: str | None = None,
            user=Depends(get_current_user)):
    """Kick off retraining in the background (production: a Celery/RQ job)."""
    def _job():
        from ml.train_model import train
        train(algorithm)
        reset_model_cache()
    background.add_task(_job)
    return {"status": "training_started",
            "note": "Model will hot-reload when complete. Poll GET /api/ml/model."}
