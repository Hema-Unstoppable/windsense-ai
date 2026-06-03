"""
ML Validation endpoints — serve the honest validation report to the
ML Validation & Testing dashboard page.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends
from app.deps import get_current_user

router = APIRouter(prefix="/api/validation", tags=["ml validation"])


@router.get("/report")
def get_report(user=Depends(get_current_user)):
    """Return the cached validation report (run it first if missing)."""
    from ml.validation import load_report
    rep = load_report()
    if rep is None:
        return {"status": "not_run",
                "message": "No validation report yet. POST /api/validation/run "
                           "or run: python -m scripts.run_validation"}
    return {"status": "ok", "report": rep}


@router.post("/run")
def run(background: BackgroundTasks, per_event_cap: int = 1500, user=Depends(get_current_user)):
    """Kick off a fresh validation run in the background (it is compute-heavy)."""
    def _job():
        from ml.validation import run_full_validation, save_report
        save_report(run_full_validation(per_event_cap=per_event_cap))
    background.add_task(_job)
    return {"status": "started",
            "message": "Validation running in background. Poll GET /api/validation/report."}
