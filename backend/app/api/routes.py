from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException

from backend.app.schemas import (
    AssistantAskRequest,
    PlanGenerateRequest,
    PlanRecheckRequest,
    PredictionInput,
    TaskConfirmRequest,
    TrainRequest,
)
from backend.app.services.predictor import get_engine, model_artifact_status, train_models


router = APIRouter(prefix="/api", tags=["crop-intelligence"])

_ADMIN_KEY_ENV = "ADMIN_API_KEY"
_ADMIN_KEY_DEFAULT = "crop-admin-secret-2026"  # override via env var in production


def _require_admin_key(x_admin_key: str | None) -> None:
    """Raise 403 if the X-Admin-Key header is missing or incorrect."""
    expected = os.environ.get(_ADMIN_KEY_ENV, _ADMIN_KEY_DEFAULT)
    if not x_admin_key or x_admin_key != expected:
        raise HTTPException(
            status_code=403,
            detail="Admin key required. Set X-Admin-Key header with the correct value.",
        )


@router.get("/health")
def health() -> dict[str, object]:
    status = model_artifact_status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail={"status": "models-not-ready", **status})
    return {"status": "ok", **status}


@router.get("/metadata")
def metadata() -> dict:
    try:
        engine = get_engine()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"Models are not ready: {exc}") from exc
    return engine.bundle.metadata["dataset_summary"] | {
        "training_report": engine.bundle.metadata["training_report"],
        "xai_assets": engine.bundle.metadata.get("xai_assets", {}),
    }


@router.post("/predict")
def predict(request: PredictionInput) -> dict:
    try:
        engine = get_engine()
        payload = request.model_dump()
        top_k = payload.pop("top_k", 3)
        return engine.predict(payload, top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Trained models not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/train")
def train(
    request: TrainRequest,
    x_admin_key: str | None = Header(default=None),
) -> dict:
    """Retrain all models from disk CSVs.

    Requires the ``X-Admin-Key`` header to be set correctly.
    Set the ``ADMIN_API_KEY`` environment variable to override the default key.
    """
    _require_admin_key(x_admin_key)
    try:
        return train_models(data_dir=request.data_dir)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Active Plan Store (in-memory persistent session) ─────────────────────────
_ACTIVE_PLANS: dict[str, dict] = {}


@router.post("/plan/generate")
def generate_plan(request: PlanGenerateRequest) -> dict:
    from backend.ml.plan_generator import generate_sowing_plan
    from datetime import date
    sowing_date = request.sowing_date or date.today().isoformat()
    try:
        plan = generate_sowing_plan(
            crop_name=request.crop_name,
            sowing_date_str=sowing_date,
            area_acres=request.area_acres,
            farmer_budget_inr=request.farmer_budget_inr,
            irrigation_source=request.irrigation_source,
        )
        from backend.storage.plan_db import save_plan
        save_plan(plan)
        _ACTIVE_PLANS[plan["plan_id"]] = plan
        return plan
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate plan: {exc}") from exc


@router.get("/plan/{plan_id}")
def get_plan(plan_id: str) -> dict:
    from backend.storage.plan_db import get_plan_by_id
    plan = get_plan_by_id(plan_id) or _ACTIVE_PLANS.get(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found.")
    return plan


@router.post("/plan/task/confirm")
def confirm_task(request: TaskConfirmRequest) -> dict:
    from datetime import datetime
    from backend.storage.plan_db import update_task_completion, get_plan_by_id

    updated_plan = update_task_completion(
        plan_id=request.plan_id,
        task_id=request.task_id,
        is_completed=request.is_completed,
        completed_at=datetime.now().isoformat() if request.is_completed else None,
    )

    if not updated_plan:
        # Fallback to in-memory if not in db
        plan = _ACTIVE_PLANS.get(request.plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan '{request.plan_id}' not found.")
        found = False
        for task in plan.get("tasks", []):
            if task.get("task_id") == request.task_id:
                task["is_completed"] = request.is_completed
                task["completed_at"] = datetime.now().isoformat() if request.is_completed else None
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail=f"Task '{request.task_id}' not found in plan.")
        plan["completed_tasks_count"] = sum(1 for t in plan["tasks"] if t.get("is_completed"))
        updated_plan = plan

    _ACTIVE_PLANS[request.plan_id] = updated_plan
    return {
        "status": "success",
        "plan_id": request.plan_id,
        "task_id": request.task_id,
        "is_completed": request.is_completed,
        "completed_tasks_count": updated_plan["completed_tasks_count"],
        "total_tasks_count": len(updated_plan["tasks"]),
    }


@router.post("/plan/recheck")
def recheck_plan(request: PlanRecheckRequest) -> dict:
    from backend.notify.escalation_engine import recalibrate_entire_plan
    from backend.storage.plan_db import get_plan_by_id, update_plan_tasks

    plan = request.plan
    if not plan and request.plan_id:
        plan = get_plan_by_id(request.plan_id) or _ACTIVE_PLANS.get(request.plan_id)

    if not plan:
        raise HTTPException(status_code=400, detail="Either plan_id or plan object must be provided.")

    recalibrated = recalibrate_entire_plan(plan, current_weather=request.weather)
    if recalibrated.get("plan_id"):
        update_plan_tasks(recalibrated["plan_id"], recalibrated.get("tasks", []))
        _ACTIVE_PLANS[recalibrated["plan_id"]] = recalibrated
    return recalibrated


@router.post("/assistant/ask")
def ask_assistant(request: AssistantAskRequest) -> dict:
    from backend.rag.assistant import answer_farmer_query
    return answer_farmer_query(
        query=request.query,
        crop_name=request.crop_name,
        farmer_context=request.farmer_context,
        language=request.language,
    )

