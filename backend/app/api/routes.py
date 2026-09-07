from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException

from backend.app.schemas import PredictionInput, TrainRequest
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

