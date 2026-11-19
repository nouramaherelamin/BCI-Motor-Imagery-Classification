"""Version 1 API routes."""

import logging

from fastapi import APIRouter, HTTPException, Request

from ...schemas import ErrorResponse, HealthResponse, ModelInfoResponse, PredictionResponse, TrialInput
from ...services.model_manager import ModelStatus, ModelUnavailable
from ...services.prediction import PredictionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    models = request.app.state.model_manager
    return HealthResponse(status="ok", model_status=models.status.value, model_loaded=models.loaded)


@router.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info(request: Request) -> ModelInfoResponse:
    models = request.app.state.model_manager
    predictor_metadata = models.metadata() if models.loaded else None
    metadata = predictor_metadata or {}
    return ModelInfoResponse(
        configured=models.configured,
        loaded=models.loaded,
        status=models.status.value,
        model_name=metadata.get("model_name"),
        model_version=metadata.get("model_version"),
        capabilities=metadata.get("capabilities"),
        expected_input={
            "contract_status": "provisional",
            "sampling_rate_hz": 250.0,
            "samples": 2500,
            "channels": ["FZ", "C3", "CZ", "C4"],
        },
        metadata=predictor_metadata,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Request body exceeds the configured limit"},
        422: {"model": ErrorResponse, "description": "Request validation failed"},
        500: {"model": ErrorResponse, "description": "Prediction or internal service failure"},
        503: {"model": ErrorResponse, "description": "Model is not ready or failed to load"},
    },
    tags=["prediction"],
)
def predict(payload: TrialInput, request: Request) -> PredictionResponse:
    try:
        result = PredictionService(request.app.state.model_manager).predict(payload)
    except ModelUnavailable as exc:
        if exc.status is ModelStatus.FAILED:
            raise HTTPException(
                status_code=503,
                detail={"code": "model_load_failed", "message": "Prediction model failed to load"},
            ) from exc
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Prediction model is not available"},
        ) from exc
    except Exception as exc:
        logger.exception("model_inference_failed")
        raise HTTPException(
            status_code=500,
            detail={"code": "prediction_failed", "message": "Model inference failed"},
        ) from exc
    return PredictionResponse(
        predicted_label=result.label,
        model_version=result.model_version,
        probabilities=result.probabilities,
    )
