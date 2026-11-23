"""Version 1 API routes."""

import logging
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from ...schemas import (
    CHANNELS,
    SAMPLE_RATE_HZ,
    ErrorResponse,
    ExampleDetail,
    ExampleSummary,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TrialInput,
)
from ...services.model_manager import ModelStatus, ModelUnavailable
from ...services.prediction import PredictionService
from ...services.trial_file import TrialFileError, parse_trial_file

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


PREDICT_RESPONSES: dict[int | str, dict[str, object]] = {
    413: {"model": ErrorResponse, "description": "Request body exceeds the configured limit"},
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    500: {"model": ErrorResponse, "description": "Prediction or internal service failure"},
    503: {"model": ErrorResponse, "description": "Model is not ready or failed to load"},
}


def _run_prediction(trial: TrialInput, request: Request, source: str) -> PredictionResponse:
    started = time.perf_counter()
    try:
        result = PredictionService(request.app.state.model_manager).predict(trial)
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
        logger.exception("model_inference_failed source=%s", source)
        raise HTTPException(
            status_code=500,
            detail={"code": "prediction_failed", "message": "Model inference failed"},
        ) from exc
    logger.info(
        "prediction_served source=%s label=%s duration_ms=%.1f",
        source,
        result.label,
        (time.perf_counter() - started) * 1000,
    )
    return PredictionResponse(
        predicted_label=result.label,
        model_version=result.model_version,
        probabilities=result.probabilities,
    )


@router.post("/predict", response_model=PredictionResponse, responses=PREDICT_RESPONSES, tags=["prediction"])
def predict(payload: TrialInput, request: Request) -> PredictionResponse:
    """Classify one raw 10-second trial (2500 x [FZ, C3, CZ, C4] at 250 Hz) sent as JSON."""
    return _run_prediction(payload, request, source="json")


@router.post("/predict/file", response_model=PredictionResponse, responses=PREDICT_RESPONSES, tags=["prediction"])
async def predict_file(request: Request, file: UploadFile = File(..., description="Trial as .csv or .xlsx")) -> PredictionResponse:
    """Classify one raw trial uploaded as a CSV/Excel file with columns FZ, C3, CZ, C4 (Time optional)."""
    content = await file.read()
    try:
        samples = parse_trial_file(file.filename or "", content)
    except TrialFileError as exc:
        logger.info("trial_file_rejected reason=%s", exc)
        raise HTTPException(status_code=422, detail={"code": "invalid_file", "message": str(exc)}) from exc
    trial = TrialInput(sampling_rate_hz=SAMPLE_RATE_HZ, channels=CHANNELS, samples=samples.tolist())
    return _run_prediction(trial, request, source="file")


def _example_or_404(request: Request, example_id: str):
    example = request.app.state.examples.get(example_id)
    if example is None:
        raise HTTPException(status_code=404, detail={"code": "example_not_found", "message": "Example trial not found"})
    return example


@router.get("/examples", response_model=list[ExampleSummary], tags=["examples"])
def list_examples(request: Request) -> list[ExampleSummary]:
    """Real recorded trials (with ground-truth labels) that can be used to try the model."""
    return [
        ExampleSummary(id=e.id, source_file=e.source_file, ground_truth_label=e.ground_truth_label)
        for e in request.app.state.examples.list()
    ]


@router.get(
    "/examples/{example_id}",
    response_model=ExampleDetail,
    responses={404: {"model": ErrorResponse}},
    tags=["examples"],
)
def get_example(example_id: str, request: Request) -> ExampleDetail:
    """Full example trial; its `payload` can be sent unchanged to `POST /api/v1/predict`."""
    e = _example_or_404(request, example_id)
    return ExampleDetail(id=e.id, source_file=e.source_file, ground_truth_label=e.ground_truth_label, payload=e.trial)


@router.get(
    "/examples/{example_id}/csv",
    response_class=PlainTextResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["examples"],
)
def get_example_csv(example_id: str, request: Request) -> PlainTextResponse:
    """Example trial as a CSV file in the upload format accepted by `POST /api/v1/predict/file`."""
    e = _example_or_404(request, example_id)
    return PlainTextResponse(
        e.to_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{e.id}.csv"'},
    )
