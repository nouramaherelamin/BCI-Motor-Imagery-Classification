# BCI motor imagery backend

This service serves the project's trained motor-imagery classifier (`ml/`) over HTTP: request validation (JSON and CSV/XLSX upload), real model inference through `ml.adapter`, example trials, health/model information, safe error responses and JSON logging. Set `BCI_MODEL_ARTIFACT` to the `.joblib` bundle to enable predictions (the Docker image does this); without it, predictions return HTTP 503. See the root [README](../README.md) for Docker usage, the full API reference and known limitations.

## Architecture

- `bci_backend/main.py`: settings-based app factory, lifespan model initialization, middleware, error handlers, and router registration.
- `bci_backend/api/routes/v1.py`: health, model information, and prediction HTTP routes.
- `bci_backend/schemas.py`: provisional EEG request and response models.
- `bci_backend/services/model_manager.py`: model lifecycle states and loader coordination.
- `bci_backend/services/prediction.py`: prediction use case.
- `bci_backend/contracts.py`: framework-independent predictor, loader, and result contracts.
- `bci_backend/config.py`: validated `BCI_*` environment settings.
- `bci_backend/core/logging.py`: JSON logs that do not include request bodies or EEG samples.

More detail is in [backend architecture](../docs/backend_architecture.md), [API reference](../docs/backend_api.md), [integration status](../docs/backend_integration_status.md), and the [ML contract](../docs/ml_backend_contract.md).

## Requirements and setup

Python 3.10 or newer. A clean Python 3.14.10 install resolved FastAPI 0.165.0, Pydantic 2.14.3, Uvicorn 0.54.0, pytest 9.2.0, and httpx2 2.16.2. The original workspace environment was also tested (FastAPI 0.141.1 / Pydantic 2.13.5 / pytest 9.1.1).

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[test]'
```

The application reads settings from process environment variables. To use the sample values in `.env.example` locally:

```bash
set -a
source .env.example
set +a
```

Review CORS origins and host before exposing the service beyond localhost. The sample file contains no secrets.

## Run

```bash
python -m bci_backend
```

The default address is `http://127.0.0.1:8000`. In development, Swagger UI is at [`/docs`](http://127.0.0.1:8000/docs), OpenAPI JSON at [`/openapi.json`](http://127.0.0.1:8000/openapi.json), and ReDoc at [`/redoc`](http://127.0.0.1:8000/redoc). Interactive docs are disabled in production mode.

The equivalent Uvicorn command is:

```bash
uvicorn bci_backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `BCI_ENV` | `development` | `development`, `test`, or `production`; production disables interactive docs. |
| `BCI_HOST` | `127.0.0.1` | Development server bind address. |
| `BCI_PORT` | `8000` | Port, 1–65535. |
| `BCI_LOG_LEVEL` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG`. |
| `BCI_CORS_ORIGINS` | empty | Comma-separated exact origins; no wildcard/default origins are enabled. |
| `BCI_MODEL_ARTIFACT` | empty | Path to a configured artifact; no artifact means `not_configured`. |
| `BCI_MAX_REQUEST_BYTES` | `2000000` | Prediction request body limit, from 100,000 through 100,000,000 bytes. |

Settings are validated at startup. The API process stays available if a configured model artifact is missing or the adapter fails to load; health and model-info report `failed`, while prediction returns 503.

## Tests

```bash
python -m pytest -vv
python -m compileall -q bci_backend
```

Tests use a deterministic adapter injected into an isolated app instance. The default app never installs that test adapter. The test extra includes both `httpx` and `httpx2` to support Starlette's test client across the declared FastAPI range.

## Model integration

Implement `PredictorLoader.load(path)` and a `Predictor` with `is_ready()`, `predict(trial)`, and `get_metadata()`. The loader must return only after validating/initializing the real model. Configure `BCI_MODEL_ARTIFACT` and wire the trusted loader into `create_app(settings, predictor_loader=YourLoader())` in the deployment entrypoint. Do not load arbitrary plugins from untrusted environment input.

The adapter owns the train-equivalent transform. The current API input (one file-shaped 2,500×4 trial) is provisional and cannot be presumed equivalent to the training notebook's whole-block zero-phase filtering. The ML developer must confirm modality, context, feature order, labels, artifact dependencies, and golden sample behavior before this route is used for real inference.
