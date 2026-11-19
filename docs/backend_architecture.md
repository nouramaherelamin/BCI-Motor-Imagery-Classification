# Backend architecture

Status: backend foundation implemented; real model adapter pending.

## Components

| Component | Responsibility |
|---|---|
| `backend/bci_backend/main.py` | `create_app` factory, startup/lifespan, CORS and body-limit middleware, error handlers, router wiring. |
| `backend/bci_backend/config.py` | Reads/validates `BCI_*` environment settings; no credentials or implicit `.env` reads. |
| `backend/bci_backend/api/routes/v1.py` | HTTP only: health, model information and prediction. |
| `backend/bci_backend/schemas.py` | Pydantic request, response and error shapes; current signal input is explicitly provisional. |
| `backend/bci_backend/services/prediction.py` | Coordinates a request with the model manager and predictor. |
| `backend/bci_backend/services/model_manager.py` | `not_configured → loading → ready/failed` lifecycle and safe readiness checks. |
| `backend/bci_backend/contracts.py` | ML-framework-neutral `Predictor`, `PredictorLoader` and validated `Prediction` result. |
| `backend/bci_backend/core/logging.py` | JSON logs; no raw request or EEG values are logged. |

There is no database or external queue because the repository has no persistence or asynchronous job requirement.

## Startup and model lifecycle

The app factory reads validated settings and creates a fresh model manager. With no `BCI_MODEL_ARTIFACT`, state remains `not_configured` and startup succeeds. With a configured artifact, the lifespan calls the supplied trusted loader, checks `is_ready()` and metadata, and marks the model `ready`. Missing artifact, absent loader, or loader failure sets `failed`; the API process remains available and prediction returns 503. The ML model is never replaced by a test mock.

The adapter contract is:

```python
class Predictor(Protocol):
    def predict(self, trial: TrialInput) -> Prediction: ...
    def is_ready(self) -> bool: ...
    def get_metadata(self) -> dict[str, object]: ...

class PredictorLoader(Protocol):
    def load(self, artifact: Path) -> Predictor: ...
```

`is_ready` is implemented by the adapter around its actual framework; the API does not assume a framework-specific loaded state or probability method.

## Request lifecycle

`HTTP request → bounded body read → Pydantic validation → route → PredictionService → ModelManager readiness → Predictor → validated response`

Health describes process availability and model status separately. Model-info distinguishes artifact configuration from successful loading and labels the input envelope `provisional`. The route layer does not perform EEG filtering or feature extraction.

## Errors and logging

- 413: request body exceeds configured limit.
- 422: invalid JSON shape or fields; validation errors omit request values.
- 503: no predictor or failed model load.
- 500: inference, metadata, or unexpected internal error; the client receives a generic message.

Responses use `{ "error": { "code": "...", "message": "...", "details": [...] } }` (details are included for validation only). Server logs are JSON and may include an exception traceback for operational diagnosis, but never include EEG payloads by design.

## Configuration and run modes

`Settings` validates environment, host, port, log level, CORS allowlist, model artifact path, and maximum prediction body size. Development/test expose Swagger/ReDoc/OpenAPI; production mode disables interactive docs. CORS is disabled unless exact origins are configured, with credentials disabled.

## ML integration boundary

The adapter must package the exact inference transform with the model or call a shared implementation used during training. The current input shape is a repository-derived proposal, not a scientifically finalized interface. Clean training filters continuous blocks; the current endpoint provides one file-sized trial. The ML developer must establish inference parity or request a context/stream-oriented API before enabling the adapter.
