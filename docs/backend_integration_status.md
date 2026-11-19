# Backend integration status

Updated: 2026-11-19.

## Completed backend work

- FastAPI application factory with dev/test/production behavior.
- Validated environment settings and `.env.example`.
- JSON logging that avoids request bodies and EEG sample values.
- Configurable CORS allowlist and prediction-body size limit.
- Versioned health, model-info and prediction routes with OpenAPI error responses.
- Model lifecycle states (`not_configured`, `loading`, `ready`, `failed`) and loader/predictor protocols.
- Startup continues when no model is supplied or model loading fails; health reports state and prediction returns 503.
- Consistent error envelopes for validation, unavailable model, inference and unexpected errors.
- Deterministic loader/predictor fakes used only in tests.
- Setup, API, architecture and ML contract documentation.

## Verification

Run from `backend/`:

```bash
python -m pytest -vv
python -m compileall -q bci_backend
```

Verified in a fresh Python 3.14.10 environment (FastAPI 0.165.0, Pydantic 2.14.3, Uvicorn 0.54.0, pytest 9.2.0, httpx2 2.16.2): **24 passed, 0 failed, 0 skipped** with both `pytest -vv` and `python -m pytest -vv`; `python -m compileall -q bci_backend` passed. `pip install -e '.[test]'`, `pip check`, and the documented `.env.example` sourcing were verified. OpenAPI lists all three routes and prediction error codes 413/422/500/503. Both `python -m bci_backend` and the documented `uvicorn bci_backend.main:app --host 127.0.0.1 --port 8000 --reload` command started Uvicorn; live health/model-info returned 200 with `not_configured`, malformed prediction returned 422, and development Swagger returned 200. Deterministic mocks establish backend routing/lifecycle behavior only; they say nothing about scientific classifier performance.

## Still provisional

- Input modality: current schema accepts a single 2,500×4 repository-shaped file.
- 250 Hz and `FZ,C3,CZ,C4` metadata are repository-derived assumptions, not hardware configuration supplied by acquisition metadata.
- A single file may not reproduce whole-block zero-phase filtering used in training.
- Cue-locked epoch/baseline timing, raw sample units, subject/session identity, and final class mapping provenance remain outside backend evidence.
- No real model name, artifact, model version, capabilities, probability support, or dependency set is known.

## Exact ML handoff required

Member 3 must provide a real artifact and loader, an adapter implementing `is_ready`, `predict` and `get_metadata`, exact input shape/channel order, train-equivalent preprocessing and context requirements, class mapping, feature order if applicable, runtime dependencies, model version, optional probability semantics, and golden input/output samples. Evaluation should use recording/session grouped splits; a subject-grouped result is needed if subject identifiers become available.

## Connection point

Implement `PredictorLoader.load(artifact: Path) -> Predictor` in the ML integration package. Supply that loader to `create_app(settings, predictor_loader=...)` in the trusted deployment entrypoint, and set `BCI_MODEL_ARTIFACT` to the model bundle path. The loader must validate the artifact and return only after `is_ready()` and metadata checks pass. No test mock is included in the deployment entrypoint.

After wiring, compare the exact offline and backend transform for golden samples, verify expected outputs, then run the ML developer's grouped evaluation before enabling prediction to users.
