# BCI Motor Imagery Classification

Classify whether a person is **imagining LEFT or RIGHT hand movement** from EEG alone, and serve the
trained model through a web demo.

The project follows the three stages in the brief (`Task.pdf`):

| Stage | Status | Where |
|---|---|---|
| 1. Preprocessing / EDA | Done | `notebooks/`, `outputs/`, `outputs_clean/`, `ml/preprocessing.py` |
| 2. Modeling | Done | `ml/train.py`, `ml/artifacts/` (trained model + evaluation report) |
| 3. Backend + Frontend | Done | `backend/` (FastAPI), `frontend/` (static web UI), `docker-compose.yml` |

---

## Quick start

```bash
git clone <repo-url>
cd BCI-Motor-Imagery-Classification
docker compose up --build
```

Then open **http://localhost:8080**, choose an input (upload a trial file or pick a real example trial),
and press **Predict**.

## Application URLs

| URL | What |
|---|---|
| http://localhost:8080 | Web frontend |
| http://localhost:8080/docs | Swagger UI (proxied by the frontend server) |
| http://localhost:8000/api/v1/health | Backend, direct access |
| http://localhost:8000/docs | Swagger UI, direct (`/redoc` and `/openapi.json` too) |

Interactive docs are hidden when `BCI_ENV=production`.

---

## Architecture

```
 Browser ──► frontend (nginx :80, published on :8080)
              ├─ /            static HTML/CSS/JS
              └─ /api/*, /docs ──proxy──► backend (FastAPI + Uvicorn :8000)
                                             ├─ input validation (JSON / CSV / XLSX)
                                             ├─ ml.preprocessing  (same code used in training)
                                             └─ ml.adapter → RandomForest artifact (.joblib)
```

* **Frontend:** plain HTML/CSS/JavaScript served by nginx. nginx also reverse-proxies `/api` to the backend
  over the internal Docker network, so the browser only talks to one origin (no CORS needed). The backend
  is resolved at request time, so the frontend starts and recovers even if the backend restarts.
* **Backend:** FastAPI app (`backend/bci_backend`). On startup it loads the trained artifact through the ML
  stage's adapter (`ml/adapter.py`), which applies **exactly** the training-time preprocessing
  (`ml/preprocessing.py`): stale-sample interpolation → median DC removal → 50 Hz notch →
  1–40 Hz Butterworth band-pass → robust scaling → Welch band power, asymmetry and Hjorth features (76) →
  RandomForest.
* **No database:** the API is stateless. The model artifact and example trials are baked into the image.
* **Health checks:** the backend container is healthy only once the model is loaded. The frontend waits for
  that (`depends_on: condition: service_healthy`).

## Technologies

* Python 3.12, FastAPI, Pydantic v2, Uvicorn, python-multipart
* NumPy 1.26, SciPy 1.11, scikit-learn **1.4.0** (the version the artifact was trained and pickled with), joblib,
  pandas + openpyxl (CSV/XLSX parsing)
* nginx 1.27 (static frontend + reverse proxy), vanilla JavaScript (no build step)
* Docker + Docker Compose, pytest

## Project structure

```
.
├── docker-compose.yml          # one-command startup
├── .env.example                # optional compose overrides (no secrets)
├── .dockerignore
├── backend/
│   ├── Dockerfile              # build context = repo root (needs ml/)
│   ├── requirements.txt        # pinned runtime stack (matches the artifact)
│   ├── requirements-dev.txt    # + pytest, httpx
│   ├── pyproject.toml
│   ├── bci_backend/
│   │   ├── main.py             # app factory, model loading, middleware, error handlers
│   │   ├── config.py           # BCI_* environment settings
│   │   ├── schemas.py          # request/response models
│   │   ├── contracts.py        # Predictor / PredictorLoader protocols
│   │   ├── api/routes/v1.py    # HTTP endpoints
│   │   └── services/           # model manager, prediction, file parsing, example trials
│   └── tests/                  # API tests (stubbed + real model)
├── frontend/
│   ├── Dockerfile, nginx.conf
│   └── index.html, styles.css, app.js
├── ml/
│   ├── preprocessing.py        # shared preprocessing + feature extraction
│   ├── adapter.py              # loads the artifact, implements the backend Predictor
│   ├── train.py                # training / evaluation script (not run by the app)
│   ├── artifacts/              # bci_mi_2160_rf_v1.joblib + evaluation_report.json
│   ├── golden_samples/         # 2 real trials with expected model outputs
│   └── tests/
├── notebooks/, outputs/, outputs_clean/   # data validation, cleaning, EDA (stage 1)
├── docs/                       # earlier backend/ML contract notes and audits
└── labels.csv
```

## Prerequisites

* Docker Engine with the Compose plugin (`docker compose version`). Nothing else is required to run the app.
* For running tests outside Docker: Python 3.12 (the pinned scikit-learn 1.4.0 has no wheels for very new
  Python versions).

## Environment variables

All have working defaults. To override them, copy `.env.example` to `.env`; Compose reads it automatically.
No secrets are needed.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `FRONTEND_PORT` | `8080` | compose | Host port for the web UI |
| `BACKEND_PORT` | `8000` | compose | Host port for the API |
| `BCI_ENV` | `development` (compose) | backend | `development`, `test` or `production` (production hides `/docs`) |
| `BCI_LOG_LEVEL` | `INFO` | backend | Log level (JSON logs; EEG samples are never logged) |
| `BCI_MAX_REQUEST_BYTES` | `2000000` | backend | Max body size for prediction requests (413 above it) |
| `BCI_CORS_ORIGINS` | `http://localhost:8080` | backend | Comma-separated origins allowed to call the API directly |
| `BCI_MODEL_ARTIFACT` | set in image | backend | Path to the `.joblib` model bundle |
| `BCI_EXAMPLES_DIR` | set in image | backend | Directory of example trials (golden samples) |
| `BCI_HOST` / `BCI_PORT` | `0.0.0.0` / `8000` | backend | Bind address inside the container |

## Running with Docker

```bash
docker compose up --build        # build and start (Ctrl+C to stop)
docker compose up --build -d     # same, in the background
docker compose ps                # both services should be "healthy"
docker compose logs -f backend   # backend logs
docker compose down              # stop and remove containers
```

---

## Input format

One trial = **10 s of raw EEG at 250 Hz = 2500 samples × 4 channels in the order FZ, C3, CZ, C4**. This is the
same layout as the dataset's `cellula_MI_data_<n>.csv` files.

* **File upload (`.csv` or `.xlsx`):** the first row is a header containing `FZ`, `C3`, `CZ` and `C4`. Column
  order and letter case don't matter. A `Time` column is allowed and ignored, as in training, where the sample
  index is used as time. Exactly 2500 numeric rows, with no empty cells.
* **JSON:** `{"sampling_rate_hz": 250.0, "channels": ["FZ","C3","CZ","C4"], "samples": [[fz,c3,cz,c4], ...2500 rows]}`

You can download a ready-made CSV from `GET /api/v1/examples/{id}/csv`, or from the "download it as CSV" link in
the UI.

## API documentation

Base path: `/api/v1`. Errors always have the shape `{"error": {"code": "...", "message": "...", "details": [...]?}}`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Process health + model state (`ready`, `failed`, `not_configured`, `loading`) |
| GET | `/model/info` | Model name/version, input contract, preprocessing steps, CV metrics, known limitations |
| POST | `/predict` | Predict from a JSON trial |
| POST | `/predict/file` | Predict from an uploaded CSV/XLSX trial (`multipart/form-data`, field `file`) |
| GET | `/examples` | List of real example trials with their true label |
| GET | `/examples/{id}` | One example, with a `payload` ready to send to `/predict` |
| GET | `/examples/{id}/csv` | The example as an uploadable CSV file |

Status codes: `200` success · `404` unknown example · `413` body too large · `422` invalid input
(`validation_error` for JSON, `invalid_file` for uploads, with a readable message) · `500` the model could not
process the trial, e.g. a flat signal (`prediction_failed`) · `503` model not loaded.

### Example request / response

```bash
# Get a real trial as CSV and upload it
curl -s -o trial.csv http://localhost:8000/api/v1/examples/golden_right_trial_1230/csv
curl -s -X POST http://localhost:8000/api/v1/predict/file -F "file=@trial.csv"
```

```json
{
  "predicted_label": "Right",
  "model_version": "bci-mi-2160-rf-v1.0.0",
  "probabilities": {"Left": 0.22513132602622643, "Right": 0.7748686739737731}
}
```

```bash
# JSON endpoint: send an example payload
curl -s http://localhost:8000/api/v1/examples/golden_left_trial_930 \
  | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['payload']))" > trial.json
curl -s -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" --data @trial.json
# {"predicted_label":"Left","model_version":"bci-mi-2160-rf-v1.0.0","probabilities":{"Left":0.72673649936505,"Right":0.27326350063495025}}
```

Invalid upload:

```bash
printf 'FZ,C3\n1,2\n' > bad.csv
curl -s -X POST http://localhost:8000/api/v1/predict/file -F "file=@bad.csv"
# 422 {"error":{"code":"invalid_file","message":"Missing required channel column(s): CZ, C4. The first row must be a header containing FZ, C3, CZ and C4."}}
```

## Running tests

The suites cover API contract and validation tests (stubbed model), real-model tests that check the API
reproduces the ML stage's golden outputs through both JSON and CSV/XLSX upload, and the ML integration tests.

**Inside Docker (no local Python needed), after `docker compose build`:**

```bash
docker run --rm --user root -v "$PWD":/src:ro -w /src/backend \
  -e PYTHONPATH=/src:/src/backend -e PYTHONDONTWRITEBYTECODE=1 \
  -e BCI_MODEL_ARTIFACT= -e BCI_EXAMPLES_DIR= -e BCI_ENV=test \
  bci-mi-backend:latest sh -c "pip install -q pytest==8.2.2 httpx==0.27.0 && python -m pytest -q -p no:cacheprovider /src/backend/tests /src/ml/tests"
```

**Locally (Python 3.12):**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt
(cd backend && python -m pytest -q)
PYTHONPATH=.:backend python -m pytest -q ml/tests
```

To run the backend locally without Docker:

```bash
export PYTHONPATH=$PWD:$PWD/backend
export BCI_MODEL_ARTIFACT=$PWD/ml/artifacts/bci_mi_2160_rf_v1.joblib BCI_EXAMPLES_DIR=$PWD/ml/golden_samples
uvicorn bci_backend.main:app --port 8000
```

## Known limitations

* **The model's accuracy is at chance level.** The model's own evaluation (`ml/artifacts/evaluation_report.json`,
  5-fold GroupKFold by session, 1,222 trials) reports a balanced accuracy of **49.5%** and a ROC AUC of **0.49**.
  The CSP+LDA baseline is similar. The app serves this trained model faithfully, as required, and the UI shows
  this figure, but the predictions should not be relied on. Improving them is a modeling-stage task: for
  example, cue-onset timing or a pre-cue baseline would allow proper epoching and ERD normalization.
* Probabilities are the RandomForest's uncalibrated class scores (`probabilities_calibrated: false`).
* The model scope is the first 2,160 trials/labels. Each request is classified as an isolated 10 s trial. Training
  filtered continuous blocks, and single-trial filtering is an approximation of that (see
  `known_limitations` in `GET /api/v1/model/info`).
* Only the 4-channel, 250 Hz, 2500-sample format is accepted, plus `.csv` and `.xlsx` (not legacy `.xls`).
* The example trials are the two golden samples shipped with the ML stage. The raw dataset itself is not in the
  repository (see the data link in `Task.pdf`).
