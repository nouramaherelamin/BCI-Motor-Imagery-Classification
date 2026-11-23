# Build context is the repository root (see docker-compose.yml) because the API
# imports the ML stage package (ml/) and its trained artifact.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/backend

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/bci_backend backend/bci_backend
COPY ml/__init__.py ml/adapter.py ml/preprocessing.py ml/
COPY ml/artifacts/bci_mi_2160_rf_v1.joblib ml/artifacts/
COPY ml/golden_samples ml/golden_samples

RUN useradd --create-home --uid 10001 appuser
USER appuser

ENV BCI_ENV=production \
    BCI_HOST=0.0.0.0 \
    BCI_PORT=8000 \
    BCI_LOG_LEVEL=INFO \
    BCI_MODEL_ARTIFACT=/app/ml/artifacts/bci_mi_2160_rf_v1.joblib \
    BCI_EXAMPLES_DIR=/app/ml/golden_samples

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import json,sys,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)); sys.exit(0 if r.get('model_loaded') else 1)"

CMD ["sh", "-c", "exec uvicorn bci_backend.main:app --host \"$BCI_HOST\" --port \"$BCI_PORT\" --log-level \"$(echo $BCI_LOG_LEVEL | tr A-Z a-z)\" --proxy-headers"]
