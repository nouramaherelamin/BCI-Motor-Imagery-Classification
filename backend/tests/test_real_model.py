"""End-to-end API tests against the real trained artifact and real recorded trials.

These use the ML stage's golden samples (real trials 930 and 1230 with known expected
model outputs), so they verify that the API serves the actual model's predictions.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bci_backend.config import Settings
from bci_backend.main import create_app, default_predictor_loader

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "ml" / "artifacts" / "bci_mi_2160_rf_v1.joblib"
GOLDEN_DIR = REPO_ROOT / "ml" / "golden_samples"
GOLDEN = sorted(GOLDEN_DIR.glob("golden_*.json"))


@pytest.fixture(scope="module")
def client():
    app = create_app(Settings(model_artifact=ARTIFACT, examples_dir=GOLDEN_DIR), default_predictor_loader())
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _golden(path: Path) -> dict:
    return json.loads(path.read_text())


def _csv_bytes(samples: list, with_time: bool = True) -> bytes:
    frame = pd.DataFrame(samples, columns=["FZ", "C3", "CZ", "C4"])
    if with_time:
        frame.insert(0, "Time", np.arange(len(frame)) / 250.0)
    return frame.to_csv(index=False).encode()


def test_real_model_is_loaded(client):
    health = client.get("/api/v1/health").json()
    assert health == {"status": "ok", "model_status": "ready", "model_loaded": True}
    info = client.get("/api/v1/model/info").json()
    assert info["model_version"] == "bci-mi-2160-rf-v1.0.0"
    assert info["metadata"]["feature_count"] == 76
    assert info["metadata"]["class_mapping"] == {"0": "Left", "1": "Right"}


@pytest.mark.parametrize("golden_path", GOLDEN, ids=lambda p: p.stem)
def test_json_prediction_matches_golden_output(client, golden_path):
    g = _golden(golden_path)
    response = client.post("/api/v1/predict", json=g["api_request_payload"])
    assert response.status_code == 200, response.text
    body = response.json()
    expected = g["expected_output"]
    assert body["predicted_label"] == expected["predicted_label"]
    assert body["model_version"] == expected["model_version"]
    for cls in ("Left", "Right"):
        assert body["probabilities"][cls] == pytest.approx(expected["probabilities"][cls], abs=1e-6)


@pytest.mark.parametrize("golden_path", GOLDEN, ids=lambda p: p.stem)
def test_csv_upload_matches_json_prediction(client, golden_path):
    g = _golden(golden_path)
    files = {"file": ("trial.csv", _csv_bytes(g["api_request_payload"]["samples"]), "text/csv")}
    response = client.post("/api/v1/predict/file", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predicted_label"] == g["expected_output"]["predicted_label"]
    assert body["probabilities"]["Left"] == pytest.approx(g["expected_output"]["probabilities"]["Left"], abs=1e-6)


def test_xlsx_upload_with_reordered_lowercase_columns(client):
    g = _golden(GOLDEN[0])
    frame = pd.DataFrame(g["api_request_payload"]["samples"], columns=["fz", "c3", "cz", "c4"])[["c4", "fz", "cz", "c3"]]
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    files = {"file": ("trial.xlsx", buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    response = client.post("/api/v1/predict/file", files=files)
    assert response.status_code == 200, response.text
    assert response.json()["predicted_label"] == g["expected_output"]["predicted_label"]


@pytest.mark.parametrize(
    ("filename", "content", "fragment"),
    [
        ("trial.txt", b"FZ,C3,CZ,C4\n1,2,3,4\n", "Unsupported file type"),
        ("trial.csv", b"", "empty"),
        ("trial.csv", b"FZ,C3,CZ\n1,2,3\n", "Missing required channel column(s): C4"),
        ("trial.csv", b"FZ,C3,CZ,C4\n" + b"1,2,3,4\n" * 10, "Expected exactly 2500 sample rows"),
        ("trial.csv", b"FZ,C3,CZ,C4\n" + b"1,2,3,4\n" * 2499 + b"1,abc,3,4\n", "Non-numeric value in column C3"),
        ("trial.csv", b"FZ,C3,CZ,C4\n" + b"1,2,3,4\n" * 2499 + b"1,2,,4\n", "Missing or non-finite value in column CZ"),
        ("trial.xlsx", b"not really an excel file", "Could not read the file"),
    ],
)
def test_invalid_uploads_return_clear_422(client, filename, content, fragment):
    response = client.post("/api/v1/predict/file", files={"file": (filename, content, "application/octet-stream")})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_file"
    assert fragment in error["message"]


def test_upload_without_file_field_is_validation_error(client):
    response = client.post("/api/v1/predict/file", data={"other": "x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_flat_signal_is_rejected_by_model_preprocessing(client):
    payload = {"sampling_rate_hz": 250.0, "channels": ["FZ", "C3", "CZ", "C4"], "samples": [[1.0, 1.0, 1.0, 1.0]] * 2500}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "prediction_failed"


def test_examples_catalogue_and_csv_round_trip(client):
    listing = client.get("/api/v1/examples").json()
    assert {e["ground_truth_label"] for e in listing} == {"Left", "Right"}
    first = listing[0]
    detail = client.get(f"/api/v1/examples/{first['id']}").json()
    assert len(detail["payload"]["samples"]) == 2500
    csv_response = client.get(f"/api/v1/examples/{first['id']}/csv")
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    from_csv = client.post("/api/v1/predict/file", files={"file": ("ex.csv", csv_response.content, "text/csv")}).json()
    from_json = client.post("/api/v1/predict", json=detail["payload"]).json()
    assert from_csv["predicted_label"] == from_json["predicted_label"]
    for cls in ("Left", "Right"):
        assert from_csv["probabilities"][cls] == pytest.approx(from_json["probabilities"][cls], abs=1e-9)
    assert client.get("/api/v1/examples/does-not-exist").status_code == 404
