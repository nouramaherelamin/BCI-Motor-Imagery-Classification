from pathlib import Path

from fastapi.testclient import TestClient

from bci_backend.config import Settings
from bci_backend.contracts import Prediction
from bci_backend.main import app, create_app


class StubPredictor:
    def predict(self, trial):
        assert trial.channels == ("FZ", "C3", "CZ", "C4")
        return Prediction(label="Left", model_version="test-fixture-1")

    def is_ready(self):
        return True

    def get_metadata(self):
        return {"model_name": "deterministic-test-fixture", "model_version": "test-fixture-1", "capabilities": ["predict"]}


class StubLoader:
    def load(self, artifact: Path):
        return StubPredictor()


class BrokenLoader:
    def load(self, artifact: Path):
        raise RuntimeError("private artifact details")


def payload():
    return {
        "sampling_rate_hz": 250.0,
        "channels": ["FZ", "C3", "CZ", "C4"],
        "samples": [[0.0, 1.0, 2.0, 3.0] for _ in range(2500)],
    }


def test_application_initializes_and_openapi_documents_routes():
    schema = app.openapi()
    assert set(schema["paths"]) == {
        "/api/v1/health",
        "/api/v1/model/info",
        "/api/v1/predict",
        "/api/v1/predict/file",
        "/api/v1/examples",
        "/api/v1/examples/{example_id}",
        "/api/v1/examples/{example_id}/csv",
    }
    assert schema["paths"]["/api/v1/predict"]["post"]["requestBody"]["required"] is True


def test_health_reports_process_up_and_model_not_configured():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_status": "not_configured", "model_loaded": False}


def test_model_info_marks_provisional_input_and_unknown_model():
    with TestClient(app) as client:
        response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is False
    assert body["loaded"] is False
    assert body["model_name"] is None
    assert body["model_version"] is None
    assert body["expected_input"]["contract_status"] == "provisional"
    assert body["metadata"] is None


def test_prediction_returns_structured_503_without_model():
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=payload())
    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": "model_unavailable", "message": "Prediction model is not available"}
    }


def test_configured_artifact_loads_real_adapter_and_predicts(tmp_path):
    artifact = tmp_path / "model.bundle"
    artifact.write_bytes(b"test fixture marker")
    application = create_app(Settings(model_artifact=artifact), StubLoader())
    with TestClient(application) as client:
        health = client.get("/api/v1/health").json()
        info = client.get("/api/v1/model/info").json()
        prediction = client.post("/api/v1/predict", json=payload())
    assert health["model_status"] == "ready"
    assert health["model_loaded"] is True
    assert info["configured"] is True and info["loaded"] is True
    assert info["model_version"] == "test-fixture-1"
    assert info["capabilities"] == ["predict"]
    assert prediction.status_code == 200
    assert prediction.json() == {
        "predicted_label": "Left",
        "model_version": "test-fixture-1",
        "probabilities": None,
    }


def test_model_load_failure_keeps_process_up_and_prediction_unavailable(tmp_path):
    artifact = tmp_path / "model.bundle"
    artifact.write_bytes(b"invalid fixture marker")
    application = create_app(Settings(model_artifact=artifact), BrokenLoader())
    with TestClient(application) as client:
        health = client.get("/api/v1/health")
        prediction = client.post("/api/v1/predict", json=payload())
    assert health.status_code == 200
    assert health.json()["model_status"] == "failed"
    assert prediction.status_code == 503
    assert prediction.json()["error"]["code"] == "model_load_failed"
    assert "private artifact details" not in prediction.text


def test_missing_artifact_is_reported_as_failed_not_ready(tmp_path):
    application = create_app(Settings(model_artifact=tmp_path / "missing.bundle"), StubLoader())
    with TestClient(application) as client:
        health = client.get("/api/v1/health").json()
    assert health["model_status"] == "failed"
    assert health["model_loaded"] is False


def test_configured_artifact_without_registered_loader_is_failed(tmp_path):
    artifact = tmp_path / "model.bundle"
    artifact.touch()
    application = create_app(Settings(model_artifact=artifact))
    with TestClient(application) as client:
        assert client.get("/api/v1/health").json()["model_status"] == "failed"


def test_model_not_ready_after_loader_is_a_load_failure(tmp_path):
    class NotReadyPredictor(StubPredictor):
        def is_ready(self):
            return False

    class NotReadyLoader:
        def load(self, artifact):
            return NotReadyPredictor()

    artifact = tmp_path / "model.bundle"
    artifact.touch()
    application = create_app(Settings(model_artifact=artifact), NotReadyLoader())
    with TestClient(application) as client:
        assert client.get("/api/v1/health").json()["model_status"] == "failed"


def test_validation_rejects_dimensions_fields_channel_order_and_rate():
    with TestClient(app, raise_server_exceptions=False) as client:
        short = payload()
        short["samples"].pop()
        assert client.post("/api/v1/predict", json=short).status_code == 422

        wrong_width = payload()
        wrong_width["samples"][0] = [0.0, 1.0, 2.0]
        assert client.post("/api/v1/predict", json=wrong_width).status_code == 422

        missing_field = payload()
        del missing_field["sampling_rate_hz"]
        assert client.post("/api/v1/predict", json=missing_field).status_code == 422

        empty = payload()
        empty["samples"] = []
        assert client.post("/api/v1/predict", json=empty).status_code == 422

        wrong_order = payload()
        wrong_order["channels"] = ["C3", "FZ", "CZ", "C4"]
        assert client.post("/api/v1/predict", json=wrong_order).status_code == 422

        wrong_rate = payload()
        wrong_rate["sampling_rate_hz"] = 256.0
        response = client.post("/api/v1/predict", json=wrong_rate)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


def test_validation_rejects_nan_and_infinity_without_echoing_values():
    with TestClient(app, raise_server_exceptions=False) as client:
        for value in ("NaN", "Infinity", "-Infinity"):
            body = f'{{"sampling_rate_hz":250.0,"channels":["FZ","C3","CZ","C4"],"samples":[[{value},1,2,3]]}}'
            response = client.post("/api/v1/predict", content=body, headers={"content-type": "application/json"})
            assert response.status_code == 422
            assert value not in response.text


def test_predictor_exception_returns_generic_500():
    class BrokenPredictor(StubPredictor):
        def predict(self, trial):
            raise RuntimeError("private model path")

    class BrokenInferenceLoader:
        def load(self, artifact):
            return BrokenPredictor()

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "model.bundle"
        artifact.touch()
        application = create_app(Settings(model_artifact=artifact), BrokenInferenceLoader())
        with TestClient(application, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/predict", json=payload())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "prediction_failed"
    assert "private model path" not in response.text


def test_invalid_model_output_is_not_serialized_as_success(tmp_path):
    class InvalidOutputPredictor(StubPredictor):
        def predict(self, trial):
            return Prediction(label="Unknown", model_version="bad-output")

    class InvalidOutputLoader:
        def load(self, artifact):
            return InvalidOutputPredictor()

    artifact = tmp_path / "model.bundle"
    artifact.touch()
    application = create_app(Settings(model_artifact=artifact), InvalidOutputLoader())
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/predict", json=payload())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


def test_probability_values_are_optional_and_validated(tmp_path):
    class ProbabilityPredictor(StubPredictor):
        def predict(self, trial):
            return Prediction(label="Right", model_version="test-prob-1", probabilities={"Left": 0.25, "Right": 0.75})

    class ProbabilityLoader:
        def load(self, artifact):
            return ProbabilityPredictor()

    artifact = tmp_path / "model.bundle"
    artifact.touch()
    application = create_app(Settings(model_artifact=artifact), ProbabilityLoader())
    with TestClient(application) as client:
        response = client.post("/api/v1/predict", json=payload())
    assert response.json()["probabilities"] == {"Left": 0.25, "Right": 0.75}


def test_content_length_limit_rejects_oversized_payload():
    application = create_app(Settings(max_request_bytes=100_000))
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/predict",
            content=b"x" * 100_001,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_cors_uses_only_configured_origins():
    application = create_app(Settings(cors_origins=("https://example.test",)))
    with TestClient(application) as client:
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://example.test",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://example.test"


def test_production_hides_interactive_docs():
    application = create_app(Settings(environment="production"))
    with TestClient(application) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404

