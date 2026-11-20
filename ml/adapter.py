"""Backend-compatible Predictor and PredictorLoader for BCI Motor Imagery (first 2160 trials)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from bci_backend.contracts import Prediction, Predictor
from bci_backend.schemas import TrialInput

from .preprocessing import CHANNELS, FEATURE_NAMES, N_SAMPLES, SAMPLE_RATE_HZ, transform_raw_trial


class BCIMotorImageryPredictor:
    def __init__(self, bundle: dict[str, Any], artifact_sha256: str) -> None:
        self._bundle = bundle
        self._artifact_sha256 = artifact_sha256
        self._estimator = bundle.get("estimator")
        self._class_mapping: dict[int, str] = {int(k): str(v) for k, v in bundle.get("class_mapping", {}).items()}
        self._feature_names: list[str] = list(bundle.get("feature_names", []))
        self._model_name: str = str(bundle.get("model_name", "RandomForestMotorImageryClassifier"))
        self._model_version: str = str(bundle.get("model_version", "bci-mi-2160-rf-v1.0.0"))
        self._supports_proba: bool = bool(bundle.get("supports_probabilities", hasattr(self._estimator, "predict_proba")))
        self._ready: bool = (
            self._estimator is not None
            and self._class_mapping == {0: "Left", 1: "Right"}
            and self._feature_names == FEATURE_NAMES
        )

    def is_ready(self) -> bool:
        return self._ready

    def get_metadata(self) -> dict[str, object]:
        capabilities = ["predict", "probabilities"] if self._supports_proba else ["predict"]
        return {
            "model_name": self._model_name,
            "model_version": self._model_version,
            "artifact_sha256": self._artifact_sha256,
            "capabilities": capabilities,
            "dataset_scope": "first_2160_files_and_labels_only",
            "input_contract": {
                "modality": "raw_trial_samples",
                "sampling_rate_hz": SAMPLE_RATE_HZ,
                "channels": list(CHANNELS),
                "shape": [N_SAMPLES, len(CHANNELS)],
                "dtype": "float64",
            },
            "preprocessing": self._bundle.get("preprocessing_steps", []),
            "feature_count": len(self._feature_names),
            "feature_names": self._feature_names,
            "class_mapping": {str(k): v for k, v in self._class_mapping.items()},
            "probabilities_calibrated": bool(self._bundle.get("probabilities_calibrated", False)),
            "evaluation_summary": self._bundle.get("evaluation_summary", {}),
            "known_limitations": self._bundle.get("known_limitations", []),
        }

    def predict_from_array(self, samples: np.ndarray) -> Prediction:
        if not self._ready:
            raise RuntimeError("Predictor is not ready")
        features = transform_raw_trial(samples).reshape(1, -1)
        pred_idx = int(self._estimator.predict(features)[0])
        if pred_idx not in self._class_mapping:
            raise ValueError(f"Unexpected predicted class index: {pred_idx}")
        label = self._class_mapping[pred_idx]

        probabilities: dict[str, float] | None = None
        if self._supports_proba and hasattr(self._estimator, "predict_proba"):
            raw_proba = self._estimator.predict_proba(features)[0]
            classes = [int(c) for c in getattr(self._estimator, "classes_", [0, 1])]
            prob_map = {self._class_mapping[c]: float(np.clip(p, 0.0, 1.0)) for c, p in zip(classes, raw_proba)}
            probabilities = {"Left": prob_map["Left"], "Right": prob_map["Right"]}

        return Prediction(label=label, model_version=self._model_version, probabilities=probabilities)

    def predict(self, trial: TrialInput) -> Prediction:
        if trial.sampling_rate_hz != SAMPLE_RATE_HZ:
            raise ValueError(f"Unsupported sampling rate: {trial.sampling_rate_hz}")
        if tuple(trial.channels) != CHANNELS:
            raise ValueError(f"Unsupported channel order: {trial.channels}")
        return self.predict_from_array(np.asarray(trial.samples, dtype=np.float64))


class BCIMotorImageryLoader:
    def load(self, artifact: Path) -> Predictor:
        sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        bundle = joblib.load(artifact)
        if not isinstance(bundle, dict):
            raise ValueError("Invalid model artifact: expected dictionary bundle")
        predictor = BCIMotorImageryPredictor(bundle=bundle, artifact_sha256=sha256)
        if not predictor.is_ready():
            raise ValueError("Loaded model bundle failed readiness/schema checks")
        return predictor
