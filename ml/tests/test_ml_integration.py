"""Integration tests for ML bundle on first 2160 trials scope."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ml.adapter import BCIMotorImageryLoader
from ml.preprocessing import FEATURE_NAMES, preprocess_trial, transform_raw_trial

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "ml" / "artifacts" / "bci_mi_2160_rf_v1.joblib"
GOLDEN_DIR = REPO_ROOT / "ml" / "golden_samples"


def test_bundle_and_golden_samples():
    assert ARTIFACT_PATH.exists()
    loader = BCIMotorImageryLoader()
    predictor = loader.load(ARTIFACT_PATH)
    assert predictor.is_ready()

    golden_files = sorted(GOLDEN_DIR.glob("golden_*.json"))
    assert len(golden_files) >= 2
    for gf in golden_files:
        g = json.loads(gf.read_text())
        arr = np.asarray(g["api_request_payload"]["samples"], dtype=np.float64)
        pred = predictor.predict_from_array(arr)
        assert pred.label == g["expected_output"]["predicted_label"]
        assert pred.model_version == g["expected_output"]["model_version"]
        assert np.isclose(pred.probabilities["Left"], g["expected_output"]["probabilities"]["Left"], atol=1e-6)
        assert np.isclose(pred.probabilities["Right"], g["expected_output"]["probabilities"]["Right"], atol=1e-6)


def test_deterministic_preprocessing():
    rng = np.random.default_rng(7)
    raw = rng.normal(100.0, 15.0, size=(2500, 4))
    clean = preprocess_trial(raw)
    assert clean.shape == (2500, 4)
    assert clean.dtype == np.float32
    feats = transform_raw_trial(raw)
    assert feats.shape == (len(FEATURE_NAMES),)
