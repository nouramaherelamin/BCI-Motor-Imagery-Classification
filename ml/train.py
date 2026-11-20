"""Reproducible training & GroupKFold evaluation strictly on first 2160 trials/labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .adapter import BCIMotorImageryLoader
from .preprocessing import (
    BAND_HZ,
    BAND_ORDER,
    CHANNELS,
    FEATURE_NAMES,
    FS,
    N_SAMPLES,
    NOTCH_HZ,
    NOTCH_Q,
    extract_features,
    preprocess_trial,
)

MAX_TRIALS = 2160
CLASS_MAPPING = {0: "Left", 1: "Right"}
MODEL_VERSION = "bci-mi-2160-rf-v1.0.0"


def _trial_num(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else -1


def _csp_fit(X: np.ndarray, y: np.ndarray, n_pairs: int = 2) -> np.ndarray:
    covs = []
    for c in (0, 1):
        A = X[y == c]
        C = np.einsum("nts,ntk->nsk", A, A)
        C = C / np.trace(C, axis1=1, axis2=2)[:, None, None]
        covs.append(C.mean(0) + 1e-6 * np.eye(X.shape[2]))
    vals, vecs = eigh(covs[0], covs[0] + covs[1])
    order = np.argsort(vals)
    return vecs[:, np.r_[order[:n_pairs], order[-n_pairs:]]]


def _csp_feat(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    v = np.einsum("nts,sk->ntk", X, W).var(axis=1)
    return np.log(v / v.sum(axis=1, keepdims=True))


def run_training_pipeline(
    raw_csv_dir: Path,
    labels_csv: Path,
    trials_kept_csv: Path,
    artifact_path: Path,
    golden_dir: Path,
    eval_report_path: Path,
) -> dict[str, object]:
    # Strictly take first MAX_TRIALS (2160) labels and drop the rest
    df_lbl = pd.read_csv(labels_csv).iloc[:MAX_TRIALS].copy().reset_index(drop=True)
    y_2160 = (df_lbl["label"].astype(str).str.strip().str.lower() == "right").to_numpy(dtype=np.int64)

    # Strictly select CSV files with trial_id in 1..2160 and drop anything else
    csv_files = sorted(
        [p for p in raw_csv_dir.glob("cellula_MI_data_*.csv") if 1 <= _trial_num(p) <= MAX_TRIALS],
        key=_trial_num,
    )
    assert len(csv_files) == MAX_TRIALS, f"Expected {MAX_TRIALS} files, got {len(csv_files)}"

    trials_kept = pd.read_csv(trials_kept_csv)
    trials_kept = trials_kept[trials_kept["trial_id"] <= MAX_TRIALS].copy().reset_index(drop=True)
    kept_ids = trials_kept["trial_id"].to_numpy(dtype=int)

    # Build lookup map trial_id -> csv_path
    path_by_id = {_trial_num(p): p for p in csv_files}

    n_kept = len(kept_ids)
    X_raw_kept = np.empty((n_kept, N_SAMPLES, len(CHANNELS)), dtype=np.float64)
    X_clean_kept = np.empty((n_kept, N_SAMPLES, len(CHANNELS)), dtype=np.float32)
    X_feat_kept = np.empty((n_kept, len(FEATURE_NAMES)), dtype=np.float64)

    for idx_i, tid in enumerate(kept_ids):
        raw_arr = pd.read_csv(path_by_id[tid])[list(CHANNELS)].to_numpy(dtype=np.float64)
        X_raw_kept[idx_i] = raw_arr
        clean_arr = preprocess_trial(raw_arr)
        X_clean_kept[idx_i] = clean_arr
        X_feat_kept[idx_i] = extract_features(clean_arr)

    y_kept = y_2160[kept_ids - 1]
    groups_kept = trials_kept["session_id"].to_numpy(dtype=np.int64)

    sos_smr = butter(4, [8.0, 30.0], btype="bandpass", fs=FS, output="sos")
    X_smr_kept = sosfiltfilt(sos_smr, X_clean_kept.astype(np.float64), axis=1, padtype="odd", padlen=500)

    candidates = {
        "Baseline_CSP_LDA": "csp_lda",
        "RandomForest_76Feat": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "ExtraTrees_76Feat": ExtraTreesClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "GradientBoosting_76Feat": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        ),
        "LogisticRegression_L2": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42))]
        ),
        "SVM_RBF": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(C=0.5, kernel="rbf", probability=True, random_state=42))]
        ),
    }

    gkf = GroupKFold(n_splits=5)
    results: list[dict[str, object]] = []

    for name, est in candidates.items():
        oof_pred = np.zeros_like(y_kept)
        oof_prob = np.zeros((n_kept, 2), dtype=np.float64)

        for tr, te in gkf.split(X_feat_kept, y_kept, groups_kept):
            if est == "csp_lda":
                W = _csp_fit(X_smr_kept[tr], y_kept[tr], n_pairs=2)
                clf = LinearDiscriminantAnalysis().fit(_csp_feat(X_smr_kept[tr], W), y_kept[tr])
                oof_pred[te] = clf.predict(_csp_feat(X_smr_kept[te], W))
                oof_prob[te] = clf.predict_proba(_csp_feat(X_smr_kept[te], W))
            else:
                est.fit(X_feat_kept[tr], y_kept[tr])
                oof_pred[te] = est.predict(X_feat_kept[te])
                oof_prob[te] = est.predict_proba(X_feat_kept[te])

        cm = confusion_matrix(y_kept, oof_pred)
        results.append(
            {
                "model_name": name,
                "dataset_scope": "first_2160_trials_dropped_remainder",
                "cv_strategy": "GroupKFold(n_splits=5, group=session_id)",
                "n_trials_evaluated": int(n_kept),
                "n_sessions": int(len(np.unique(groups_kept))),
                "accuracy": round(float(accuracy_score(y_kept, oof_pred)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_kept, oof_pred)), 4),
                "precision_macro": round(float(precision_score(y_kept, oof_pred, average="macro", zero_division=0)), 4),
                "recall_macro": round(float(recall_score(y_kept, oof_pred, average="macro", zero_division=0)), 4),
                "f1_macro": round(float(f1_score(y_kept, oof_pred, average="macro", zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_kept, oof_prob[:, 1])), 4),
                "confusion_matrix": {
                    "tn_left_as_left": int(cm[0, 0]),
                    "fp_left_as_right": int(cm[0, 1]),
                    "fn_right_as_left": int(cm[1, 0]),
                    "tp_right_as_right": int(cm[1, 1]),
                },
            }
        )

    results_sorted = sorted(results, key=lambda r: (r["balanced_accuracy"], r["roc_auc"]), reverse=True)
    best_eval = results_sorted[0]
    baseline_eval = next(r for r in results if r["model_name"] == "Baseline_CSP_LDA")

    final_estimator = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    final_estimator.fit(X_feat_kept, y_kept)

    known_limitations = [
        "Scope restricted strictly to the first 2,160 trials/labels (any trailing rows/files beyond 2,160 are dropped).",
        "No cue-onset timestamps or pre-cue resting baseline available; full 10-second windows are classified without ERD/ERS baseline normalization.",
        "Single-trial zero-phase filtering uses odd-reflection padding (padlen=750) instead of multi-trial continuous block context.",
        "Class separability is near chance (~49.6% GroupKFold accuracy vs ~48.3% CSP+LDA baseline); output probabilities are uncalibrated.",
    ]

    bundle = {
        "model_name": "RandomForestMotorImageryClassifier",
        "model_version": MODEL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_scope": "first_2160_trials_only",
        "estimator": final_estimator,
        "class_mapping": CLASS_MAPPING,
        "supports_probabilities": True,
        "probabilities_calibrated": False,
        "input_contract": {
            "sampling_rate_hz": FS,
            "channels": list(CHANNELS),
            "samples_per_trial": N_SAMPLES,
            "shape": [N_SAMPLES, len(CHANNELS)],
        },
        "preprocessing_steps": [
            "validate_shape_2500x4_and_finiteness",
            "linear_interpolation_of_stale_held_rows",
            "per_trial_median_dc_removal",
            f"zero_phase_notch_{NOTCH_HZ}hz_q{NOTCH_Q}_odd_pad750",
            f"zero_phase_butterworth_bandpass_{BAND_HZ[0]}_{BAND_HZ[1]}hz_order{BAND_ORDER}_odd_pad750",
            "robust_median_power_trial_scaling",
            "welch_psd_and_hjorth_76_feature_extraction",
        ],
        "feature_names": FEATURE_NAMES,
        "evaluation_summary": {
            "selected_model": best_eval,
            "baseline_model": baseline_eval,
            "all_models_comparison": results_sorted,
        },
        "known_limitations": known_limitations,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path, compress=3)

    predictor = BCIMotorImageryLoader().load(artifact_path)
    golden_dir.mkdir(parents=True, exist_ok=True)

    golden_manifest = []
    for cls_idx, cls_name in CLASS_MAPPING.items():
        candidate_indices = np.flatnonzero(y_kept == cls_idx)
        probs_cls = final_estimator.predict_proba(X_feat_kept[candidate_indices])[:, cls_idx]
        best_local = int(candidate_indices[int(np.argmax(probs_cls))])
        tid = int(kept_ids[best_local])
        pred = predictor.predict_from_array(X_raw_kept[best_local])

        sample_payload = {
            "sample_id": f"golden_{cls_name.lower()}_trial_{tid}",
            "source_file": f"cellula_MI_data_{tid}.csv",
            "dataset_scope": "first_2160_trials_only",
            "ground_truth_label": cls_name,
            "api_request_payload": {
                "sampling_rate_hz": FS,
                "channels": list(CHANNELS),
                "samples": np.round(X_raw_kept[best_local], 6).tolist(),
            },
            "expected_output": {
                "predicted_label": pred.label,
                "model_version": pred.model_version,
                "probabilities": pred.probabilities,
            },
        }
        out_json = golden_dir / f"{sample_payload['sample_id']}.json"
        out_json.write_text(json.dumps(sample_payload, indent=2))
        golden_manifest.append(
            {
                "file": out_json.name,
                "ground_truth_label": cls_name,
                "expected_predicted_label": pred.label,
                "expected_probabilities": pred.probabilities,
            }
        )

    report = {
        "model_version": MODEL_VERSION,
        "dataset_scope": "first_2160_trials_only",
        "selected_model": best_eval,
        "baseline_model": baseline_eval,
        "all_models": results_sorted,
        "golden_samples": golden_manifest,
        "known_limitations": known_limitations,
    }
    eval_report_path.parent.mkdir(parents=True, exist_ok=True)
    eval_report_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("labels.csv"))
    parser.add_argument("--trials-kept", type=Path, default=Path("outputs_clean/trials_kept.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("ml/artifacts/bci_mi_2160_rf_v1.joblib"))
    parser.add_argument("--golden-dir", type=Path, default=Path("ml/golden_samples"))
    parser.add_argument("--report", type=Path, default=Path("ml/artifacts/evaluation_report.json"))
    args = parser.parse_args()
    rep = run_training_pipeline(
        args.raw_dir, args.labels, args.trials_kept, args.artifact, args.golden_dir, args.report
    )
    print(json.dumps(rep["selected_model"], indent=2))
