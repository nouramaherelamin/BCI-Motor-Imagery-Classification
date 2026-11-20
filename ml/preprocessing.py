"""Deterministic EEG preprocessing and feature extraction pipeline (2500 x 4 at 250 Hz)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

CHANNELS: tuple[str, str, str, str] = ("FZ", "C3", "CZ", "C4")
FS: float = 250.0
SAMPLE_RATE_HZ: float = FS
N_SAMPLES: int = 2500

NOTCH_HZ: float = 50.0
NOTCH_Q: float = 30.0
BAND_HZ: tuple[float, float] = (1.0, 40.0)
BAND_ORDER: int = 4
SMR_BAND_HZ: tuple[float, float] = (8.0, 30.0)

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu": (8.0, 12.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 30.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

_NOTCH_B, _NOTCH_A = iirnotch(NOTCH_HZ, NOTCH_Q, fs=FS)
_SOS_BP = butter(BAND_ORDER, BAND_HZ, btype="bandpass", fs=FS, output="sos")
_SOS_SMR = butter(BAND_ORDER, SMR_BAND_HZ, btype="bandpass", fs=FS, output="sos")


def get_feature_names() -> list[str]:
    names: list[str] = []
    for band in BANDS:
        for ch in CHANNELS:
            names.append(f"{ch}_{band}_logpow")
    for band in BANDS:
        for ch in CHANNELS:
            names.append(f"{ch}_{band}_relpow")
    for band in ("mu", "beta", "smr"):
        names.extend([
            f"LI_{band}_C4_minus_C3",
            f"AI_{band}_C4_C3",
            f"ratio_{band}_C3_CZ",
            f"ratio_{band}_C4_CZ",
        ])
    for ch in CHANNELS:
        names.extend([f"{ch}_hjorth_mob", f"{ch}_hjorth_comp"])
    return names


FEATURE_NAMES: list[str] = get_feature_names()


def validate_raw_trial(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.shape != (N_SAMPLES, len(CHANNELS)):
        raise ValueError(f"Expected shape {(N_SAMPLES, len(CHANNELS))}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("EEG trial contains NaN or infinite values")
    return arr


def interpolate_stale_rows(x: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(x), dtype=bool)
    mask[1:] = (np.diff(x, axis=0) == 0).all(axis=1)
    if not mask.any():
        return x
    good = ~mask
    if good.sum() < 2:
        raise ValueError("Too many stale samples to interpolate trial")
    idx = np.arange(len(x))
    out = x.copy()
    for c in range(x.shape[1]):
        out[:, c] = np.interp(idx, idx[good], x[good, c])
    return out


def preprocess_trial(raw_samples: np.ndarray) -> np.ndarray:
    x = validate_raw_trial(raw_samples)
    x = interpolate_stale_rows(x)
    x = x - np.median(x, axis=0, keepdims=True)
    padlen = min(750, len(x) - 1)
    x = filtfilt(_NOTCH_B, _NOTCH_A, x, axis=0, padtype="odd", padlen=padlen)
    x = sosfiltfilt(_SOS_BP, x, axis=0, padtype="odd", padlen=padlen)
    scale = float(np.sqrt(np.median(np.mean(x ** 2, axis=1))))
    if scale < 1e-6:
        raise ValueError("Signal variance is near zero")
    return (x / scale).astype(np.float32)


def extract_features(clean_trial: np.ndarray) -> np.ndarray:
    x = validate_raw_trial(clean_trial)
    freqs, psd = welch(x, fs=FS, nperseg=500, noverlap=250, axis=0)
    total_mask = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    total_pow = trapezoid(psd[total_mask, :], freqs[total_mask], axis=0) + 1e-12

    abs_log, rel_pow = {}, {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        bp = trapezoid(psd[mask, :], freqs[mask], axis=0) + 1e-12
        abs_log[band] = np.log10(bp)
        rel_pow[band] = bp / total_pow

    m_smr = (freqs >= SMR_BAND_HZ[0]) & (freqs <= SMR_BAND_HZ[1])
    abs_log["smr"] = np.log10(trapezoid(psd[m_smr, :], freqs[m_smr], axis=0) + 1e-12)

    feats: list[float] = []
    for band in BANDS:
        feats.extend(abs_log[band].tolist())
    for band in BANDS:
        feats.extend(rel_pow[band].tolist())

    c3, cz, c4 = 1, 2, 3
    for band in ("mu", "beta", "smr"):
        p_c3, p_cz, p_c4 = float(abs_log[band][c3]), float(abs_log[band][cz]), float(abs_log[band][c4])
        feats.extend([
            p_c4 - p_c3,
            (p_c4 - p_c3) / (abs(p_c4) + abs(p_c3) + 1e-9),
            p_c3 - p_cz,
            p_c4 - p_cz,
        ])

    x_smr = sosfiltfilt(_SOS_SMR, x, axis=0, padtype="odd", padlen=500)
    dx = np.diff(x_smr, axis=0)
    ddx = np.diff(dx, axis=0)
    v0 = np.var(x_smr, axis=0) + 1e-12
    v1 = np.var(dx, axis=0) + 1e-12
    v2 = np.var(ddx, axis=0) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / (mob + 1e-12)
    for c in range(4):
        feats.extend([float(mob[c]), float(comp[c])])

    return np.asarray(feats, dtype=np.float64)


def transform_raw_trial(raw_samples: np.ndarray) -> np.ndarray:
    return extract_features(preprocess_trial(raw_samples))
