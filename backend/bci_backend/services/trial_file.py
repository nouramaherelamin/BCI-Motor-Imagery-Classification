"""Parse an uploaded single-trial EEG file (CSV or Excel) into the API trial shape.

Accepted layout matches the project's raw recordings (``cellula_MI_data_<n>.csv``):
a header row with the channel columns ``FZ, C3, CZ, C4`` (any order, case-insensitive)
and exactly 2500 sample rows. An optional ``Time`` column (packet arrival time) and any
unnamed index column are ignored, as in the training pipeline which uses sample index as time.
"""

from __future__ import annotations

import io
from pathlib import PurePath

import numpy as np
import pandas as pd

from ..schemas import CHANNELS, SAMPLES_PER_FILE

SUPPORTED_EXTENSIONS = (".csv", ".xlsx")


class TrialFileError(ValueError):
    """User-facing problem with an uploaded trial file (safe to return to clients)."""


def _read_table(filename: str, content: bytes) -> pd.DataFrame:
    suffix = PurePath(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise TrialFileError(f"Unsupported file type '{suffix or 'none'}'. Upload a .csv or .xlsx file.")
    if not content:
        raise TrialFileError("The uploaded file is empty.")
    try:
        if suffix == ".csv":
            # round_trip parsing keeps values bit-identical to the JSON path.
            return pd.read_csv(io.BytesIO(content), float_precision="round_trip")
        return pd.read_excel(io.BytesIO(content), sheet_name=0, engine="openpyxl")
    except Exception as exc:  # pandas raises many parser-specific exception types
        raise TrialFileError(f"Could not read the file as {suffix[1:].upper()}. Check that it is a valid, uncorrupted file.") from exc


def parse_trial_file(filename: str, content: bytes) -> np.ndarray:
    """Return a float64 array of shape (2500, 4) in FZ, C3, CZ, C4 order."""
    frame = _read_table(filename, content)
    columns = {str(col).strip().upper(): col for col in frame.columns}
    missing = [ch for ch in CHANNELS if ch not in columns]
    if missing:
        raise TrialFileError(
            "Missing required channel column(s): "
            + ", ".join(missing)
            + ". The first row must be a header containing FZ, C3, CZ and C4."
        )
    data = frame[[columns[ch] for ch in CHANNELS]]
    if len(data) != SAMPLES_PER_FILE:
        raise TrialFileError(
            f"Expected exactly {SAMPLES_PER_FILE} sample rows (10 s at 250 Hz), found {len(data)}."
        )
    numeric = data.apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna() & ~data.isna()
    if bad.to_numpy().any():
        row, col = np.argwhere(bad.to_numpy())[0]
        raise TrialFileError(f"Non-numeric value in column {CHANNELS[col]} at data row {row + 1}.")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        row, col = np.argwhere(~np.isfinite(values))[0]
        raise TrialFileError(f"Missing or non-finite value in column {CHANNELS[col]} at data row {row + 1}.")
    return values
