# ML/backend integration contract

Status: **draft, pending member 3 and data-owner confirmation** (2026-11-19).

This contract records verified repository facts and explicitly labels open decisions. It is not a claim that the currently checked-in API can deliver scientifically valid classifications.

## Verified facts

- The source folder is described as 2,160 files named `cellula_MI_data_<n>.csv`.
- Each file is described as 2,500 samples with columns `Time,FZ,C3,CZ,C4`.
- The later clean handoff sets sample rate to 250 Hz and treats sample index as time because `Time` records packet arrival and is irregular.
- `labels.csv` has 2,400 rows. The latest cleaning notebook asserts the project owner confirmed label `i` maps to file `i` for the first 2,160 rows; older README/manifests still call this provisional. The confirmation needs provenance in the handoff.
- Clean training preparation filters continuous blocks and rejects block-edge trials. A single-file request does not itself include neighboring context.
- No trained model, model artifact, production preprocessing function, feature order, or final class-probability interface is currently present.

## Input contract: provisional API boundary

The API foundation currently accepts one record shaped like one repository CSV, without the `Time` column:

| Field | Current validation | Evidence/limit |
|---|---|---|
| `sampling_rate_hz` | Exactly `250.0` | Later handoff assumption supported by duration and 50 Hz peak, but device acquisition docs are absent. |
| `channels` | Exactly `FZ,C3,CZ,C4`, in this order | Source CSV layout and notebook configuration. |
| `samples` | Exactly 2,500 rows × 4 finite numbers | Source CSV shape; dtype is JSON number → Python float. Typical scale resembles ADC counts; no units conversion is performed by API. |
| time window | One 10-second file based on `2500 / 250` | File-shaped input only; active imagery interval within it is unknown. |

This shape is only an API scaffold. Member 3 must choose and document whether the real predictor accepts raw samples, a preprocessed epoch, or an ordered feature vector. If the model needs continuous-block context, the request schema must change to include an appropriate stream/context or the ML bundle must provide a validated streaming-equivalent transform.

### Preprocessing ownership

One implementation must own all train/inference transforms. Preferred ownership is an inference-ready pipeline packaged with the trained estimator (or a shared Python module imported by both training and backend). The backend must not independently recreate filter/feature calculations.

The latest clean notebook reports: interpolate stale rows, subtract block median, notch 50 Hz Q=30, fourth-order zero-phase Butterworth 1–40 Hz filtering across a continuous block, QC and edge rejection, then one robust scalar per retained trial. It also reports that the exported signal retains 2,500 samples. This transform cannot be silently assumed to work equivalently on one isolated test-time file. Member 3 must specify exact implementation, coefficients/library versions, required context, and behavior for stale samples/QC failures.

The older consolidated diagnostic uses a different pipeline (linear detrend, 0.5–40 Hz band-pass, notch, CAR), while the older feature audit has centered data and feature variants. Do not combine these pipelines or infer that the 120-feature CSV is the final model input.

## Model interface

The backend uses a framework-neutral `Predictor` protocol:

```python
class Predictor(Protocol):
    def predict(self, trial: TrialInput) -> Prediction: ...
    def get_metadata(self) -> dict[str, object]: ...
```

The concrete adapter is responsible for loading the real artifact and performing every declared model-side preprocessing step. Add a separate probability method only if the chosen model has a well-defined output; the API should not demand it from all estimators.

## Output contract

The initial response envelope is:

```json
{
  "predicted_label": "Left",
  "model_version": "<artifact version>",
  "probabilities": null
}
```

- `predicted_label`: final human-readable class label. Dataset evidence supports `Left` and `Right`, but model encoding/order must be confirmed by the model bundle.
- `model_version`: immutable bundle/version identifier supplied by the adapter.
- `probabilities`: null unless the actual model exposes per-class probabilities. If supplied, keys/order must match class metadata. Do not call an uncalibrated estimator score a confidence or calibrated probability.
- No `confidence` field is defined until calibration and meaning are documented.

No predictor installed → HTTP 503. Validation → HTTP 422. Model inference failure → generic HTTP 500 response; internal exception details are not sent to clients.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Process health and boolean `model_loaded`. |
| GET | `/api/v1/model/info` | Loaded state and adapter metadata, or `metadata: null`. |
| POST | `/api/v1/predict` | Validated trial to real adapter; disabled with 503 until installed. |

## Model delivery checklist for member 3

1. Trained artifact and serialization format, immutable model version/hash.
2. Exact accepted input modality: raw block/trial, processed epoch, or features.
3. Input shape, numeric dtype, channel order, sample rate and time interval.
4. Exact feature names/order if features are the input.
5. Train-equivalent preprocessing code/parameters, fitted scaler/selector artifacts, stale-data and edge behavior, and required neighboring signal context.
6. Label encoding and human-readable class mapping/order.
7. Prediction call and output shape; whether probabilities are supported and calibrated.
8. Runtime dependencies and tested versions, device requirements, expected latency/memory.
9. At least one representative golden input and expected class/output, preferably multiple classes and QC edge cases.
10. Reproducible grouped evaluation: split groups, accuracy/balanced accuracy/F1/confusion matrix and known limitations.
11. Known failure modes, training data/version, and artifact licensing/location.

## Integration acceptance checks

- The exact same raw sample processed offline and by the deployed adapter produces equivalent model input (numerically within an agreed tolerance).
- Channel order, length, labels and feature order are checked, not inferred from dict ordering.
- Scalers, feature selection and any learned preprocessing were fit only on training folds.
- Golden sample predictions match expected class; probability outputs are tested only if offered.
- Missing/corrupt artifact and inference exceptions do not result in a fake label.
- Evaluation split groups by recording/session and, when known, includes subject-held-out evaluation.

## Questions blocking final contract sign-off

1. Does the final estimator consume the full 10-second file, a cue-locked subwindow, a full continuous recording, or the handoff feature table?
2. Can the training pipeline expose a single-trial transform that is numerically equivalent to its block-continuous zero-phase filter, or must the API accept context/streams?
3. Is the owner-confirmed first-2,160 label/file mapping grounded in acquisition records, and what do the 240 extra labels represent?
4. What class ordering, model versioning, probability/calibration behavior, and latency target should the backend expose?
