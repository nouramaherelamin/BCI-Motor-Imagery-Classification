# BCI motor imagery project audit

Audit date: 2026-11-19. This report is based on the tracked repository at `ca09a37` and the saved notebook outputs. The raw `ML CSV/` EEG folder is not present, so raw-data findings below are verified from notebook code, persisted reports, and plots rather than by rereading the original recordings.

## A. Executive summary

The project aims to distinguish imagined left-hand from right-hand movement using four EEG channels (`FZ`, `C3`, `CZ`, `C4`). The repository contains substantial exploration, validation, cleaning, and feature-export work, but **no trained model, model artifact, or inference implementation**. There is no backend before this change.

The latest handoff is `notebooks/02_Cleaning_QC_and_Handoff.ipynb` plus `outputs_clean/`: it treats each numbered 10-second CSV as one labeled trial, records a project-owner confirmation of the first 2,160 label/file pairs, filters continuous recordings, rejects trials, and writes a 1,222-trial cleaned handoff. The notebook reports no decodable signal in its CSP/LDA checks (accuracy 0.497; shifted-label controls 0.481–0.522; permutation mean 0.496). This is a negative diagnostic on the current whole-trial/features setup, not proof that the data contain no MI signal.

The key remaining scientific gap is event semantics: cue times, imagery windows, rest/baseline intervals, subject/session provenance, and device reference/units are not supplied. The raw data is also not committed. A backend can be started independently, but production prediction cannot be completed until the ML developer provides the trained bundle and proves that its inference preprocessing matches the training preprocessing. The API foundation added here validates the repository-shaped raw trial and explicitly returns HTTP 503 when no real predictor is installed.

## B. Repository inventory

The Git worktree was clean on branch `main` at `ca09a37` (`origin/main`); the previous commit is `4b67e8e`. The project has no `AGENTS.md`, dependency manifest, Python application, tests, or CI configuration before the new `backend/` and `docs/` files.

| Location | Purpose and state |
|---|---|
| `01_Data_Collection_and_Validation.ipynb` | 219-cell earlier validation and feature audit; includes saved outputs and older provisional assumptions. |
| `02_Cleaning_QC_and_Handoff.ipynb` | 29-cell cleaning/model handoff; same cell source as the copy under `notebooks/`. |
| `notebooks/01_Data_Collection_and_Validation.ipynb` | One empty cell; not a duplicate of the root validation notebook and not useful as a pipeline source. |
| `notebooks/02_Cleaning_QC_and_Handoff.ipynb` | Executed copy of the root 29-cell cleaning notebook; latest clean handoff source. |
| `notebooks/BCI_MI_Data_Validation_and_EDA.ipynb` | 40-cell diagnostic exploring two incompatible label/epoch mappings; result is explicitly exploratory. |
| `labels.csv` | 2,400 one-column labels (`Left` 1,187; `Right` 1,213), no missing values. |
| `outputs/` | 94 CSVs, 3 JSONs, 37 images and one text report: earlier raw/centered feature audit and mapping diagnostic outputs. Many versions coexist. |
| `outputs_clean/` | 5 CSVs, 2 JSONs, 8 figures: later cleaning and 1,222-trial handoff. The ignored `.npy` arrays are not tracked/present in the inventory. |
| `docs/csv_inventory.csv` | Machine-readable row/column schema, dtypes, null counts, duplicate-row counts and sample records for each tracked CSV. |
| `Ex/README.md` | Short overview that still says label mapping and model readiness are unresolved. |
| `Ex/README (1).md` | Long earlier report (1,160 lines) that describes a 120-feature pre-modeling dataset and calls mapping/fs provisional. |
| `.gitignore` | Ignores `ML CSV/`, NumPy arrays, pickle files, virtual environments, and caches. This excludes raw EEG and model-ready array artifacts from version control. |

No YAML, dependency/environment file, model artifact, backend code, or API documentation existed. `outputs/Final_Pre_Modeling_Report.txt`, `outputs/pre_modeling_manifest.json`, and `outputs/final_pre_modeling_audit.json` all say modeling has not started, but their technical details are older than the clean handoff.

## C. Notebook analysis

### Root `01_Data_Collection_and_Validation.ipynb` (219 cells)

- **Purpose:** initial file validation, timestamp/signal quality exploration, centering, and successive feature audits.
- **Inputs:** Windows paths to 2,160 `ML CSV/cellula_MI_data_<n>.csv` files and `labels.csv`; raw files are absent from this clone.
- **Operations:** structure checks; sampling-rate estimates from timestamps; held-row and outlier inspection; class summaries; mean-centering; time-domain features; Welch spectral and MI-oriented features; correlation/redundancy audits; exported reports.
- **Saved result:** approximately 48 time-domain features, a later 120-feature review table/122-column dataset, 2,160 provisionally labeled files, and numerous quality/redundancy CSVs. No filtering or trial rejection in the initial pre-modeling manifest. It includes a notebook image-extraction cell targeting a Windows `outputs/#Image` path; the extracted figures in this repository are numbered under `outputs/figures/` instead.
- **Dependencies:** NumPy, pandas, matplotlib, seaborn, SciPy, scikit-learn, statsmodels and tqdm across cells.
- **Status/problems:** executed outputs exist, but no saved error output was found. It predates the later cleaning and mapping evidence. Its full report says frequency features are provisional and mapping is unconfirmed. It is not a safe source of serving-time transforms.

### `notebooks/01_Data_Collection_and_Validation.ipynb` (1 cell)

This file is a one-cell empty placeholder, not a second analysis version. The complete earlier notebook is at the repository root.

### Root and `notebooks/02_Cleaning_QC_and_Handoff.ipynb` (29 cells each)

Cell source is identical between these two files; these are copies, with the `notebooks/` copy executed and carrying saved outputs. The notebook:

1. Loads raw 2500×4 arrays and timestamps; pairs each file with the corresponding first 2,160 labels and encodes `Left=0`, `Right=1`.
2. Uses duration (~250 Hz) and a 50 Hz mains peak to support `fs=250`; treats sample index as time because `Time` is packet-arrival timing.
3. Segments continuous recordings by time gaps and filter blocks by relative DC jumps; notebook output reports 240 recordings and 240 filter blocks, with 9 files per recording in its sample table.
4. Finds held rows (461 files), interpolates them, and expands the contaminated region by ±0.2 seconds.
5. Notches 50 Hz (Q=30), applies fourth-order zero-phase 1–40 Hz Butterworth band-pass to a whole filter block, and rejects block-edge, stale/contaminated, flat, nonfinite, and spike trials. It writes both non-CAR and CAR arrays in the notebook; warns CAR on four channels is rank 3 and unsuitable for CSP.
6. Scales each retained trial by one robust scalar; retains inter-channel variance ratios. The exported clean feature table has 30 band/lateralization values plus ID and label.
7. Reports 1,222/2,160 kept (56.6%), 602 Left/620 Right, 240 recording/session IDs, and 10 gain regimes. `outputs_clean/trials_all.csv` has one row per input; `trials_kept.csv` and `features_clean.csv` each have 1,222 rows.
8. Runs whole-trial CSP+LDA and shifted/permuted label diagnostics, then time-resolved checks. The saved outputs show no significant class feature after FDR (0/30) and no decodable signal in the reported CSP/LDA check.

It does **not** train or export a final classifier. It depends on NumPy, pandas, matplotlib, SciPy, and optional tqdm; notebook cells also import scikit-learn / scipy.linalg for the diagnostic. It documents an owner-confirmed label-to-file pairing, but no acquisition/event file is included to independently establish cue timing or the precise trial semantics.

### `notebooks/BCI_MI_Data_Validation_and_EDA.ipynb` (40 cells)

- **Purpose:** consolidated validation and diagnostic of two label/epoch mapping hypotheses.
- **Inputs/transforms:** scans the same raw file layout; assumes `fs=250`; detects groups of 9 files (2,500 samples each); interpolates held rows, linear detrends, applies 0.5–40 Hz band-pass and 50 Hz notch, and optionally common-average references.
- **Hypotheses:** A treats each of 2,160 files as a trial and pairs first 2,160 labels; B divides each 9-file block into ten 9-second segments (2,400 total). A/B both remain diagnostic in this notebook. It rejects by held fraction, amplitude, and flatness; computes Welch log band powers, PSD/ERD plots, grouped LDA probe, and a boundary offset scan.
- **Saved diagnostics:** A: 1,857 kept and balanced accuracy 0.530; B: 2,011 kept and balanced accuracy 0.513. The offset scan is 0.500–0.521. These are close to chance and do not validate the B timing hypothesis.
- **Dependencies:** NumPy, pandas, matplotlib, SciPy, scikit-learn, tqdm/IPython.
- **Status:** executed, no saved error. Its text explicitly says cue/trial timing is unknown. Its filter order and settings differ from the later clean handoff, so its arrays/features are not interchangeable with `outputs_clean/`.

### Notebook findings, warnings, and cell execution

The root validation notebook and clean notebook have completed execution counts and saved outputs. The 40-cell diagnostic also has saved outputs. No notebook saved error output was found. Outputs include old Windows-specific local paths and prior library versions; a fresh run is not reproducible here because raw files are absent. The root 01 notebook has a lengthy evolution of outputs and intermediate feature versions, so its latest feature counts should not be interpreted as a single, stable model contract.

## D. Data and scientific pipeline

| Stage | Repository evidence / current state | Owner / issue |
|---|---|---|
| Acquisition | Cellula motor-imagery EEG CSVs are referenced, 2,160 files, not committed. Device/source metadata is absent. | Team member 1 handoff; source provenance needed. |
| File structure | Each reported file contains 2,500 rows, `Time,FZ,C3,CZ,C4`; notebook sample values around 3e5. | Team member 1 completed structural checks (from saved outputs). |
| Labels | `labels.csv`: 2,400 strings, 1,187 Left and 1,213 Right. Later clean notebook asserts first label ↔ first file and says project owner confirmed; 240 labels remain unused. | Mapping claim is notebook text; get provenance/source record into docs. |
| Sampling | Later handoff supports 250 Hz using duration and 50 Hz peak. Per-row `Time` intervals are severely irregular (30.12% <0.1 ms; 8.54% >10 ms in earlier report) and treated as packet arrival; sample index used as time. | 250 Hz is a data-based inference, not device configuration evidence. |
| Recording structure | Earlier notebook finds 240×9-file blocks. Clean notebook finds 240 recordings and 240 filter blocks; uses timestamp gaps and relative DC-step heuristic. | Recordings are used as CV groups; user-facing subject/session identity is unknown. |
| Quality control | Held/stale rows repeated across four channels; clean notebook linearly interpolates, counts ±0.2 s contamination, filters continuous blocks, rejects block edges and QC failures. `trials_kept.csv`: 1,222 rows. | Thresholds are configurable heuristic choices, not clinically validated. |
| Trial timing | File length is 10s at 250Hz; clean notebook treats each file as one label-aligned trial. Cue, imagery, rest and baseline times are absent. | Critical unresolved scientific metadata. No ERD/ERS baseline is established. |
| Preprocessing | Latest clean path: interpolate stale rows; subtract block median; notch 50Hz Q30; 4th-order zero-phase Butterworth 1–40Hz across blocks; reject block edges/QC; robust single scalar per kept trial. | Training transforms must be delivered with final fitted model. |
| Features | `features_clean.csv` has 32 columns: `trial_id`, 28 channel-band features (delta 1–4, theta 4–8, mu 8–12, beta_low 13–20, beta_high 20–30, beta 13–30, gamma 30–40, for 4 channels), `LI_mu`, `LI_beta`, and `y`. These are later handoff diagnostics, not evidence that the ML model will use them. | ML developer chooses final representation and documents order. |
| Model training/validation | No trained artifact, fitted scaler, class-probability calibration, or model code in repo. Clean notebook reports CSP+LDA decodability check only; no reported selected model. | Team member 3 owns. |
| Inference | No serving transform. Whole-block zero-phase filtering in training cannot be reproduced from an isolated incoming file unless the deployed contract supplies context or uses a separately validated streaming transform. | Shared design required before prediction is enabled. |

### Exact classification evidence and unresolved points

- **Task/classes:** binary imagined left vs right hand; data strings are `Left` and `Right`; later notebook uses `Left=0`, `Right=1`.
- **Recorded input:** 2,500 samples × 4 channels, order `FZ,C3,CZ,C4`, per referenced CSV; 250 Hz is the later handoff setting. Amplitude appears raw ADC counts; scale conversion is unknown.
- **Feature vector/model input:** unresolved. The 120-feature older file and 30-feature clean diagnostic table are different artifacts; neither is declared the final model input.
- **Normalization:** older time features use per-trial mean centering. Clean handoff uses block median subtraction and one robust scalar per trial. No fitted across-trial scaler is specified in current final handoff evidence.
- **Split:** clean notebook recommends grouped CV by recording `session_id`; saved manifest says never shuffle across recordings. Subject/session membership beyond time-derived recording IDs is unknown. No final train/validation/test split or deployed metric is documented.
- **Class balance:** label source is 1,187/1,213; first-2,160 file-paired set is 1,082/1,078; cleaned set is 602/620. Mild imbalance, not evidence of a serving issue.
- **Leakage risk:** random trial split could place adjacent pieces from the same continuous recording in train and test. Fit any learned scaler/feature selection only on training folds. Group by recording; if true subject IDs become available, evaluate subject-wise holdout as well.
- **Current diagnostic:** the clean notebook reports 0/30 features significant after FDR and CSP/LDA accuracy 0.497; this does not support claiming a working classifier.

## E. Figures and results

All 45 saved PNGs under `outputs/` and `outputs_clean/` were visually inspected using contact sheets; notebook figures repeat or complement these outputs.

| Figure set | What it shows | Legitimate conclusion |
|---|---|---|
| `outputs/figures/01_timestamps_and_rate.png`, `outputs_clean/figures/02_fs_evidence.png`, `outputs/timestamp_analysis_513.png`, figures 001–003 | Duration-rate distributions, irregular intervals and a very slow timestamp trace for file 513. | Time field does not represent uniform sample times; later 250 Hz is supported by duration/mains evidence. Do not resample rows by packet-arrival timestamp as if it were acquisition time. |
| `outputs/figures/02_recording_blocks.png`, `outputs_clean/figures/02b_continuity.png` | Raw levels and inter-file time/DC changes across file order. | Recordings have discontinuities; filter continuous chunks with boundaries respected. |
| `outputs/figures/03_held_samples.png`, `outputs_clean/figures/03_stale_samples.png`, figures 011–014, 021–022 | Held-row percentages and long repeated plateaus, including an example signal segment. | Data contain substantial repeated-value stretches; interpolation does not recover the original signal and QC rejection is material. |
| `outputs/figures/04_signal_std.png`, figures 008–010, 015 | Channel spread/outlier runs and class signal variability. | Strong file/block-specific scale and outlier regimes; broad boxplot overlap, no evidence of class separation. |
| Figures 004–007 and `outputs_clean/figures/05_examples_before_after.png` | Raw oscillatory traces, centering/filtering examples, per-channel traces. | DC drift/offset and filter edge effects are visible; the later block-continuous filtering approach is motivated. |
| `outputs_clean/figures/05_psd_before_after.png`, `outputs_clean/figures/07_median_psd_by_class.png`, `outputs/figures/05_psd_by_mapping.png`, figures 017–020 and 023–026 | Strong narrow 50 Hz peak in older spectra; cleaned PSDs and left/right median curves. | 50 Hz contamination is strong in the older processing; class medians largely overlap. PSD comparison is not event-locked evidence of MI response. |
| `outputs/figures/06_tf_erd_A.png`, `06_tf_erd_B.png`, `07_offset_scan.png` | Mapping A/B time-frequency comparisons and offset-scan diagnostic. | No clear class pattern or meaningful offset improves decoding; mapping B remains a hypothesis. |
| `outputs_clean/figures/06_sessions.png`, `08b_time_resolved_auc.png` | Recording/gain regimes and time-resolved class AUC curves. | Recordings/gain vary; time-varying AUC is weak/variable and not a validated model result. |
| `outputs/feature_correlation_heatmap.png`, `combined_feature_correlation_heatmap.png`, figure 016 | Correlation matrices for old engineered feature sets. | Several highly correlated clusters; these are redundancy clues, not evidence of predictive value. |

No plot establishes cue-locked ERD/ERS because cue timing and baseline are unavailable. The older report's very high 45–55 Hz relative power and overlapping class spectra are compatible with line noise, but the exact interpretation remains conditional on the sampling and preprocessing assumptions.

## F. Data quality and handoff assessment

**Assessment: substantial exploration is complete, but the project is not scientifically ready for a user-facing classifier.** Team member 1 has delivered extensive structural and signal-quality analysis. The clean notebook is a useful later handoff, not a final model dataset contract.

| Severity | Issue and evidence | Suggested owner |
|---|---|---|
| Critical | Cue/imagery/baseline timing and true subject/session metadata are absent. A 10-second file label does not identify the active MI interval. | Team member 1 / dataset owner, then team member 3 to epoch/evaluate. |
| Critical | Raw EEG files and `.npy` clean arrays are not present in the tracked repo; `.gitignore` excludes `ML CSV/` and `*.npy`. Fresh reproducibility/integration is blocked without a controlled data handoff. | Team member 1; agree secure storage/reference with team. |
| High | Label mapping is called provisional in older README, report and manifest but owner-confirmed in later clean notebook. 240 labels remain unused. | Team member 1 / data owner: record confirmation and rationale. |
| High | Training cleans a whole continuous block with zero-phase filtering; isolated-trial serving can produce different filter-edge behavior. | Team members 2 and 3 jointly; model bundle must own/declare transform and required context. |
| High | Current CSP/LDA diagnostic is near chance (0.497) and class analyses show no reliable signal. No classifier has been trained. | Team member 3, after event/timing resolution. |
| Medium | Several generations of features/reports disagree: old 48 time features, 120 numeric features, mapping A/B outputs, and clean 28 spectral features + label/id. | Team member 3 selects one training artifact; team member 2 serves only the final bundle's declared input. |
| Medium | Cleaning thresholds and gain/session grouping are heuristic; 938/2,160 trials rejected and 480 block edges are rejected. QC rationale needs sensitivity analysis before final model claims. | Team member 1 + 3. |
| Low | README and report contain stale Windows paths, old conclusions, and no dependency setup. | Team member 2 (docs); member 1 confirms scientific edits. |

The clean handoff reports retained class balance Left 602 / Right 620. `trials_all.csv` has missing `session_id` and `regime_id` only for rejected records where appropriate; retained data has all tracking values. `stale_report.csv` and `trials_kept.csv` carry IDs and quality summaries. The manifest reports rejection reasons that overlap; they should not be summed as mutually exclusive counts.

## G. Team responsibilities

- **Member 1 (data):** delivered initial validation, notebook-based timestamp/block/quality work and a clean handoff. Confirm the owner-sourced label mapping, event timing, subject/session metadata, hardware/sample-rate/reference details, raw data location, and whether the clean-array files can be shared reproducibly.
- **Member 2 (backend):** can own API scaffolding, validation, model adapter boundary, error behavior, API tests, developer setup, and the integration contract. This turn adds that foundation; final inference stays disabled pending model/preprocessing evidence.
- **Member 3 (ML):** owns final trial/epoch interpretation with member 1, feature/model selection, leakage-safe evaluation, training pipeline and artifact packaging. Must provide an inference-ready bundle and representative expected outputs.
- **Shared:** approve one preprocessing implementation and input contract; compare training and API predictions on the same stored sample before enabling production predictions.

## H. Backend vs. ML dependency analysis

Backend work should start now. API scaffolding, request validation, health, metadata, error behavior, test harness, and deployment shape do not require a trained model. The actual prediction transform and response confidence semantics do.

| Component | Can backend start now? | ML dependency | Required coordination |
|---|---|---|---|
| Project structure | Yes | None | Keep backend isolated; current addition does. |
| Configuration | Yes, basic app settings | Artifact path and runtime settings later | Agree model bundle location and environment variables. |
| Database | No documented need | None | Add only if persistence is later required. |
| Input validation | Yes, provisional raw-trial schema | Confirm final input modality/shape | Current schema matches reported raw file shape only. |
| Health endpoint | Yes | Whether readiness requires model | Current health reports loaded state separately. |
| Prediction endpoint | Shell can start now | Final input and model output | It returns 503 until real predictor exists. |
| Model loading | Adapter boundary now | Artifact and framework | Member 3 supplies load code / bundle. |
| Feature preprocessing | Interface only | Exact training transforms and context | Must be shared with or included in model bundle. |
| Prediction service | Adapter boundary now | Predictor implementation | No estimator-specific code in routes. |
| Output serialization | Yes, minimal class/version | Class order, scores/calibration | Only emit probabilities if model explicitly supports them. |
| Tests | Yes, deterministic stub only | Golden real-model samples later | Current stub is test-only, never default. |
| Deployment | Local/dev instructions now | Runtime/artifact/latency limits | Finalize after framework and bundle choice. |

## I. API design and current implementation

All paths are versioned under `/api/v1`.

| Method/path | Purpose | Request | Response / errors |
|---|---|---|---|
| `GET /api/v1/health` | Process health plus whether a predictor is installed. | None | `{"status":"ok","model_loaded":false}`; does not claim the model is ready if absent. |
| `GET /api/v1/model/info` | Safe model metadata. | None | `{"loaded":false,"metadata":null}` until adapter exists. |
| `POST /api/v1/predict` | Validate one 2,500×4 trial and delegate to the installed adapter. | Sampling rate 250 Hz; channel order `FZ,C3,CZ,C4`; 2,500 rows of four finite numbers. | Predicted label, model version, optional model-provided probabilities. 422 invalid input; 503 no model; 500 inference failure. |

Example request envelope (the `samples` array must contain exactly 2,500 rows):

```json
{
  "sampling_rate_hz": 250.0,
  "channels": ["FZ", "C3", "CZ", "C4"],
  "samples": [[324761.375, 321002.625, 335287.28125, 321417.90625]]
}
```

Example response from a **test adapter only**:

```json
{"predicted_label":"Left","model_version":"test-fixture-1","probabilities":null}
```

The request is a backend starting point, not an approved end-to-end inference contract. Serving a single raw file cannot necessarily reproduce whole-block zero-phase preprocessing. Endpoint prediction remains unavailable in the default application until a real bundle is wired in; no fallback/fake label is returned.

## J. ML/backend integration contract

See [ML/backend integration contract](ml_backend_contract.md). The contract deliberately leaves model input (raw trial, preprocessed epoch, or features), final feature list/order, filtering context, model framework and score semantics open until member 3 confirms them. The only safe current API facts are the repository file shape/channel names and the later handoff's 250 Hz assumption.

## K. Implementation and verification status

Added `backend/` with a FastAPI app, Pydantic raw-trial schema, framework-independent predictor protocol, model-info/health/predict endpoints, and tests using an injected deterministic stub. Added this audit, an integration contract and backend setup guide.

Initial backend verification at the time this audit was drafted: `rtk python -m pytest -vv` completed with 7 passed. The backend was expanded in a follow-up; see [backend integration status](backend_integration_status.md) for the current test count and live-server check. No notebook was run because the raw data folder is missing and notebook execution would regenerate/overwrite outputs.

```bash
cd backend
python -m pip install -e '.[test]'
pytest
uvicorn bci_backend.main:app --reload
```

Real artifact loading, preprocessing compatibility, prediction quality/performance, and final deployment remain unimplemented because model and scientific inputs are missing.

## L. Blockers and risks

1. **Event timing and dataset metadata (critical):** obtain cue/task intervals, subject/session IDs, sampling/device details, and reference/unit information.
2. **Model artifact and inference pipeline (critical):** no model is present; raw trial → training-equivalent preprocessing is not specified.
3. **Scientific readiness (high):** current diagnostics are at chance; do not present predictions as validated.
4. **Data accessibility/reproducibility (high):** raw and clean arrays are ignored/not committed; establish a team-accessible source and version/hash manifest.
5. **Documentation drift (high):** old docs say provisional mapping while newer notebook says owner-confirmed. Preserve the owner confirmation with provenance and update docs after member 1 verifies.

## M. Ordered next actions

### Backend developer (member 2)

1. Agree on whether the frontend supplies one raw 10-second CSV-shaped trial or a continuous stream. Completion: signed input schema and context/latency needs from member 3.
2. Add model bundle loading/configuration without a fake fallback. Completion: missing/bad artifact produces a clear non-ready state; real artifact metadata is exposed safely.
3. Integrate the model-owned transform/adapter. Completion: golden test inputs match offline model predictions and feature order exactly.
4. Add deployment/runtime settings only after the model framework, resource size, and inference latency are known.

### ML developer (member 3)

1. Resolve epoch meaning and labels with member 1 before model selection. Completion: documented trial/cue window, class mapping, subject/session grouping.
2. Select a single train-time pipeline and grouped evaluation. Completion: reproducible held-out metrics, confusion matrix and limitations.
3. Package an inference bundle with transform parameters, class mapping, model version, dependency versions, input shape/order, and one golden input/output pair. Completion: adapter can load and make a deterministic prediction.
4. State whether probability-like outputs are supported and calibrated; otherwise return only the class.

### Data owner/member 1

1. Confirm label/file mapping with source evidence and account for 240 unused labels.
2. Supply cue timing, subjects/sessions, acquisition settings (sample rate, reference, units), and controlled raw-data location.
3. Explain the 240 recording/filter-block groupings and QC/rejection thresholds; verify clean handoff arrays can be regenerated from the provided data.

### Shared integration and delivery

1. Reconcile README/report claims with clean notebook and source evidence.
2. Run offline-vs-API golden sample parity test; test malformed, stale, and model-unavailable paths.
3. Review grouped held-out performance and latency; enable prediction only after scientific/model review.
