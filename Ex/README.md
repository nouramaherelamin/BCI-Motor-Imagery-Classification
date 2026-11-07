# BCI Motor Imagery Classification (Left vs Right)

Classifying **imagined left vs right hand movement** from 4-channel EEG (FZ, C3, CZ, C4).  
This repository covers everything **before modeling**: data audit, signal-quality analysis, label-structure analysis, preprocessing, exploratory analysis, and a diagnostic of how EEG files map to labels.

> **Status: pre-modeling.** No classifier has been trained yet, because the link between the EEG files and the labels still needs to be confirmed (see [Open questions](#8-open-questions-for-the-data-owner)).

---

## 1. Project goal

Decide whether a person is imagining **left** or **right** hand movement using EEG only (no actual movement). Motor imagery suppresses mu (8-12 Hz) and beta (13-30 Hz) power over the **contralateral** sensorimotor cortex (event-related desynchronization, ERD), so **C3** and **C4** are the key channels.

Planned pipeline: `load -> validate -> preprocess -> EDA -> features (CSP) -> models (CSP+SVM, EEGNet) -> evaluation -> Streamlit demo`

## 2. Data

| Item | Value |
|---|---|
| EEG files | 2160 CSV files, `cellula_MI_data_<N>.csv` |
| Columns | `Time, FZ, C3, CZ, C4` |
| Samples per file | 2500 (10 s at 250 Hz) |
| Labels | `labels.csv`, **2400** labels (Left 1187, Right 1213) |
| Amplitude | values around 3e5; look like **raw ADC counts**, not microvolts (scale factor unknown) |

The raw data is **not stored in this repository** (size). Download link: _add your Google Drive / source link here_.  
Expected layout after download:

```
ML CSV/cellula_MI_data_1.csv ... cellula_MI_data_2160.csv
labels.csv
```

## 3. Key findings so far

### Established from the data
1. **Sampling rate is 250 Hz.** Treating rows as evenly spaced at 250 Hz puts the 50 Hz mains peak exactly at 50 Hz. The `Time` column does **not** follow sampling: it records packet *arrival*, so many consecutive rows share almost the same timestamp followed by jumps (up to ~0.4 s). Row order is therefore used as the time base.
2. **Files come in recording blocks.** Consecutive files are cut from continuous recordings: the start time of file *k+1* equals the end time of file *k*. This breaks every **9 files**, where `Time` jumps and the signal's DC level changes abruptly.  
   **2160 files = 240 blocks x 9 files** (90 s per block).
3. **Labels follow the block structure, not the file structure.** **2400 labels = 240 blocks x 10 labels.** Labels and files therefore cannot be matched one-to-one by index.
4. **Held samples.** Some rows repeat the previous row on all channels while the timestamp barely moves (the last value is held when no new packet arrives). A subset of files contain long runs of this, up to ~77 % of a file.
5. **Quality problems are concentrated, not random.** Flagged outliers cluster in specific blocks, mostly around files 451-700 (raw level roughly 430,000-520,000, far above the typical 250,000-350,000).
6. **No integrity problems in the usual sense:** no missing, duplicate or non-numeric values; consistent column layout and file length.

### Hypothesis (needs confirmation from the data owner)
Each 90 s block holds **10 trials (~9 s each)** that were later cut into fixed 2500-sample files, so **a file is probably not a trial**. This would explain 2160 files vs 2400 labels.

### Left vs Right analysis
- **Mapping A** (file = trial, first 2160 labels) and **Mapping B** (10 trials x 2250 samples per block, labels in order) were both tested with simple mu/beta log band-power features and a block-grouped cross-validated LDA probe. Both are **near chance** (balanced accuracy about 0.50-0.52; lateralization AUC about 0.47-0.53).
- Median PSD curves for Left and Right overlap on C3 and C4; no contralateral ERD is visible in the time-frequency maps.
- Shifting the trial boundaries inside a block (offset scan) did not reveal a better alignment.

**Interpretation:** the block structure is solid, but the trial definition inside a block is still unknown, so neither mapping can be trusted yet. This is a diagnostic result, **not** a statement that the data contain no motor-imagery signal.

## 4. Preprocessing

| Step | Setting |
|---|---|
| Time base | row order at 250 Hz (not the `Time` column) |
| Held samples | replaced by linear interpolation |
| Detrend | linear (removes the large DC offset and slow drift) |
| Band-pass | 0.5-40 Hz, Butterworth order 4, zero-phase |
| Notch | 50 Hz |
| Re-reference | Common Average Reference (4 channels) |
| Filtering scope | **whole continuous block (22,500 samples) before cutting**, which removes the edge transients seen when each 10 s file is filtered alone |
| Rejection | segments with >20 % held samples, peak-to-peak above 8x the dataset median, or a flat channel (about 14-16 % of segments; thresholds are configurable) |

## 5. Repository structure

```
.
├── notebooks/
│   ├── 01_Data_Collection_and_Validation.ipynb    # file audit, timestamps, held samples, outliers
│   ├── 02_Cleaning_QC_and_Handoff.ipynb           # cleaning, quality control, hand-off datasets
│   ├── BCI_MI_Processing_EDA.ipynb                # first processing + EDA version (per-file)
│   └── BCI_MI_Data_Validation_and_EDA.ipynb       # main notebook: full validation, blocks, preprocessing, EDA, mapping diagnostic
├── outputs/                  # reports (CSV/JSON) and figures
├── outputs_clean/            # cleaned-data reports and figures
├── labels.csv
├── requirements.txt
└── README.md
```

The raw `ML CSV/` folder and all `*.npy` / `*.npz` files are excluded through `.gitignore`.

### Notebook guide

| Notebook | Purpose |
|---|---|
| `01_Data_Collection_and_Validation` | Load all files, validation report, timestamp analysis, held-sample analysis, outlier reports |
| `02_Cleaning_QC_and_Handoff` | Cleaning, quality control and hand-off of data for modeling |
| `BCI_MI_Processing_EDA` | First pipeline version (per-file, written before the block structure was discovered) |
| **`BCI_MI_Data_Validation_and_EDA`** | **Main notebook.** Single-pass scan with caching, 50 Hz sampling-rate check, recording-block detection, held-sample and outlier analysis, label arithmetic, block-continuous preprocessing, mapping hypotheses A and B, PSD / band power / lateralization / time-frequency ERD, grouped-CV probe, boundary-offset scan, findings and open questions |

## 6. How to run

```bash
git clone https://github.com/nouramaherelamin/BCI-Motor-Imagery-Classification.git
cd BCI-Motor-Imagery-Classification
pip install -r requirements.txt
```

1. Download the data and place it as shown in [Data](#2-data).
2. Open `notebooks/BCI_MI_Data_Validation_and_EDA.ipynb`.
3. In the first code cell set `PROJECT_DIR` (or the environment variables `BCI_PROJECT_DIR`, `BCI_DATA_DIR`, `BCI_LABELS`, `BCI_OUT`).
4. **Run All.** The first run reads every CSV once and caches `file_scan.csv` and `X_raw.npy` in `outputs/`; later runs start immediately.

All parameters (filters, thresholds, block size, rejection rules) are in one `CFG` dictionary and exported to `outputs/config.json`.

## 7. Outputs of the main notebook

`data_validation_report.csv`, `file_scan.csv`, `recording_blocks.csv`, `outlier_files_report.csv`, `provisional_label_mapping_A.csv`, `segments_A.csv`, `segments_B.csv`, `band_power_A.csv`, `band_power_B.csv`, `config.json`, and figures `01_timestamps_and_rate`, `02_recording_blocks`, `03_held_samples`, `04_signal_std`, `05_psd_by_mapping`, `06_tf_erd_A/B`, `07_offset_scan`.

## 8. Open questions for the data owner

1. **How is a trial defined inside a 90 s block?** (start of each trial, cue time, imagery window, rest period.) This decides the epoch window and the file-to-label mapping.
2. Is the label order inside a block the chronological trial order?
3. How many **subjects/sessions** are there, and which blocks belong to which? (needed for leakage-free train/test splits)
4. What is the amplitude scale (ADC counts to microvolts), and was any filtering or referencing applied on the device?

## 9. Current status and next steps

- [x] Data audit and validation
- [x] Sampling-rate verification and timestamp analysis
- [x] Recording-block structure discovered
- [x] Held-sample and outlier analysis
- [x] Block-continuous preprocessing
- [x] EDA under two mapping hypotheses
- [ ] **Trial timing and label mapping confirmed with the data owner** (current blocker)
- [ ] Final epochs and splits grouped by block/session
- [ ] Baseline: **CSP + SVM** with block-grouped cross-validation
- [ ] **EEGNet**, compared on the same folds
- [ ] Metrics: accuracy, balanced accuracy, F1, AUC, confusion matrix
- [ ] Streamlit demo wrapping the best model, with identical preprocessing

## 10. Project log

1. Read the project brief (processing, EDA, CSP+SVM, EEGNet, Streamlit).
2. First inspection: 5 columns, 2500 rows per file, huge DC offset, irregular `Time`, **2400 labels vs 2160 files**.
3. First pipeline (per-file filtering): about 9 % of trials rejected; no Left/Right difference in PSD or ERD plots; DC-level changes between groups of files noticed.
4. Validation notebook confirmed the irregular timestamps, the held samples and the label-count mismatch.
5. Discovered that `Time` is continuous across files inside groups of 9, giving **240 x 9 = 2160** files and **240 x 10 = 2400** labels: the block structure.
6. Tested "10 trials per block" and a boundary-offset scan: near-chance results; trial timing still unknown.
7. Built the consolidated notebook `BCI_MI_Data_Validation_and_EDA` with caching, two mapping hypotheses and a diagnostic.
8. Published the project to GitHub (large files excluded).

## Author

**Noura Maher Elamin**: Computer & Information Systems student, Egyptian Chinese University (Data track).  
[LinkedIn](https://www.linkedin.com/in/nouramaherelamin/) | [GitHub](https://github.com/nouramaherelamin)

## Data note

Course/research project. The EEG data belongs to its original source and is not redistributed here.
