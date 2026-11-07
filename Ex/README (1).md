# BCI Motor Imagery EEG Project

## Comprehensive Data Analysis & Pre-Modeling Technical Report

**Project:** BCI Motor Imagery EEG Analysis\
**Project Directory:** `C:\Users\noura\Z-BCI-Motor`\
**Dataset:** Cellula Motor Imagery EEG Dataset\
**Status:** Pre-Modeling Phase Completed\
**Modeling:** Not Started

------------------------------------------------------------------------

# 1. Executive Summary

This project focuses on preparing and analyzing EEG signals for a
Brain-Computer Interface (BCI) Motor Imagery classification task.

The objective is to investigate EEG recordings associated with two motor
imagery classes:

-   Left
-   Right

The work covered data collection and validation, signal quality
assessment, exploratory data analysis, signal centering, time-domain
feature extraction, frequency-domain feature extraction, motor imagery
feature engineering, statistical comparisons, correlation analysis,
redundancy investigation, and final pre-modeling validation.

The current feature dataset contains:

-   2,160 EEG trials.
-   4 EEG channels: FZ, C3, CZ, and C4.
-   2,500 samples per trial.
-   120 engineered features.
-   2 target classes: Left and Right.

The structural validation checks passed. However, several scientific
limitations remain, particularly the unconfirmed sampling rate,
irregular timestamps, signal quality issues, and provisional
label-to-file mapping.

No machine learning models have been trained.

# 2. Project Dataset and Data Collection

## 2.1 Dataset Structure

The EEG recordings are stored in:

`C:\Users\noura\Z-BCI-Motor\ML CSV`

The label file is:

`C:\Users\noura\Z-BCI-Motor\labels.csv`

The dataset contains 2,160 CSV files:

`cellula_MI_data_1.csv` through `cellula_MI_data_2160.csv`

Each recording contains:

  Property              Value
  --------------------- ----------------
  Number of EEG files   2,160
  Samples per file      2,500
  Number of columns     5
  Time column           Time
  EEG channels          FZ, C3, CZ, C4

The total number of EEG samples is 5,400,000, representing 21,600,000
channel measurements.

## 2.2 Label Dataset

The original labels file contains 2,400 records.

  Class     Count
  ------- -------
  Left      1,187
  Right     1,213
  Total     2,400

The current analysis uses the first 2,160 labels to create a provisional
mapping between the EEG files and their corresponding labels.

The provisional paired distribution is:

  Class     Trials
  ------- --------
  Left       1,082
  Right      1,078
  Total      2,160

There are 240 additional labels that have not been mapped to EEG files.

**Important limitation:** The first-2,160-label mapping is provisional
and has not been independently confirmed against the original
acquisition metadata or experiment sequence.

# 3. Data Validation

The first stage examined the structural integrity and consistency of all
EEG files.

## 3.1 File Validation

All 2,160 EEG files were read successfully.

The following checks were performed:

-   File readability.
-   Expected number of rows.
-   Expected column names.
-   Numeric data validation.
-   Missing-value detection.
-   Duplicate-row detection.
-   Unexpected-column detection.
-   NaN and infinite-value detection.

## 3.2 Validation Results

  Validation Check                    Result
  -------------------------- ---------------
  Successfully read files      2,160 / 2,160
  Expected rows per file               2,500
  Expected columns                         5
  Missing values                           0
  Duplicate complete rows                  0
  Non-numeric EEG values                   0
  Unexpected columns                       0
  NaN values in X_raw                      0
  Infinite values in X_raw                 0

The raw EEG array was constructed with:

`X_raw.shape = (2160, 2500, 4)`

Data type: `float32`.

**Conclusion:** The dataset passed the structural validation checks.
This does not establish that every EEG recording has acceptable signal
quality.

# 4. Timestamp and Sampling Rate Investigation

Timestamp analysis was performed to investigate the actual recording
frequency and temporal consistency.

## 4.1 Estimated Sampling Rate

The duration-based estimated sampling rate produced the following
results:

  Metric                         Value
  -------------------- ---------------
  Mean                   250.309488 Hz
  Standard deviation       3.017252 Hz
  Minimum                217.689588 Hz
  Median                 250.273105 Hz
  Maximum                286.127089 Hz

The minimum was observed in file 512, and the maximum in file 513.

## 4.2 Timestamp Interval Analysis

The timestamps were strictly increasing, but the intervals between
consecutive samples were highly irregular.

  Interval Range     Percentage
  ---------------- ------------
  Below 0.1 ms           30.12%
  0.1--1 ms              0.127%
  1--4 ms                43.68%
  4--10 ms               17.53%
  Above 10 ms             8.54%

Additional findings:

-   Mean interval: approximately 0.003996 seconds.
-   Median interval across all observations: approximately 0.002479
    seconds.
-   Standard deviation: approximately 0.007106 seconds.
-   No non-positive timestamp intervals were detected.

The duration-based average sampling rate does not resolve the
irregularity of individual timestamp intervals.

**Decision:** The sampling rate was classified as:

`Unconfirmed - Irregular Timestamps`

A provisional value of 250 Hz was used for exploratory frequency
calculations, but frequency-domain findings must remain provisional.

No definitive resampling or frequency-dependent preprocessing was
performed.

# 5. Signal Quality Assessment

Signal quality was investigated before feature extraction to identify
suspicious recordings and potential acquisition or export issues.

## 5.1 Initial Quality Checks

The analysis examined:

-   Channel standard deviation.
-   Signal range.
-   Interquartile range.
-   Outlier flags.
-   Repeated adjacent samples.
-   Long constant-value plateaus.
-   Shared repetition across multiple channels.

The initial standard-deviation and range investigation identified 1,114
flagged channel records across 260 trials.

Trial 550 was particularly unusual:

-   FZ standard deviation: approximately 11,490.
-   CZ standard deviation: approximately 19,635.
-   CZ signal range: approximately 81,884.

These values were flagged for further investigation rather than
automatically removed.

## 5.2 Repeated Samples Investigation

Repeated EEG samples were identified by comparing consecutive EEG
observations.

  Finding                                       Count
  ------------------------------------------- -------
  Trials with repeated adjacent EEG samples       461
  Trials with longest constant run ≥ 10           423
  Trials with longest constant run ≥ 50           133
  Trials with longest constant run ≥ 100           31
  Trials with repeated percentage ≥ 50%            22

The maximum repeated-sample percentage was approximately 76.59%.

The maximum longest constant run was 307 samples.

## 5.3 Important Investigated Trials

### Trial 610 --- File 611

-   1,329 repeated adjacent EEG rows.
-   Longest repeated run: 174 samples.

### Trial 605 --- File 606

-   1,914 shared repeated sample pairs.
-   Repeated percentage: approximately 76.59%.
-   Longest constant run: 257 samples.
-   Only approximately 556--584 unique values were observed in the
    investigated recording.
-   Timestamp span in the investigated repeated region was approximately
    0.0001059 seconds.

### Trial 301 --- File 302

-   1,908 shared repeated sample pairs.
-   Longest constant run: 87 samples.
-   Timestamp duration in the investigated region was approximately
    0.0000779 seconds.

### Trial 246

-   499 shared repeated sample pairs.
-   Longest shared run: 306 samples.
-   Repeated percentage: approximately 19.97%.

These cases suggest substantial signal continuity problems in some
recordings. The available information does not establish whether the
cause is acquisition, device behavior, or data export.

## 5.4 Final Signal Quality Categories

  Quality Category               Trials   Percentage
  ---------------------------- -------- ------------
  No Detected Issue               1,549       71.71%
  Repeated Samples                  286       13.24%
  Outlier Only                      150        6.94%
  Long Plateau                      117        5.42%
  Outlier + Repeated Samples         42        1.94%
  Outlier + Long Plateau             16        0.74%

## 5.5 Final Review Priorities

  Review Category                  Trials
  ------------------------------ --------
  Critical - Severe Repetition         22
  High - Long Shared Plateau           12
  Review Required                     577
  No Detected Issue                 1,549

These categories are review priorities, not final decisions to exclude
trials.

No recordings were deleted or modified.

# 6. Signal Centering and Preprocessing

A per-trial, per-channel mean-centering operation was performed.

The operation was:

`X_centered = X_raw - X_raw.mean(axis=1, keepdims=True)`

The resulting array:

`X_centered.shape = (2160, 2500, 4)`

Data type: `float32`.

The maximum absolute residual channel mean after centering was
approximately 5.2819624, attributable to floating-point representation.

## 6.1 Preprocessing Decisions

  Operation                    Status
  ---------------------------- ---------------
  Mean centering               Completed
  Bandpass filtering           Not performed
  Notch filtering              Not performed
  Resampling                   Not performed
  Rereferencing                Not performed
  Epoch cropping               Not performed
  Baseline correction          Not performed
  Normalization                Not performed
  Automatic artifact removal   Not performed

These decisions were made because the true sampling rate, reference
scheme, cue timing, and baseline interval were not confirmed.

Mean centering is not equivalent to baseline correction.

# 7. Exploratory Data Analysis (EDA)

EDA was performed to understand the signal characteristics, channel
behavior, class distribution, and general recording patterns.

## 7.1 Raw Signal Exploration

Raw EEG signals showed:

-   Large DC offsets.
-   Oscillatory variations.
-   Abrupt signal changes.
-   Differences in signal amplitude across channels and trials.

## 7.2 Centered Signal Exploration

After mean centering, the DC offset was reduced while the signal
fluctuations remained.

Approximately 59--60 peaks were observed in the first 300 samples in the
investigated examples.

These peak counts were not converted into physical frequencies because
the sampling rate is unconfirmed.

## 7.3 Cross-Channel Correlation

Average cross-channel correlations were calculated separately for Left
and Right trials.

  Channel Pair      Left   Right
  -------------- ------- -------
  FZ--C3           0.623   0.596
  FZ--CZ           0.766   0.749
  FZ--C4           0.680   0.668
  C3--CZ           0.798   0.786
  C3--C4           0.732   0.721
  CZ--C4           0.903   0.901

The strongest average relationship was observed between CZ and C4.

The weakest was observed between FZ and C3.

The correlation patterns were broadly similar across the two classes.

# 8. Time-Domain Feature Engineering

Time-domain features were extracted independently for each of the four
EEG channels.

Twelve features were calculated per channel.

## 8.1 Extracted Features

1.  Mean
2.  Standard Deviation
3.  Root Mean Square (RMS)
4.  Minimum
5.  Maximum
6.  Peak-to-Peak (PTP)
7.  Median
8.  Interquartile Range (IQR)
9.  Skewness
10. Kurtosis
11. Zero Crossing Rate
12. Line Length

The resulting dataset contained:

`48 Time-Domain Features`

The feature dataset shape was:

`(2160, 50)`

The two additional columns were trial_index and label.

## 8.2 Feature Validation

-   Missing feature values: 0.
-   Infinite feature values: 0.
-   Constant features: 0.

## 8.3 Outlier Investigation

The following channel-level outlier counts were identified:

  Channel             Left        Right
  --------- -------------- ------------
  FZ            80 (7.39%)   73 (6.77%)
  C3            45 (4.16%)   47 (4.36%)
  CZ          109 (10.07%)   97 (9.00%)
  C4            40 (3.70%)   34 (3.15%)

Overall, 525 channel records were flagged, involving 208 unique trials.

These are statistical outlier flags, not confirmed artifacts.

## 8.4 Statistical Comparison Between Classes

The Mann--Whitney U test was used to compare Left and Right
distributions.

Benjamini--Hochberg False Discovery Rate correction was applied across
the 48 time-domain features.

Result:

-   Significant features at q \< 0.05: 0 / 48.
-   The smallest adjusted p-value was approximately 0.9557.
-   Observed effect sizes were generally small.

No statistically significant individual time-domain differences were
detected after correction.

This does not establish that multivariate classification is impossible.

# 9. Frequency-Domain Feature Engineering

Frequency-domain features were extracted using Welch's Power Spectral
Density estimation.

## 9.1 Welch Configuration

  Parameter                              Value
  --------------------------- ----------------
  Provisional Sampling Rate             250 Hz
  nperseg                                1,024
  noverlap                                 512
  Frequency bins                           513
  Frequency resolution          0.244140625 Hz
  Assumed Nyquist frequency             125 Hz

The sampling rate was assumed provisionally and was not confirmed by
acquisition documentation.

## 9.2 Frequency Bands

  Band    Frequency Range
  ------- -----------------
  Delta   0.5--4 Hz
  Theta   4--8 Hz
  Alpha   8--13 Hz
  Mu      8--12 Hz
  Beta    13--30 Hz
  Gamma   30--40 Hz

## 9.3 Extracted Frequency Features

Twenty features were calculated for each trial-channel combination:

-   Dominant Frequency.
-   Total Power.
-   Spectral Centroid.
-   Spectral Bandwidth.
-   Spectral Entropy.
-   Delta Power and Relative Power.
-   Theta Power and Relative Power.
-   Alpha Power and Relative Power.
-   Mu Power and Relative Power.
-   Beta Power and Relative Power.
-   Gamma Power and Relative Power.

The resulting frequency feature table contained:

`8,640 rows × 20 columns`

This corresponds to 2,160 trials × 4 channels.

## 9.4 50 Hz Region Investigation

A strong concentration of spectral power was observed in the 45--55 Hz
region.

The average relative power in this region was approximately:

`0.926409`

This corresponds to approximately 92.64% of the total spectral power
under the provisional frequency calculation.

The mean relative powers in adjacent investigated regions were much
smaller:

  Frequency Region     Mean Relative Power
  ------------------ ---------------------
  40--45 Hz                       0.001505
  45--55 Hz                       0.926409
  55--60 Hz                       0.001077
  60--80 Hz                       0.000984
  80--125 Hz                      0.001576

This may be consistent with strong power-line interference around 50 Hz,
but it cannot be confirmed without reliable sampling-rate and
acquisition information.

It may also reflect recording or export problems.

The 50 Hz concentration affects the interpretation of total power,
dominant frequency, spectral centroid, bandwidth, entropy, and relative
band power.

## 9.5 Mu and Beta Investigation

Mu and Beta power and relative power were compared between Left and
Right classes.

The comparisons used Mann--Whitney U tests and Benjamini--Hochberg
correction.

Results:

-   No significant Mu or Beta comparisons were detected.
-   Adjusted p-values were approximately 0.931187.
-   The maximum absolute effect size was approximately 0.024458.

The Left and Right median PSD curves were also visually similar in the
investigated 1--45 Hz range.

No strong, consistent class-specific Mu or Beta separation was
established.

All frequency-domain conclusions remain provisional because the sampling
rate is unconfirmed.

# 10. Motor Imagery Feature Engineering

Four-channel frequency features were used to derive additional motor
imagery-oriented features.

The resulting table contained:

`2,160 rows × 14 columns`

It included:

-   Trial Index.
-   Label.
-   C3 Mu Power.
-   C4 Mu Power.
-   C3 Beta Power.
-   C4 Beta Power.
-   C3 Mu Relative Power.
-   C4 Mu Relative Power.
-   C3 Beta Relative Power.
-   C4 Beta Relative Power.
-   C3 Mu/Beta Ratio.
-   C4 Mu/Beta Ratio.
-   C3--C4 Mu Power Difference.
-   C3--C4 Beta Power Difference.

These features were derived from the frequency-domain calculations and
therefore inherit the sampling-rate uncertainty.

## 10.1 ERD/ERS

Event-Related Desynchronization and Synchronization (ERD/ERS) were not
calculated.

The required cue timing and baseline interval were unavailable.

Mu/Beta ratios and C3--C4 differences are not substitutes for ERD/ERS.

# 11. Combined Feature Dataset

The time-domain, frequency-domain, and motor imagery feature tables were
combined.

Eight duplicated `_mi` columns were removed from the combined table
because they duplicated corresponding frequency feature columns.

The final combined feature dataset contains:

`2,160 rows × 122 columns`

Structure:

  Component                     Count
  --------------------------- -------
  Time-Domain Features             48
  Frequency-Domain Features        68
  Motor Imagery Features            4
  Trial Index                       1
  Label                             1
  Total Columns                   122

The final dataset contains 120 numerical features, one identifier, and
one target label.

Saved dataset:

`pre_modeling_feature_dataset.csv`

No feature was removed because of correlation or redundancy during the
final audit.

# 12. Feature Metadata and Audit

A feature metadata table was created to document each feature.

It contains:

-   Feature name.
-   Feature group.
-   Target indicator.
-   Identifier indicator.
-   Validation status.
-   Scientific meaning.

The metadata table contains 122 rows, including the 120 features,
identifier, and target.

A separate feature review report records:

-   Feature group.
-   Target and identifier indicators.
-   Validation status.
-   Scientific meaning.
-   High-correlation count.
-   Review priority.
-   Reliability classification.

# 13. Feature Correlation Analysis

Correlation analysis was performed across all 120 numerical features.

The correlation matrix shape was:

`120 × 120`

High-correlation pairs were extracted using the upper triangle of the
matrix to avoid counting both directions of the same relationship.

Threshold:

`Absolute correlation ≥ 0.90`

Results:

  Metric                            Result
  ------------------------------- --------
  Total possible unique pairs        7,140
  High-correlation unique pairs        176
  Percentage                         2.47%

## 13.1 Correlation by Feature Groups

  Feature Group Relationship               Pair Count   Mean Correlation
  -------------------------------------- ------------ ------------------
  Time-Domain vs Time-Domain                       80           0.955589
  Frequency-Domain vs Frequency-Domain             76           0.943710
  Time-Domain vs Frequency-Domain                  19           0.925104
  Motor Imagery vs Motor Imagery                    1           0.915157

The results indicate substantial statistical relationships among some
features.

High correlation alone does not establish mathematical redundancy.

# 14. Time-Domain Redundancy Investigation

The 80 high-correlation time-domain pairs were investigated.

## 14.1 Confirmed Mathematical Redundancy

Peak-to-Peak range was compared with Maximum minus Minimum.

The validation confirmed:

-   FZ: exact equality.
-   C3: exact equality.
-   CZ: exact equality.
-   C4: exact equality.

The maximum and mean absolute differences were zero for all four
channels.

Decision:

`Confirmed redundancy`

Action:

`Consider removing PTP later`

No PTP feature was deleted during this phase.

## 14.2 Standard Deviation and RMS

Standard Deviation and RMS showed perfect correlation in the
investigated feature relationships.

However, they were not mathematically equal.

Decision:

`Near-redundancy`

Action:

`Review relative differences`

Both features were retained.

## 14.3 Other Relationships

Several other relationships involved IQR, line length, minimum, maximum,
standard deviation, RMS, and peak-to-peak measurements.

Most strong relationships occurred within the same channel.

Cross-channel relationships may reflect shared EEG activity.

No automatic feature removal was performed.

# 15. Frequency-Domain Redundancy Investigation

The frequency-domain investigation identified 76 high-correlation pairs.

The main findings were:

## 15.1 Alpha and Mu Overlap

Alpha: 8--13 Hz.

Mu: 8--12 Hz.

These bands overlap substantially.

The following relationships were recorded:

-   Alpha Power vs Mu Power.
-   Alpha Relative Power vs Mu Relative Power.

Decision:

`Confirmed Feature Overlap`

This refers to overlapping band definitions, not exact equality of the
resulting feature values.

Both feature types were retained pending final feature selection.

## 15.2 Beta and Gamma Correlation

Beta: 13--30 Hz.

Gamma: 30--40 Hz.

High correlations were observed despite the absence of direct
frequency-band overlap.

The relationships were classified as requiring investigation.

The investigation considered the strong spectral concentration around
the 50 Hz region and overall signal quality.

No confirmed redundancy was established.

Both feature types were retained.

## 15.3 Total Power Across Channels

Strong cross-channel total-power relationships were identified.

Decision:

`Keep`

These relationships may represent shared EEG activity and are not
sufficient grounds for removing features.

## 15.4 Other Frequency Relationships

Other high-correlation relationships were retained for now.

No feature was removed based on correlation alone.

# 16. Motor Imagery Redundancy Investigation

One high-correlation pair was identified:

`Mu C3-C4 Difference vs Beta C3-C4 Difference`

Correlation:

`0.915157`

Decision:

`Keep - Requires Review`

Action:

Retain both features because no confirmed mathematical redundancy was
established.

# 17. Final Redundancy Audit

The final redundancy audit was rebuilt to correct the earlier grouping
of frequency-domain relationships under the time-domain category.

The final audit contains nine decisions.

  -----------------------------------------------------------------------
  Relationship                        Decision
  ----------------------------------- -----------------------------------
  PTP vs MAX-MIN                      Confirmed redundancy

  STD vs RMS                          Near-redundancy

  Alpha Power vs Mu Power             Confirmed Feature Overlap

  Alpha Relative Power vs Mu Relative Confirmed Feature Overlap
  Power                               

  Beta Power vs Gamma Power           Requires Investigation

  Beta Relative Power vs Gamma        Requires Investigation
  Relative Power                      

  Total Power across channels         Keep

  Other Frequency Relationships       Keep for Now

  Mu C3-C4 Difference vs Beta C3-C4   Keep - Requires Review
  Difference                          
  -----------------------------------------------------------------------

Final audit shape:

`(9, 4)`

The audit contains four columns:

-   Relationship.
-   Finding.
-   Decision.
-   Action.

No features were deleted as a result of this audit.

# 18. Feature Reliability and Review Priority

A reliability review table was created for all 120 numerical features.

The original review flags included:

-   Sampling Rate Unconfirmed.
-   Signal Quality Review.
-   Near Redundant.
-   Mathematical Redundancy Candidate.
-   Alpha/Mu Overlap.
-   High Correlation Review.

The flag distribution was:

  Review Flag                           Features
  ----------------------------------- ----------
  Sampling Rate Unconfirmed                   72
  Signal Quality Review                       48
  Near Redundant                               8
  Mathematical Redundancy Candidate            4
  Alpha/Mu Overlap                            16
  High Correlation Review                    120

The final review priority was assigned using the following categories:

-   Critical Review.
-   Technical Review.
-   Correlation Review.
-   No Review Required.

The current classification produced:

  Feature Group      Review Priority      Feature Count
  ------------------ ------------------ ---------------
  Frequency-Domain   Critical Review                 16
  Frequency-Domain   Technical Review                52
  Motor Imagery      Technical Review                 4
  Time-Domain        Critical Review                  4
  Time-Domain        Technical Review                44

Overall:

  Priority             Feature Count
  ------------------ ---------------
  Critical Review                 20
  Technical Review               100
  Total                          120

These priorities indicate which features require further investigation.
They are not definitive scientific reliability scores and do not mean
that the features are unusable.

# 19. Final Pre-Modeling Validation

The final validation checked the combined dataset, metadata, reliability
report, labels, and redundancy audit.

## 19.1 Dataset Validation

  Check                       Result
  -------------------- -------------
  Dataset Shape          2,160 × 122
  Number of Trials             2,160
  Number of Features             120
  Duplicate Trials                 0
  Missing Values                   0
  Infinite Values                  0

## 19.2 Metadata Validation

  Check                     Result
  ----------------------- --------
  Metadata Features            120
  Dataset Features             120
  Missing from Metadata          0
  Missing from Dataset           0

## 19.3 Reliability Validation

  Check                        Result
  -------------------------- --------
  Reliability Features            120
  Missing from Reliability          0
  Extra in Reliability              0

## 19.4 Label Validation

  Class              Count
  ---------------- -------
  Left               1,082
  Right              1,078
  Missing Labels         0

## 19.5 Redundancy Audit Validation

Total reviewed relationships: 9.

  Decision                      Count
  --------------------------- -------
  Confirmed Feature Overlap         2
  Requires Investigation            2
  Confirmed Redundancy              1
  Near-Redundancy                   1
  Keep                              1
  Keep for Now                      1
  Keep - Requires Review            1

All final structural validation checks passed.

# 20. Project Outputs and Generated Files

The project outputs are stored in:

`C:\Users\noura\Z-BCI-Motor\outputs`

Important generated artifacts include:

  ------------------------------------------------------------------------------------
  File                                             Description
  ------------------------------------------------ -----------------------------------
  `X_raw.npy`                                      Raw EEG data array

  `X_centered.npy`                                 Mean-centered EEG data

  `pre_modeling_feature_dataset.csv`               Combined dataset with 120 features

  `time_domain_features.csv`                       Time-domain feature table

  `high_correlation_pairs.csv`                     Unique high-correlation pairs

  `ptp_mathematical_validation.csv`                PTP mathematical validation

  `final_redundancy_audit.csv`                     Final redundancy decisions

  `final_feature_engineering_audit.csv`            Feature engineering audit

  `final_feature_reliability_classification.csv`   Feature reliability and review
                                                   classification

  `final_feature_reliability_summary.csv`          Reliability summary

  `cross_channel_correlation.csv`                  Cross-channel correlations

  `time_domain_statistical_comparison.csv`         Time-domain class comparisons

  `mi_statistical_comparison.csv`                  Motor imagery statistical
                                                   comparisons

  `motor_imagery_statistical_audit_summary.csv`    Motor imagery statistical summary

  `timestamp_quality_summary.csv`                  Timestamp and sampling-rate
                                                   investigation
  ------------------------------------------------------------------------------------

The notebook containing the full analysis is:

`C:\Users\noura\Z-BCI-Motor\01_Data_Collection_and_Validation.ipynb`

It contains 219 cells and 412 outputs.

The separate file under the `notebooks` directory was found to contain
only one empty cell and was not the main analysis notebook.

The image extraction destination was specified as:

`C:\Users\noura\Z-BCI-Motor\outputs\#Image`

The extraction script was prepared to extract embedded PNG, JPEG, SVG,
and WebP images from the correct notebook without executing the analysis
cells. The final extraction count has not yet been confirmed.

# 21. Outstanding Limitations and Future Work

The following items remain unresolved and should be considered before
starting Modeling.

## 21.1 Confirm Label Mapping

The current mapping uses the first 2,160 labels.

The remaining 240 labels are unmatched.

The mapping should be verified against the original experiment records.

## 21.2 Confirm Sampling Rate

The timestamps are irregular.

The provisional 250 Hz sampling rate must not be treated as confirmed.

The acquisition configuration or original recording documentation is
required.

## 21.3 Investigate Signal Quality

The repeated-sample and plateau findings require investigation.

The cause should be established before defining final trial-exclusion
rules.

## 21.4 Confirm Frequency-Domain Validity

Frequency-domain and motor imagery features depend on the sampling-rate
assumption.

The strong 45--55 Hz concentration also requires investigation.

## 21.5 Confirm Event Timing

Cue timing and baseline information are required to calculate ERD/ERS
correctly.

## 21.6 Future Preprocessing

After resolving the necessary recording information, possible processing
steps include:

-   Appropriate bandpass filtering.
-   Appropriate 50/60 Hz notch filtering if justified by the acquisition
    environment.
-   Resampling if required.
-   Artifact investigation and rejection criteria.
-   Rereferencing based on the actual recording reference scheme.
-   Epoching using verified cue timing.
-   Baseline correction.
-   Normalization fitted using training data only during a future
    modeling workflow.

These are future recommendations, not completed operations.

# 22. Final Project Status

The following activities were completed:

  Phase                                  Status
  -------------------------------------- -------------
  Data Collection and File Loading       Completed
  Structural Data Validation             Completed
  Timestamp Investigation                Completed
  Signal Quality Assessment              Completed
  Mean Centering                         Completed
  Exploratory Data Analysis              Completed
  Time-Domain Feature Engineering        Completed
  Frequency-Domain Feature Engineering   Completed
  Motor Imagery Feature Engineering      Completed
  Statistical Comparisons                Completed
  Feature Correlation Analysis           Completed
  Redundancy Investigation               Completed
  Feature Metadata Audit                 Completed
  Feature Reliability Classification     Completed
  Final Pre-Modeling Validation          Completed
  Machine Learning Modeling              Not Started

## Final Conclusion

The BCI Motor Imagery EEG dataset has completed its structural
validation, exploratory analysis, feature engineering, and pre-modeling
audit.

The final combined dataset contains 2,160 trials and 120 engineered
features, with no missing values, infinite values, or duplicate trial
identifiers.

The feature metadata and reliability tables match the final dataset
exactly.

The redundancy audit identified confirmed mathematical duplication,
overlapping frequency-band definitions, near-redundancy, and
high-correlation relationships requiring further investigation. No
features were deleted during the final audit.

However, structural completeness must not be confused with scientific
readiness. The sampling rate, timestamp irregularities, signal quality
concerns, and label mapping still require confirmation.

**The project is currently complete up to the Pre-Modeling stage. No
model training, model selection, hyperparameter tuning, or evaluation
has been performed.**
