# SmellNet Data Dictionary

This document records the dataset assumptions used by the current code. It avoids hard-coding claims that must come from the dataset itself, such as class counts or units, unless they are visible in the local files or model artifact.

## Dataset Location

Expected local root:

```text
data/raw/SmellNet
```

Primary subset used by the baseline:

```text
data/raw/SmellNet/base_data
```

The training script also detects SmellNet folder splits when the following folders are present:

```text
base_data/training
base_data/testing
```

## Baseline Learning Unit

| Concept | Current assumption |
|---|---|
| One CSV file | One gas-sensor trial |
| Rows | Ordered time steps within the trial |
| Numeric sensor columns | Gas sensor channels |
| Target label | Inferred from the CSV filename |
| Prediction target | One smell class for the whole CSV |
| Mixture files | Not modeled as mixture labels by the current baseline |

## Label Parsing Rule

Labels are inferred from filenames in `src/models/train_baseline.py`.

Examples:

| Filename | Parsed label |
|---|---|
| `allspice_6.csv` | `allspice` |
| `black_pepper_12.csv` | `black_pepper` |
| `unknown.csv` | `unknown` |

Rule:

```text
If the filename ends with _<number>, remove that final numeric suffix.
Otherwise use the full filename stem.
```

## Time Column Detection

Time columns are detected in `src/data/preprocess.py`.

Preferred names:

```text
time, timestamp, timestamp_ms, ts, t, second, seconds, sec, sample_idx, index
```

Fallback:

```text
Any column whose normalized name contains "time" or ends with "_ms".
```

If no time column is found, row order is treated as the timeline.

## Sensor Column Detection

Sensor columns are inferred as numeric columns after excluding likely metadata fields.

Excluded metadata hints:

```text
label, class, target, id, trial, sample, filepath, file_path
```

A numeric column must also have at least three unique non-missing values by default. This prevents constants and IDs from being treated as sensors.

## Sensors In The Saved Windowed Baseline

The current model card lists these expected sensor columns for `models/baseline_windowed/model.joblib`:

```text
NO2, C2H5OH, VOC, CO, Alcohol, LPG
```

Uploads missing any expected trained sensor are blocked by the app instead of silently making a bad prediction.

## Preprocessing Rules

| Step | Default |
|---|---|
| Missing values | Forward fill, then backward fill |
| Warm-up trimming | Drop first 5 percent of rows |
| Resampling | Interpolate to 300 points |
| Normalization | Per-trial z-score for each sensor |
| Windowing | Full trial when `--window-size 0`; otherwise sliding windows |

## Feature Groups

Features are created in `src/features/extract_features.py`.

| Group | Examples |
|---|---|
| Level and spread | mean, standard deviation, median, quartiles, min, max |
| Shape over time | slope, area under curve, final value, delta |
| Timing | time to max, time to min |
| Dynamics | first differences, response energy |
| Frequency | low power, high power, low/high balance |
| Cross-sensor relationships | pairwise correlation, mean difference, delta difference |

## Open Data Questions

These should be checked against the source dataset or paper before making stronger claims:

- Exact units for each gas sensor channel.
- Whether all SmellNet subsets use the same sensor schema.
- Whether acquisition conditions such as humidity, temperature, or collection session are available.
- Whether mixture labels should be represented as multi-label targets, proportions, or a separate task.

## Responsible Interpretation

The baseline is meaningful only for CSVs with the same sensor schema and similar acquisition conditions as the training data. It is a research classifier, not a safety detector.
