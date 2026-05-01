# Research Workflow

This project is a machine olfaction baseline, not a generic CSV classifier. The important idea is that each CSV is a gas-sensor trial: rows are time, columns are sensor channels, and the smell label is attached to the whole trial.

## One-Screen Mental Model

```text
SmellNet CSV trial
    -> detect time and sensor columns
    -> fill missing readings
    -> remove early warm-up drift
    -> resample to a common timeline
    -> normalize each sensor within the trial
    -> split into time windows
    -> extract signal features
    -> classify each window
    -> aggregate window probabilities
    -> report one smell prediction for the CSV
```

## Why This Is Machine Olfaction

Gas sensors do not usually identify an odor from one static value. A useful signal is the response pattern over time: how each channel rises, stabilizes, drifts, peaks, and relates to other channels. The feature pipeline therefore uses signal-shape information instead of treating the CSV as an arbitrary table.

Implemented signal ideas:

| Idea | Code location | Why it matters |
|---|---|---|
| Warm-up trimming | `src/data/preprocess.py` | Early sensor readings can be unstable. |
| Fixed timeline resampling | `src/data/preprocess.py` | Trials become comparable even when they have different row counts. |
| Per-trial normalization | `src/data/preprocess.py` | Reduces scale differences between trials. |
| Sliding windows | `src/models/train_baseline.py` | Lets the model learn local response patterns inside a trial. |
| Statistical and shape features | `src/features/extract_features.py` | Captures sensor level, spread, drift, slope, peak timing, and energy. |
| Frequency features | `src/features/extract_features.py` | Captures low-vs-high variation in the signal. |
| Cross-sensor features | `src/features/extract_features.py` | Electronic-nose behavior depends on relationships across channels. |
| Trial-level aggregation | `src/models/predict.py` | The final prediction is for the full uploaded CSV, not one row. |

## File Map For Junior Contributors

Start here if you are new to the repository:

| File | Purpose |
|---|---|
| `inspect_smellnet_preview.py` | Quick dataset inspection before training. |
| `src/data/preprocess.py` | All CSV cleaning and time-series standardization. |
| `src/features/extract_features.py` | Converts sensor windows into fixed ML features. |
| `src/models/train_baseline.py` | End-to-end classical model training script. |
| `src/models/evaluate.py` | Recomputes metrics and confusion-matrix artifacts. |
| `src/models/predict.py` | Shared inference path used by the app. |
| `src/app/streamlit_app.py` | Research demo UI for uploads, plots, predictions, and model comparison. |
| `tests/` | Small regression tests for preprocessing, features, and prediction. |

## Current Baseline Choice

The main saved baseline is `models/baseline_windowed_trialmax`, a windowed classical model selected with trial-level max aggregation. It is intentionally used before deep learning because it is easier to inspect and debug:

- preprocessing choices are explicit
- feature names are saved with the model artifact
- multiple classifiers are compared
- the Streamlit app shows top-5 probabilities and confidence
- evaluation reports both window-level and trial-level metrics

The optional PyTorch model in `src/models/train_timeseries.py` is experimental. It should only become the main path after it beats the classical baseline with a comparable split and real metrics.

## Common Extension Ideas

Good next steps for a research showcase:

1. Add feature-importance plots for tree-based models.
2. Add per-class error analysis from `per_class_report.csv`.
3. Add drift experiments by evaluating classes across collection sessions if the metadata supports it.
4. Add calibration metrics so confidence scores are easier to interpret.
5. Add a mixture-label task instead of forcing mixture files into the base-class classifier.

## Responsible Use

This project is for research exploration on SmellNet-style data. It should not be presented as a production food-safety, allergen, hazardous-gas, or quality-control system.
