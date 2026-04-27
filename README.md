# Machine Olfaction: Smell Recognition from Gas Sensor Time-Series Data

## Motivation
Low-cost gas sensors are promising for practical smell recognition, but their signals are noisy, drift over time, and have warm-up effects.
This repository builds a reproducible end-to-end machine learning pipeline for smell classification from sensor time-series.

## Dataset
- Source: DeweiFeng/SmellNet
- Local location (recommended): data/raw/SmellNet
- Key subsets:
	- base_data: single-substance sensor recordings for classification
	- mixture_data: mixture composition metadata and linked sensor recordings
	- gcms_data and gcms_processed: chemistry priors and related files

## Problem Formulation
- Primary baseline task: multiclass classification of smell classes from base_data sensor time-series.
- Input: one trial CSV of sensor readings over time.
- Output: predicted smell class plus class probabilities.

## Project Structure
```
data/
	raw/            # immutable source data
	interim/        # intermediate transformed data
	processed/      # model-ready tables
	samples/        # lightweight demo CSVs for app users
docs/
	data_dictionary.md
models/
	baseline/
	timeseries/
src/
	eda/
	data/
	features/
	models/
	app/
```

## Pipeline Summary
1. Inspect dataset structure and schema.
2. Visualize one sample to detect drift/noise/warm-up.
3. Preprocess (missing values, warm-up trimming, resampling, normalization).
4. Extract fixed statistical features from time-series windows.
5. Train baseline models with leakage-aware grouped splits.
6. Evaluate with top-1, top-5, macro F1, weighted F1, confusion matrix.
7. Save full inference pipeline artifact.
8. Optionally train a time-series PyTorch model.
9. Serve predictions and plots through Streamlit.

## Preprocessing Decisions (Current Default)
- Missing values: forward fill then backward fill.
- Warm-up trimming: drop first 5 percent of rows.
- Resampling: interpolate each trial to 300 time points.
- Normalization: per-sensor z-score.
- Leakage guard: GroupShuffleSplit by source file path.

## Models Implemented
- Baselines (scikit-learn):
	- LogisticRegression
	- RandomForestClassifier
	- ExtraTreesClassifier
	- HistGradientBoostingClassifier
	- Optional SVM (flag)
- Improved sequence model (PyTorch):
	- Tiny 1D CNN baseline in src/models/train_timeseries.py

## Metrics
- Top-1 accuracy
- Top-5 accuracy
- Macro F1
- Weighted F1
- Trial-level top-1/top-5 accuracy by averaging window probabilities per original CSV
- Confusion matrix
- Per-class report

## Setup
From project root:

```powershell
uv sync
```

## Usage

### 1) Inspect dataset quickly
```powershell
uv run python inspect_smellnet_preview.py --data-dir data/raw/SmellNet --subset base_data --max-files 5 --head 4
```

### 2) Plot one smell sample
```powershell
uv run python src/eda/plot_one_sample.py --csv-path data/raw/SmellNet/base_data/allspice_6.csv
```

### 3) Train baseline feature model
```powershell
uv run python src/models/train_baseline.py --data-root data/raw/SmellNet --output-dir models/baseline_windowed --window-size 100 --window-stride 25 --include-svm
```

### 4) Evaluate saved baseline model
```powershell
uv run python src/models/evaluate.py --model-path models/baseline_windowed/model.joblib --eval-data models/baseline_windowed/eval_data.npz --output-dir models/baseline_windowed
```

### 5) Optional: train improved time-series model
```powershell
uv run python src/models/train_timeseries.py --data-root data/raw/SmellNet --output-dir models/timeseries
```

### 6) Create lightweight demo sample CSVs
```powershell
uv run python src/data/make_samples.py --data-root data/raw/SmellNet --output-dir data/samples
```

### 7) Run Streamlit demo
```powershell
uv run streamlit run src/app/streamlit_app.py
```

## Streamlit Demo Behavior
1. Upload a sensor CSV or choose a real sample CSV copied from SmellNet.
2. Validate the uploaded schema against the saved model's expected sensor columns.
3. Preview raw sensor curves.
4. Run the saved baseline model with the same preprocessing and window aggregation used during evaluation.
5. Show predicted smell class, confidence label, analyzed window count, top-5 probabilities, and preprocessed curves.

## Current Results

| Model | Split Strategy | Top-1 | Top-5 | Macro F1 | Weighted F1 | Notes |
|---|---|---:|---:|---:|---:|---|
| Extra Trees windowed baseline | SmellNet folder split, window-level | 0.398 | 0.700 | 0.364 | 0.364 | 100-point windows, 25-point stride |
| Extra Trees windowed baseline | SmellNet folder split, trial-level | 0.460 | 0.740 | 0.408 | 0.408 | Window probabilities averaged per source CSV |
| TinySensorCNN | Group split | 0.100 | 0.300 | N/A | N/A | Experimental only; classical baseline is preferred |

The saved default app model is `models/baseline_windowed/model.joblib`.

## Limitations
- Sensor drift and environment changes can reduce real-world robustness.
- Class balance and repeated-trial structure may affect generalization.
- Mixture predictions require distinct target handling beyond base classification.
- The demo is not a food-safety, allergen, quality-control, or hazardous-gas safety system.
- Predictions are only meaningful for CSVs with the same sensor columns and similar acquisition conditions as SmellNet base_data.

## Future Work
- Domain adaptation for cross-session drift.
- Sequence models with attention and uncertainty calibration.
- Multi-task training with mixture and GC-MS priors.
- Better deployment packaging and model monitoring.
