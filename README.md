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

## What This Project Does For a User
The project is a research demo for machine olfaction: it takes a CSV recorded from a gas sensor array and estimates which smell class from SmellNet the signal most resembles. A user does not need to inspect model code to use it:

1. Open the Streamlit app.
2. Select one of the demo SmellNet CSV files or upload a compatible sensor CSV.
3. Choose a saved model artifact from the sidebar.
4. Open the **Prediction Results** tab.
5. Read the predicted smell class, confidence score, top-5 alternatives, and sensor plots.

The app also explains whether the CSV schema matches the trained model and labels low-confidence outputs. It is intended for dataset exploration and ML research, not production safety decisions.

## Why This Is More Than a CSV Project
The CSV file is only the storage format. The modeling problem is time-series smell recognition from an electronic-nose style sensor array.

The pipeline uses smell-signal assumptions throughout:

- one file is treated as one sensor trial, not independent rows
- early warm-up drift is trimmed before modeling
- each trial is resampled to a shared timeline
- sensor channels are normalized within the trial
- windows capture local response patterns over time
- features describe curve shape, drift, energy, frequency balance, and cross-sensor relationships
- window probabilities are aggregated back to one trial-level smell prediction

For a junior-friendly walkthrough of the research logic, see [docs/research_workflow.md](docs/research_workflow.md). For dataset assumptions, see [docs/data_dictionary.md](docs/data_dictionary.md).

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
tests/
	preprocess, feature, and prediction regression tests
```

## How To Read The Code

If you are reviewing this as a research project or onboarding as a junior contributor, read the files in this order:

1. `docs/QUICK_START_GUIDE.md` - **Start here if you're new!** A friendly guide for non-technical users.
2. `docs/research_workflow.md` for the project mental model.
3. `src/data/preprocess.py` for how raw sensor trials become comparable.
4. `src/features/extract_features.py` for the signal features used by classical ML models.
5. `src/models/train_baseline.py` for dataset building, splitting, model comparison, and artifact saving.
6. `src/models/predict.py` for how the app reproduces training-time preprocessing at inference.
7. `src/app/streamlit_app.py` for the demo layer.
8. `tests/` for small examples of expected behavior.

## Pipeline Summary
1. Inspect dataset structure and schema.
2. Visualize one sample to detect drift/noise/warm-up.
3. Preprocess (missing values, warm-up trimming, resampling, normalization).
4. Extract statistical, derivative, frequency, and cross-sensor features from time-series windows.
5. Train baseline and soft-voting ensemble models with leakage-aware grouped splits.
6. Evaluate with top-1, top-5, macro F1, weighted F1, confusion matrix.
7. Save full inference pipeline artifact.
8. Optionally train a time-series PyTorch model.
9. Serve predictions and plots through Streamlit.

## Preprocessing Decisions (Current Default)
- Missing values: forward fill then backward fill.
- Warm-up trimming: drop first 5 percent of rows.
- Resampling: interpolate each trial to 300 time points.
- Normalization: per-sensor z-score.
- Leakage guard: SmellNet `training`/`testing` folder split when present, with grouped validation by source file path.

## Models Implemented
- Baselines (scikit-learn):
	- LogisticRegression
	- RandomForestClassifier
	- ExtraTreesClassifier
	- HistGradientBoostingClassifier
	- Optional SVM (flag)
	- Soft-voting ensembles over tree, boosting, and SVM candidates
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
Use the sidebar **Navigation** menu to switch between:

- **Project Guide**: explains the project use case, why it is useful, how to navigate the app, and key limitations.
  - 📘 **New!** Also see `docs/QUICK_START_GUIDE.md` for a beginner-friendly walkthrough with screenshots and FAQs.
- **Prediction Demo**: loads saved models, accepts demo/uploaded CSVs, and shows model predictions.

1. Upload a sensor CSV or choose a real sample CSV copied from SmellNet.
2. Validate the uploaded schema against the saved model's expected sensor columns.
3. Preview raw sensor curves.
4. Run the saved baseline model with the same preprocessing and window aggregation used during evaluation.
5. Show predicted smell class, confidence label, analyzed window count, top-5 probabilities, and preprocessed curves.
6. Explain which sensor-response clues supported a logistic-regression prediction.
7. Review research evidence: ablation results, weakest classes, and common held-out test mix-ups.

## Model Selection in the App
The Streamlit sidebar automatically discovers saved pipelines at `models/*/model.joblib`. Each option shows the artifact folder, the best classifier inside that artifact, and the trial-level top-1 metric when available.

Current saved choices include:

- `models/baseline/model.joblib`: older full-sequence baseline.
- `models/baseline_v2/model.joblib`: older windowed random-forest baseline.
- `models/baseline_windowed/model.joblib`: earlier windowed soft-voting ensemble.
- `models/baseline_windowed_trialmax/model.joblib`: improved trial-level artifact selected with max window aggregation.

Use the **Models** tab to compare saved metrics and see the candidate models trained inside the selected artifact. The app selects the artifact with the highest saved trial top-1 score by default.

## Accuracy Improvement Path
The current accuracy improvement comes from the windowed feature baseline:

- creates multiple windows from each sensor trial
- extracts statistical, derivative, frequency, timing, and cross-sensor features
- can add whole-trial context and window-position features with `--use-context-features`
- can use max window aggregation with `--trial-aggregation max`; the current saved trial-max artifact reaches 64.0% trial top-1 and 92.0% trial top-5
- compares logistic regression, feature-selected logistic regression, optional cross-validated logistic regression, random forest, extra trees, histogram gradient boosting, SVM, and soft-voting ensembles
- selects the best validation model and reports held-out test metrics

Recommended training command:

```powershell
uv run python src/models/train_baseline.py --data-root data/raw/SmellNet --output-dir models/baseline_windowed_trialmax --window-size 100 --window-stride 25 --include-svm --include-logistic-cv --feature-select-k 80 --trial-aggregation max
```

After retraining, rerun evaluation and restart Streamlit:

```powershell
uv run python src/models/evaluate.py --model-path models/baseline_windowed_trialmax/model.joblib --eval-data models/baseline_windowed_trialmax/eval_data.npz --output-dir models/baseline_windowed_trialmax --trial-aggregation max
uv run streamlit run src/app/streamlit_app.py
```

## Current Results

| Model | Split Strategy | Top-1 | Top-5 | Macro F1 | Weighted F1 | Notes |
|---|---|---:|---:|---:|---:|---|
| Trial-max windowed logistic regression | SmellNet folder split, trial-level | 0.640 | 0.920 | 0.543 | 0.543 | Current best trial-level artifact; selected using max window aggregation |
| Trial-max windowed logistic regression | SmellNet folder split, window-level | 0.513 | 0.780 | 0.486 | 0.486 | Same artifact before trial aggregation |
| Contextual window random forest | SmellNet folder split, window-level | 0.596 | 0.827 | 0.506 | 0.506 | Adds whole-trial features and window position |
| Contextual window random forest | SmellNet folder split, trial-level | 0.580 | 0.820 | 0.486 | 0.486 | Mean aggregation; max aggregation reached 0.600 trial top-1 in post-run analysis |
| Soft-vote full windowed baseline | SmellNet folder split, window-level | 0.531 | 0.838 | 0.478 | 0.478 | 100-point windows, 25-point stride |
| Soft-vote full windowed baseline | SmellNet folder split, trial-level | 0.600 | 0.880 | 0.509 | 0.509 | Mean aggregation; max aggregation improves top-5 to 0.900 |
| TinySensorCNN | Group split | 0.100 | 0.300 | N/A | N/A | Experimental only; classical baseline is preferred |

The saved default app model is selected automatically by trial top-1; with the current artifacts this is `models/baseline_windowed_trialmax/model.joblib`.

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

## Documentation
- 📘 **Quick Start Guide**: `docs/QUICK_START_GUIDE.md` - Beginner-friendly walkthrough for non-technical users
- 📊 **Research Workflow**: `docs/research_workflow.md` - Technical deep-dive into the ML pipeline
- 📋 **Data Dictionary**: `docs/data_dictionary.md` - Dataset structure and schema details
- 🤖 **Model Card**: `docs/model_card.md` - Model performance, limitations, and intended use
