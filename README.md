# Machine Olfaction: Smell Recognition from Gas Sensor Time-Series Data

## Overview
This project uses gas sensor recordings to predict smell classes from the SmellNet dataset. Each CSV file is treated as one smell exposure over time. The system cleans the sensor signal, extracts response-pattern features, trains baseline models, and serves predictions through a Streamlit app.

The project is designed to be understandable as well as runnable. A non-technical user can open the app, choose a sample sensor file, and see the predicted smell class, confidence, top alternatives, and sensor curves.

## Why This Project Exists
Smell is difficult to describe and measure consistently. Low-cost gas sensors can help, but their readings are noisy, can drift over time, and often change during warm-up.

This repository turns those sensor readings into a repeatable workflow for smell classification. It is intended for learning, research, and dataset exploration, not for real-world safety decisions.

## Dataset
- Source: DeweiFeng/SmellNet
- Local location (recommended): data/raw/SmellNet
- Key subsets:
  - base_data: single-substance sensor recordings for classification
  - mixture_data: mixture composition metadata and linked sensor recordings
  - gcms_data and gcms_processed: chemistry-related files

## Problem Formulation
- Primary baseline task: multiclass classification of smell classes from base_data sensor time-series.
- Input: one trial CSV of sensor readings over time.
- Output: predicted smell class plus class probabilities.

## What A User Can Do
A user does not need to inspect the model code to try the project:

1. Open the Streamlit app.
2. Select one of the demo SmellNet CSV files or upload a compatible sensor CSV.
3. Choose a saved model artifact from the sidebar.
4. Open the **Prediction Results** tab.
5. Read the predicted smell class, confidence score, top-5 alternatives, and sensor plots.

The app also explains whether the CSV schema matches the trained model and labels low-confidence outputs. It is intended for dataset exploration and ML research, not production safety decisions.

## Why This Is Not Just a CSV Classifier
The CSV file is only the storage format. The actual problem is time-series smell recognition from a gas sensor array.

The pipeline is built around sensor behavior:

- one file is treated as one sensor trial, not independent rows
- early warm-up drift is trimmed before modeling
- each trial is resampled to a shared timeline
- sensor channels are normalized within the trial
- windows capture local response patterns over time
- features describe curve shape, drift, energy, frequency balance, and cross-sensor relationships
- window probabilities are aggregated back to one trial-level smell prediction

The README includes the main project explanation, setup steps, model behavior, results, and limitations.

## Project Structure
```
data/
  raw/            # immutable source data
  interim/        # intermediate transformed data
  processed/      # model-ready tables
  samples/        # lightweight demo CSVs for app users
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

If you are new to the project, read the files in this order:

1. `README.md` for the project story, setup steps, usage, results, and limitations.
2. `src/app/streamlit_app.py` for the user-facing Streamlit demo.
3. `src/models/predict.py` for how the app runs saved-model inference.
4. `src/data/preprocess.py` for how raw sensor trials become comparable.
5. `src/features/extract_features.py` for the signal features used by classical ML models.
6. `src/models/train_baseline.py` for dataset building, splitting, model comparison, and artifact saving.
7. `tests/` for small examples of expected behavior.

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
  - Tiny 1D CNN baseline in `src/models/train_timeseries.py`

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

### 5) Train InceptionTime Deep Neural Network
```powershell
uv run python src/models/train_inception.py --data-root data/raw/SmellNet --output-dir models/inception_time --epochs 25
```

### 6) Train Smell Mixture Deconvolution Regressor
```powershell
uv run python src/models/train_mixture.py --data-root data/raw/SmellNet/mixture_data --output-dir models/mixture_regressor
```

### 7) Run FastAPI Real-Time Microservice (for IoT / Edge Devices)
```powershell
uv run uvicorn src.api.main:app --reload --port 8000
```
*Interactive Swagger API documentation available at `http://localhost:8000/docs`.*

### 8) Run Streamlit Research App
```powershell
uv run streamlit run src/app/streamlit_app.py
```

## Streamlit App Capabilities
The interactive web app includes:
- **Prediction Results**: Full multiclass smell recognition, confidence gauge, top-5 probability distribution, preprocessed model inputs, and linear feature contribution explanations.
- **Aroma & Chemical Volatiles Profile**: Botanical categorization, key volatile organic compounds (VOCs - *e.g., Eugenol, Limonene, Cinnamaldehyde, Allicin*), and sensory notes integrated from `SmellNet text_data`.
- **Signal Analysis**: Interactive Plotly multi-channel response curves and per-sensor statistical summaries.
- **Mixture Deconvolution**: Decomposition of compound smell signals into constituent ingredients and percentage shares (e.g. 80% Orange + 20% Almond).
- **Hardware Drift & Stress Simulator**: Real-time playground to inject thermal drift, electrical Gaussian noise, and sensor channel dropout to test model resilience live.
- **Sensor Hardware Importance**: Ranking of gas sensor channels by predictive contribution for physical array pruning and BOM hardware cost reduction.
- **Odor Knowledge Base**: Searchable catalog of all 50 target smell classes, chemical formulas, and sensory flavor descriptors.
- **Research Evidence**: Error analysis, confusion matrix, weak classes, and held-out test evaluation.

## Model Selection in the App
The Streamlit sidebar automatically discovers saved pipelines at `models/*/model.joblib`. Each option shows the artifact folder, the best classifier inside that artifact, and the trial-level top-1 metric when available.

Current saved choices include:

- `models/baseline/model.joblib`: older full-sequence baseline.
- `models/baseline_v2/model.joblib`: older windowed random-forest baseline.
- `models/baseline_windowed/model.joblib`: earlier windowed soft-voting ensemble.
- `models/baseline_windowed_trialmax/model.joblib`: improved trial-level artifact selected with max window aggregation.

Use the **Models** tab to compare saved metrics and see the candidate models trained inside the selected artifact. The app selects the artifact with the highest saved trial top-1 score by default.

## Current Accuracy Path
The strongest current results come from the windowed feature baseline:

- creates multiple windows from each sensor trial
- extracts statistical, derivative, frequency, timing, and cross-sensor features
- can add whole-trial context and window-position features with `--use-context-features`
- can use max window aggregation with `--trial-aggregation max`; the current saved trial-max artifact reaches 64.0% trial top-1 and 92.0% trial top-5
- compares logistic regression, feature-selected logistic regression, optional cross-validated logistic regression, random forest, extra trees, histogram gradient boosting, SVM, and soft-voting ensembles
- selects the best validation model and reports held-out test metrics

Training command for the current best saved artifact:

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


