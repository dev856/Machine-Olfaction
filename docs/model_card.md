# Model Card: SmellNet Windowed Trial-Max Baseline

## Intended Use

This model is a research demo for multiclass smell recognition from gas sensor time-series CSV files in the SmellNet `base_data` subset. It predicts one of the known base smell classes and displays top-5 class probabilities.

It is not intended for food safety, allergen detection, medical decisions, hazardous gas monitoring, or production quality control.

## Model

- Artifact: `models/baseline_windowed_trialmax/model.joblib`
- Selected estimator: `logistic_regression`
- Input sensors: `NO2`, `C2H5OH`, `VOC`, `CO`, `Alcohol`, `LPG`
- Preprocessing: forward/backward fill, 5% warm-up trim, interpolation to 300 points, per-trial z-score normalization
- Windowing: 100 time points with stride 25
- Trial aggregation: max probability across windows, normalized before scoring
- Features: statistical, shape, derivative, low/high frequency, and cross-sensor interaction features

## Evaluation

The model was trained on the SmellNet `base_data/training` folder and evaluated on `base_data/testing`.

| Level | Top-1 | Top-5 | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Window | 0.513 | 0.780 | 0.486 | 0.486 |
| Trial CSV | 0.640 | 0.920 | 0.543 | 0.543 |

Trial-level metrics aggregate window probabilities for each original CSV before scoring. This better matches the Streamlit app, where users submit one CSV at a time.

## Limitations

- Performance is uneven across classes; inspect `models/baseline_windowed/per_class_report.csv` before drawing class-specific conclusions.
- The model assumes the same sensor schema and similar collection conditions as SmellNet.
- Gas sensor drift, humidity, temperature, hardware differences, and acquisition timing can reduce reliability.
- Mixture samples are not modeled as mixture labels by this baseline.

## Responsible Use

Treat predictions as exploratory signals. Low or medium confidence should be interpreted as uncertainty, not as proof of absence. Do not use this model as a safety system.
