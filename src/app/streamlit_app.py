from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import identify_time_column, infer_sensor_columns
from src.models.pipeline_io import load_pipeline
from src.models.predict import predict_dataframe, validate_uploaded_schema


@st.cache_resource(show_spinner=False)
def cached_load_pipeline(model_path: str) -> dict[str, Any]:
    return load_pipeline(Path(model_path))


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --mo-bg: #f7f9fa;
            --mo-surface: #ffffff;
            --mo-surface-soft: #f4faf9;
            --mo-border: #d7e1e4;
            --mo-text: #183c40;
            --mo-muted: #526a6e;
            --mo-accent: #2f6f73;
            --mo-accent-warm: rgba(235, 152, 78, 0.16);
            --mo-shadow: rgba(32, 52, 58, 0.08);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --mo-bg: #0e1116;
                --mo-surface: #161b22;
                --mo-surface-soft: #1d252b;
                --mo-border: #34454c;
                --mo-text: #edf6f7;
                --mo-muted: #b5c6ca;
                --mo-accent: #6cc3c8;
                --mo-accent-warm: rgba(235, 152, 78, 0.12);
                --mo-shadow: rgba(0, 0, 0, 0.28);
            }
        }
        .stApp { background: var(--mo-bg); color: var(--mo-text); }
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1380px; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            border: 1px solid var(--mo-border);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: var(--mo-surface);
            box-shadow: 0 1px 2px var(--mo-shadow);
        }
        div[data-testid="stAlert"] { border-radius: 8px; }
        .app-hero {
            border: 1px solid var(--mo-border);
            border-radius: 8px;
            padding: 1.05rem 1.2rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(135deg, var(--mo-surface), var(--mo-surface-soft)),
                radial-gradient(circle at 84% 18%, var(--mo-accent-warm), transparent 26%);
            box-shadow: 0 8px 24px var(--mo-shadow);
        }
        .app-hero h1 { margin-bottom: 0.25rem; }
        .app-hero p { color: var(--mo-muted); margin: 0.2rem 0 0 0; }
        .hero-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.95rem;
        }
        .hero-step {
            border: 1px solid var(--mo-border);
            border-left: 3px solid var(--mo-accent);
            background: var(--mo-surface);
            padding: 0.65rem 0.7rem;
            border-radius: 6px;
            min-height: 86px;
        }
        .hero-step strong { display: block; color: var(--mo-text); margin-bottom: 0.2rem; }
        .hero-step span { color: var(--mo-muted); font-size: 0.92rem; }
        .usecase-card {
            border: 1px solid var(--mo-border);
            border-radius: 8px;
            padding: 0.85rem;
            background: var(--mo-surface);
            min-height: 132px;
        }
        .usecase-card strong { color: var(--mo-text); }
        .usecase-card p { margin: 0.35rem 0 0 0; color: var(--mo-muted); font-size: 0.94rem; }
        .status-chip {
            display: inline-block;
            border: 1px solid var(--mo-border);
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            color: var(--mo-text);
            background: var(--mo-surface-soft);
            font-size: 0.88rem;
            margin-right: 0.35rem;
        }
        .result-panel {
            border: 1px solid var(--mo-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: var(--mo-surface);
            margin-bottom: 0.75rem;
        }
        .result-panel .label { color: var(--mo-muted); font-size: 0.9rem; margin-bottom: 0.1rem; }
        .result-panel .value { color: var(--mo-text); font-size: 1.75rem; font-weight: 700; line-height: 1.15; }
        .result-panel .note { color: var(--mo-muted); margin-top: 0.35rem; }
        .guide-table {
            border: 1px solid var(--mo-border);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 0.75rem;
        }
        @media (max-width: 900px) {
            .hero-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 620px) {
            .hero-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sample_csvs(sample_root: Path) -> list[Path]:
    if not sample_root.exists():
        return []
    return [path for path in sorted(sample_root.rglob("*.csv")) if ".cache" not in path.parts]


def discover_model_artifacts(models_root: Path = Path("models")) -> list[Path]:
    if not models_root.exists():
        return []
    return sorted(models_root.glob("*/model.joblib"))


def metric_value(metrics: dict, key: str) -> float | None:
    test_metrics = metrics.get("test_metrics", {})
    value = test_metrics.get(key)
    if value is None and key.startswith("trial_"):
        value = test_metrics.get(key.replace("trial_", "", 1))
    return float(value) if value is not None else None


def model_display_name(model_path: Path) -> str:
    metrics = load_metrics(model_path)
    best_model = metrics.get("best_model", "unknown")
    trial_top1 = metric_value(metrics, "trial_top1_accuracy")
    suffix = f" | trial top-1 {trial_top1 * 100:.1f}%" if trial_top1 is not None else ""
    return f"{model_path.parent.name} ({best_model}){suffix}"


def model_comparison_rows(model_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in model_paths:
        metrics = load_metrics(path)
        bundle_hint = path.parent.name
        test_metrics = metrics.get("test_metrics", {})
        rows.append(
            {
                "artifact": bundle_hint,
                "best_model": metrics.get("best_model", "unknown"),
                "split": metrics.get("split_strategy", "unknown"),
                "feature_mode": metrics.get("feature_mode", "window"),
                "trial_top1": metric_value(metrics, "trial_top1_accuracy"),
                "trial_top5": metric_value(metrics, "trial_top5_accuracy"),
                "macro_f1": test_metrics.get("trial_macro_f1", test_metrics.get("macro_f1")),
                "window_top1": test_metrics.get("top1_accuracy"),
                "classes": len(metrics.get("class_names", [])) or None,
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def default_model_index(model_paths: list[Path]) -> int:
    if not model_paths:
        return 0
    scores = [metric_value(load_metrics(path), "trial_top1_accuracy") or -1.0 for path in model_paths]
    return int(np.argmax(scores))


def load_input_dataframe(uploaded_file, selected_sample: str) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    if selected_sample:
        path = Path(selected_sample)
        return pd.read_csv(path), str(path)
    return None, None


def display_sample_name(value: str, sample_root: Path) -> str:
    if value == "":
        return "No sample"

    path = Path(value)
    try:
        return str(path.relative_to(sample_root))
    except ValueError:
        return path.name


def load_metrics(model_path: Path) -> dict:
    metrics_path = model_path.parent / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def plot_theme() -> dict[str, str]:
    if st.get_option("theme.base") == "dark":
        return {
            "face": "#161b22",
            "text": "#edf6f7",
            "grid": "#34454c",
            "spine": "#4d6168",
        }
    return {
        "face": "#ffffff",
        "text": "#183c40",
        "grid": "#d7e1e4",
        "spine": "#9fb2b7",
    }


def apply_plot_theme(fig, ax) -> None:
    colors = plot_theme()
    fig.patch.set_facecolor(colors["face"])
    ax.set_facecolor(colors["face"])
    ax.title.set_color(colors["text"])
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.tick_params(colors=colors["text"])
    ax.grid(color=colors["grid"], alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color(colors["spine"])


def plot_sensor_curves(df: pd.DataFrame, time_col: str | None, sensor_cols: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    apply_plot_theme(fig, ax)
    x = df[time_col] if time_col and time_col in df.columns else np.arange(len(df))
    x_label = time_col if time_col and time_col in df.columns else "sample_index"

    for col in sensor_cols:
        ax.plot(x, df[col], label=col, linewidth=1.35, alpha=0.82)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sensor response")
    apply_plot_theme(fig, ax)
    legend = ax.legend(loc="upper right", fontsize=8, ncols=2, frameon=True)
    legend.get_frame().set_facecolor(plot_theme()["face"])
    legend.get_frame().set_edgecolor(plot_theme()["spine"])
    for text in legend.get_texts():
        text.set_color(plot_theme()["text"])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def confidence_label(confidence: float) -> tuple[str, str]:
    if confidence >= 0.70:
        return "High", "The model is clearly favoring one smell class."
    if confidence >= 0.40:
        return "Medium", "Use the top-5 list and curve shape before trusting the top class."
    return "Low", "The signal is ambiguous for the current model; treat this as exploratory."


def plot_top_predictions(top5_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    apply_plot_theme(fig, ax)
    ordered = top5_df.iloc[::-1]
    colors = ["#8aa6a9"] * len(ordered)
    if colors:
        colors[-1] = "#2f6f73"
    ax.barh(ordered["class"], ordered["probability"], color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean window probability")
    apply_plot_theme(fig, ax)
    ax.grid(axis="x", color=plot_theme()["grid"], alpha=0.35)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_model_metrics(metrics: dict) -> None:
    test_metrics = metrics.get("test_metrics", {})
    if not test_metrics:
        st.caption("No metrics file found beside the model artifact.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Trial Top-1", f"{test_metrics.get('trial_top1_accuracy', 0.0) * 100:.1f}%")
    m2.metric("Trial Top-5", f"{test_metrics.get('trial_top5_accuracy', 0.0) * 100:.1f}%")
    m3.metric("Macro F1", f"{test_metrics.get('trial_macro_f1', 0.0):.3f}")
    st.caption(f"Best trained model: {metrics.get('best_model', 'unknown')} | Split: {metrics.get('split_strategy', 'unknown')}")


def render_usecase_overview() -> None:
    st.subheader("Use Case")
    st.write(
        "This project is a research demo for recognizing food or ingredient smell classes from gas sensor time-series CSV files. "
        "A sensor trial is converted into cleaned time windows, engineered signal features, and class probabilities."
    )
    st.markdown(
        """
        <div class="hero-grid">
          <div class="hero-step"><strong>1. Sensor CSV</strong><span>Upload or select a SmellNet gas sensor recording.</span></div>
          <div class="hero-step"><strong>2. Signal cleanup</strong><span>Fill gaps, trim warm-up drift, resample, and normalize.</span></div>
          <div class="hero-step"><strong>3. Model choice</strong><span>Use the best saved ensemble or compare older baselines.</span></div>
          <div class="hero-step"><strong>4. Prediction</strong><span>See the predicted smell, confidence, top-5 alternatives, and curves.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_value_cards() -> None:
    st.subheader("What Users Can Do")
    st.markdown(
        """
        <div class="hero-grid">
          <div class="usecase-card"><strong>Explore sensor behavior</strong><p>Plot raw and preprocessed curves to see how each gas sensor responds over time.</p></div>
          <div class="usecase-card"><strong>Classify a trial</strong><p>Run a trained model and inspect the top predicted smell class plus alternatives.</p></div>
          <div class="usecase-card"><strong>Compare models</strong><p>Select saved artifacts and compare real held-out metrics before trusting a model.</p></div>
          <div class="usecase-card"><strong>Understand limits</strong><p>Low-confidence predictions and schema errors are surfaced instead of hidden.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_comparison(comparison_df: pd.DataFrame) -> None:
    if comparison_df.empty:
        st.info("No saved model artifacts were found under `models/*/model.joblib`.")
        return

    display = comparison_df.copy()
    for col in ["trial_top1", "trial_top5", "macro_f1", "window_top1"]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%" if col != "macro_f1" else f"{value:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_training_options() -> None:
    st.subheader("Model Changing Options")
    st.write(
        "Model changes happen in training, then the app selects the saved `model.joblib` artifact. "
        "The strongest current path is the trial-max windowed baseline because it optimizes the same trial-level aggregation used by the demo."
    )
    options = pd.DataFrame(
        [
            {"option": "Full-sequence baseline", "command flag": "--window-size 0", "tradeoff": "Fast and simple, but weaker because each CSV becomes one sample."},
            {"option": "Windowed baseline", "command flag": "--window-size 100 --window-stride 25", "tradeoff": "More training samples and better trial aggregation; current default."},
            {"option": "Contextual window baseline", "command flag": "--use-context-features", "tradeoff": "Adds whole-trial shape and window position to each window, improving the model's view of time-series structure."},
            {"option": "Window aggregation", "command flag": "--trial-aggregation max", "tradeoff": "Uses the strongest window evidence per class at inference; current best trial metrics are 64.0% top-1 and 92.0% top-5."},
            {"option": "Include SVM", "command flag": "--include-svm", "tradeoff": "Adds an RBF SVM and a larger soft-voting ensemble; slower but can improve accuracy."},
            {"option": "Tune preprocessing", "command flag": "--target-points, --warmup-ratio", "tradeoff": "Changes signal shape seen by every model; rerun evaluation after changing."},
            {"option": "Optional sequence model", "command flag": "src/models/train_timeseries.py", "tradeoff": "Experimental PyTorch path; classical baseline is currently stronger."},
        ]
    )
    st.dataframe(options, use_container_width=True, hide_index=True)
    st.code(
        "uv run python src/models/train_baseline.py --data-root data/raw/SmellNet "
        "--output-dir models/baseline_windowed_trialmax --window-size 100 --window-stride 25 --include-svm --trial-aggregation max",
        language="powershell",
    )


def render_schema_panel(df: pd.DataFrame, trained_sensor_cols: list[str], detected_sensor_cols: list[str], missing_cols: list[str]) -> None:
    checks = pd.DataFrame(
        {
            "sensor": trained_sensor_cols,
            "present": [col in df.columns for col in trained_sensor_cols],
            "numeric": [pd.api.types.is_numeric_dtype(df[col]) if col in df.columns else False for col in trained_sensor_cols],
            "missing_values": [int(df[col].isna().sum()) if col in df.columns else None for col in trained_sensor_cols],
        }
    )

    st.markdown(
        f"""
        <span class="status-chip">{len(detected_sensor_cols)} numeric sensors detected</span>
        <span class="status-chip">{len(missing_cols)} missing expected sensors</span>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)


def render_method_notes() -> None:
    st.subheader("What Makes This Project Specific")
    st.write(
        "This is not just a generic CSV classifier. The pipeline is shaped around gas sensor time-series behavior: "
        "warm-up trimming, resampling, per-trial normalization, time windows, trial-level probability aggregation, "
        "and features that capture sensor drift, response shape, frequency content, and cross-sensor relationships."
    )

    st.subheader("Where Accuracy Comes From")
    points = pd.DataFrame(
        [
            {"area": "Preprocessing", "implementation": "Fill missing values, trim sensor warm-up, resample every trial to a common timeline."},
            {"area": "Feature extraction", "implementation": "Use statistical, derivative, FFT band, peak timing, and cross-sensor correlation features."},
            {"area": "Model selection", "implementation": "Compare logistic regression, forests, boosting, SVM, and soft-voting ensembles."},
            {"area": "Evaluation", "implementation": "Use the SmellNet folder split and report both window-level and trial-level metrics."},
            {"area": "App behavior", "implementation": "Validate schema, plot sensor curves, show top-5 predictions, and label low-confidence outputs."},
        ]
    )
    st.dataframe(points, use_container_width=True, hide_index=True)


def render_research_workflow() -> None:
    st.subheader("Research Workflow")
    workflow = pd.DataFrame(
        [
            {"step": "1. Trial CSV", "research question": "What does one smell exposure look like across the sensor array?", "implementation": "Treat one CSV file as one gas-sensor trial."},
            {"step": "2. Signal cleanup", "research question": "How do we reduce sensor warm-up effects and row-count differences?", "implementation": "Fill gaps, trim warm-up rows, resample, and normalize each sensor."},
            {"step": "3. Windowing", "research question": "Which parts of the response curve carry useful odor information?", "implementation": "Split each trial into fixed time windows."},
            {"step": "4. Feature extraction", "research question": "How can classical ML see signal shape without raw deep learning?", "implementation": "Extract drift, slope, energy, peak timing, frequency, and cross-sensor features."},
            {"step": "5. Model comparison", "research question": "Which baseline works best before adding complex models?", "implementation": "Compare linear, forest, boosting, SVM, and soft-voting classifiers."},
            {"step": "6. Trial prediction", "research question": "What smell class does the whole uploaded trial most resemble?", "implementation": "Aggregate window probabilities and show top-5 alternatives."},
        ]
    )
    st.dataframe(workflow, use_container_width=True, hide_index=True)


def render_project_guide() -> None:
    st.markdown(
        """
        <div class="app-hero">
          <h1>Project Guide</h1>
          <p>A quick orientation for what this machine olfaction project does, why it is useful, and where to find each result in the app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_usecase_overview()
    st.divider()
    render_research_workflow()
    st.divider()

    st.subheader("What Makes This More Than Another CSV Classifier")
    st.write(
        "A normal CSV classifier treats each row or file as a flat table. This project is different because the input is a gas sensor response over time. "
        "The useful information is in how sensor curves rise, drift, stabilize, interact, and differ across a trial."
    )
    strengths = pd.DataFrame(
        [
            {"strong point": "Time-series aware preprocessing", "why it matters": "The pipeline trims warm-up drift, resamples trials to a shared timeline, fills gaps, and normalizes per sensor before modeling."},
            {"strong point": "Sensor-shape features", "why it matters": "Features capture slope, area under curve, peak timing, derivatives, FFT power, and response energy instead of only raw CSV values."},
            {"strong point": "Cross-sensor relationships", "why it matters": "The model uses correlations and differences between sensor channels, which is closer to how electronic noses distinguish odors."},
            {"strong point": "Window plus trial reasoning", "why it matters": "The improved training mode can combine local window behavior with whole-trial context and window position."},
            {"strong point": "Leakage-aware evaluation", "why it matters": "Evaluation uses SmellNet's train/test folder split when available and reports trial-level metrics by aggregating windows per source CSV."},
            {"strong point": "Transparent research demo", "why it matters": "The app shows confidence, top-5 alternatives, schema checks, raw curves, preprocessed curves, and model comparison instead of hiding uncertainty."},
        ]
    )
    st.dataframe(strengths, use_container_width=True, hide_index=True)

    st.subheader("Why This Is Helpful")
    st.write(
        "Gas sensor arrays produce time-series signals that can be difficult to interpret by eye. "
        "This project turns those raw curves into a repeatable smell-recognition workflow: clean the signal, extract features, "
        "compare trained classifiers, and show predictions with confidence instead of only returning a class label."
    )
    st.markdown(
        """
        <div class="hero-grid">
          <div class="usecase-card"><strong>Research exploration</strong><p>Inspect how different smell samples change sensor response curves over time.</p></div>
          <div class="usecase-card"><strong>Baseline ML system</strong><p>Use a classical model pipeline before moving to more complex sequence models.</p></div>
          <div class="usecase-card"><strong>Transparent predictions</strong><p>Review confidence and top-5 alternatives when a signal is ambiguous.</p></div>
          <div class="usecase-card"><strong>Reusable workflow</strong><p>Retrain saved artifacts and compare real metrics in the app.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("How To Navigate")
    navigation = pd.DataFrame(
        [
            {"where": "Sidebar > Navigation", "what you do there": "Switch between this guide and the prediction demo."},
            {"where": "Sidebar > Model artifact", "what you do there": "Select which saved `model.joblib` pipeline should make predictions."},
            {"where": "Sidebar > Sample CSV", "what you do there": "Pick a demo SmellNet CSV, or upload your own CSV in the main page."},
            {"where": "Prediction Demo > Overview", "what you do there": "Confirm the use case, selected model health, and whether the CSV schema is valid."},
            {"where": "Prediction Demo > Prediction Results", "what you do there": "See the predicted smell, confidence score, top-5 classes, and sensor curves."},
            {"where": "Prediction Demo > CSV Details", "what you do there": "Inspect detected sensors, missing columns, and raw CSV rows."},
            {"where": "Prediction Demo > Models", "what you do there": "Compare saved model artifacts and see the model-changing training options."},
            {"where": "Prediction Demo > Method", "what you do there": "Understand preprocessing, feature extraction, evaluation, and limitations."},
        ]
    )
    st.dataframe(navigation, use_container_width=True, hide_index=True)

    st.subheader("Important Limitations")
    st.warning(
        "This is a research demo for SmellNet-style gas sensor data. It should not be used as a food-safety, allergen, "
        "quality-control, or hazardous-gas detection system."
    )


def main() -> None:
    st.set_page_config(page_title="Machine Olfaction Demo", layout="wide")
    inject_style()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Project Guide", "Prediction Demo"], label_visibility="collapsed")
        st.divider()

    if page == "Project Guide":
        render_project_guide()
        return

    st.markdown(
        """
        <div class="app-hero">
          <h1>Machine Olfaction Research Demo</h1>
          <p>Upload a gas sensor time-series CSV, select a trained smell classifier, and inspect the predicted smell class with confidence and top-5 alternatives.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_paths = discover_model_artifacts()
    if not model_paths:
        st.error("No model artifacts found under `models/*/model.joblib`. Train a model before running the demo.")
        return
    comparison_df = model_comparison_rows(model_paths)

    with st.sidebar:
        st.header("Run Setup")
        selected_model = st.selectbox(
            "Model artifact",
            model_paths,
            index=default_model_index(model_paths),
            format_func=model_display_name,
            help="Choose which saved pipeline artifact to use for prediction.",
        )
        model_path = Path(selected_model)
        sample_root = Path(st.text_input("Sample CSV folder", value="data/samples"))
        samples = sample_csvs(sample_root)
        sample_options = [""] + [str(path) for path in samples]
        default_sample_index = 1 if samples else 0
        selected_sample = st.selectbox(
            "Sample CSV",
            sample_options,
            index=default_sample_index,
            help="A real SmellNet sample is selected by default so the app works without your own CSV.",
            format_func=lambda value: display_sample_name(value, sample_root),
        )
        if samples:
            st.caption(f"{len(samples)} real demo CSVs found in {sample_root}.")
        else:
            st.caption("No demo CSVs found. Run the sample creation command from README.")
        auto_predict = st.checkbox("Analyze selected CSV automatically", value=True)
        aggregation = st.selectbox(
            "Window aggregation",
            ["max", "mean", "median"],
            index=0,
            help="How to combine probabilities from multiple windows in one CSV. Max improves current best held-out top-5 accuracy; mean is the original behavior.",
        )
        st.divider()
        sidebar_metrics = load_metrics(model_path)
        st.caption(f"Selected model: `{model_path.parent.name}`")
        render_model_metrics(sidebar_metrics)
        st.divider()
        st.warning("Research use only. This is not a food-safety, allergen, or hazardous-gas detector.")

    uploaded = st.file_uploader("Upload gas sensor CSV", type=["csv"])
    df, source_name = load_input_dataframe(uploaded, selected_sample)

    if not model_path.exists():
        st.error(f"Model artifact not found: {model_path}")
        return

    try:
        bundle = cached_load_pipeline(str(model_path))
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    metrics = load_metrics(model_path)
    trained_sensor_cols = list(bundle["sensor_columns"])

    overview_tab, predict_tab, details_tab, models_tab, method_tab = st.tabs(["Overview", "Prediction Results", "CSV Details", "Models", "Method"])

    with overview_tab:
        render_usecase_overview()
        render_user_value_cards()
        st.divider()
        left, right = st.columns([1.15, 1.0])
        with left:
            st.subheader("Selected Model Health")
            render_model_metrics(metrics)
        with right:
            st.subheader("Input Status")
            if df is None:
                st.info("Upload a CSV or use the sample selector in the sidebar. Real demo samples can be created from SmellNet with `uv run python src/data/make_samples.py --data-root data/raw/SmellNet --output-dir data/samples`.")
            else:
                time_col = identify_time_column(df)
                detected_sensor_cols = infer_sensor_columns(df, time_column=time_col)
                _, _, missing_cols = validate_uploaded_schema(df, trained_sensor_cols)
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{len(df):,}")
                c2.metric("Columns", f"{len(df.columns):,}")
                c3.metric("Sensors", f"{len(detected_sensor_cols):,}")
                st.caption(f"Source: {source_name}")
                if missing_cols:
                    st.error(f"Missing expected sensors: {missing_cols}")
                else:
                    st.success("The CSV matches the trained sensor schema.")

    if df is None:
        return

    time_col = identify_time_column(df)
    detected_sensor_cols = infer_sensor_columns(df, time_column=time_col)
    _, _, missing_cols = validate_uploaded_schema(df, trained_sensor_cols)

    with predict_tab:
        if missing_cols:
            st.error(f"Prediction blocked because these expected sensors are missing: {missing_cols}")
            return

        st.subheader("Prediction Result")
        run_now = auto_predict or st.button("Analyze smell signal", type="primary")
        if run_now:
            try:
                result = predict_dataframe(df, bundle, aggregation=aggregation)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

            classes = result["class_names"]
            proba = result["probabilities"]
            top_idx = result["top_indices"][: min(5, len(classes))]
            top5_df = pd.DataFrame({"class": classes[top_idx], "probability": proba[top_idx]})
            confidence_name, confidence_text = confidence_label(result["confidence"])

            st.markdown(
                f"""
                <div class="result-panel">
                  <div class="label">Predicted smell class</div>
                  <div class="value">{result["predicted_class"]}</div>
                  <div class="note">{confidence_name} confidence: {result["confidence"] * 100:.1f}% across {result["n_windows"]} analyzed window(s), using {result["aggregation"]} aggregation.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted smell", result["predicted_class"])
            r2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            r3.metric("Confidence band", confidence_name)
            r4.metric("Windows", result["n_windows"])
            st.info(confidence_text)

            left, right = st.columns([1.0, 1.0])
            with left:
                st.subheader("Top-5 Predictions")
                plot_top_predictions(top5_df)
            with right:
                display_df = top5_df.assign(probability=lambda x: (x["probability"] * 100).round(2))
                st.subheader("Probability Table")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            with st.expander("Preprocessed signal used by the model", expanded=False):
                processed_df = result["processed_df"]
                st.caption(f"Shape after preprocessing: {processed_df.shape}")
                st.dataframe(processed_df.head(30), use_container_width=True)
                plot_sensor_curves(processed_df, "time", trained_sensor_cols, title="Preprocessed model input")
        else:
            st.info("Click `Analyze smell signal` to run the selected model on this CSV.")

        st.subheader("Sensor Curves")
        plot_sensor_curves(df, time_col, trained_sensor_cols, title="Raw sensor response")

    with details_tab:
        st.subheader("Schema and CSV Preview")
        render_schema_panel(df, trained_sensor_cols, detected_sensor_cols, missing_cols)
        st.dataframe(df.head(40), use_container_width=True)

    with models_tab:
        st.subheader("Saved Model Comparison")
        render_model_comparison(comparison_df)
        render_training_options()
        if metrics.get("val_results"):
            st.subheader("Validation Candidates Inside Selected Artifact")
            val_df = pd.DataFrame(metrics["val_results"]).T.reset_index(names="candidate_model")
            st.dataframe(val_df, use_container_width=True, hide_index=True)

    with method_tab:
        render_method_notes()


if __name__ == "__main__":
    main()
