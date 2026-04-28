from __future__ import annotations

import json
import sys
from pathlib import Path

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


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1320px; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            border: 1px solid #d9e2e3;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #fbfdfd;
        }
        div[data-testid="stAlert"] { border-radius: 8px; }
        .app-hero {
            border-bottom: 1px solid #d9e2e3;
            padding-bottom: 0.9rem;
            margin-bottom: 1rem;
        }
        .app-hero p { color: #496569; margin: 0.2rem 0 0 0; }
        .status-chip {
            display: inline-block;
            border: 1px solid #c9d7d9;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            color: #25494d;
            background: #f4faf9;
            font-size: 0.88rem;
            margin-right: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sample_csvs(sample_root: Path) -> list[Path]:
    if not sample_root.exists():
        return []
    return [path for path in sorted(sample_root.rglob("*.csv")) if ".cache" not in path.parts]


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


def plot_sensor_curves(df: pd.DataFrame, time_col: str | None, sensor_cols: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    x = df[time_col] if time_col and time_col in df.columns else np.arange(len(df))
    x_label = time_col if time_col and time_col in df.columns else "sample_index"

    for col in sensor_cols:
        ax.plot(x, df[col], label=col, linewidth=1.35, alpha=0.82)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sensor response")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncols=2, frameon=True)
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
    ordered = top5_df.iloc[::-1]
    colors = ["#8aa6a9"] * len(ordered)
    if colors:
        colors[-1] = "#2f6f73"
    ax.barh(ordered["class"], ordered["probability"], color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean window probability")
    ax.grid(axis="x", alpha=0.2)
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


def main() -> None:
    st.set_page_config(page_title="Machine Olfaction Demo", layout="wide")
    inject_style()

    st.markdown(
        """
        <div class="app-hero">
          <h1>Machine Olfaction Research Demo</h1>
          <p>Smell recognition from gas sensor time-series CSV files, with confidence and top-5 alternatives.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Run Setup")
        model_path = Path(st.text_input("Model artifact", value="models/baseline_windowed/model.joblib"))
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
        st.divider()
        st.warning("Research use only. This is not a food-safety, allergen, or hazardous-gas detector.")

    uploaded = st.file_uploader("Upload gas sensor CSV", type=["csv"])
    df, source_name = load_input_dataframe(uploaded, selected_sample)

    if not model_path.exists():
        st.error(f"Model artifact not found: {model_path}")
        return

    try:
        bundle = load_pipeline(model_path)
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    metrics = load_metrics(model_path)
    trained_sensor_cols = list(bundle["sensor_columns"])

    overview_tab, predict_tab, details_tab, method_tab = st.tabs(["Overview", "Prediction", "Details", "Method"])

    with overview_tab:
        left, right = st.columns([1.15, 1.0])
        with left:
            st.subheader("Model Health")
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

        top_area, chart_area = st.columns([0.82, 1.18])
        with top_area:
            st.subheader("Run Inference")
            run_now = st.button("Analyze smell signal", type="primary", use_container_width=True)
            st.caption("The model averages predictions across time windows from the uploaded trial.")
        with chart_area:
            st.subheader("Sensor Curves")
            plot_sensor_curves(df, time_col, trained_sensor_cols, title="Raw sensor response")

        if run_now:
            try:
                result = predict_dataframe(df, bundle)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

            classes = result["class_names"]
            proba = result["probabilities"]
            top_idx = result["top_indices"][: min(5, len(classes))]
            top5_df = pd.DataFrame({"class": classes[top_idx], "probability": proba[top_idx]})
            confidence_name, confidence_text = confidence_label(result["confidence"])

            st.subheader("Result")
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

    with details_tab:
        st.subheader("Schema and CSV Preview")
        render_schema_panel(df, trained_sensor_cols, detected_sensor_cols, missing_cols)
        st.dataframe(df.head(40), use_container_width=True)

    with method_tab:
        render_method_notes()


if __name__ == "__main__":
    main()
