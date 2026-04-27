from __future__ import annotations

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


def plot_sensor_curves(df: pd.DataFrame, time_col: str | None, sensor_cols: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))

    x = df[time_col] if time_col and time_col in df.columns else np.arange(len(df))
    x_label = time_col if time_col and time_col in df.columns else "sample_index"

    for col in sensor_cols:
        ax.plot(x, df[col], label=col, linewidth=1.25, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sensor value")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    st.pyplot(fig)


def confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "High confidence"
    if confidence >= 0.40:
        return "Medium confidence"
    return "Low confidence"


def plot_top_predictions(top5_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ordered = top5_df.iloc[::-1]
    ax.barh(ordered["class"], ordered["probability"], color="#2f6f73")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted probability")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Machine Olfaction Demo", layout="wide")

    st.title("Machine Olfaction Research Demo")
    st.caption("Smell classification from gas sensor time-series CSV files.")

    with st.sidebar:
        st.header("Model")
        model_path = Path(st.text_input("Model artifact", value="models/baseline_windowed/model.joblib"))
        sample_root = Path(st.text_input("Sample CSV folder", value="data/samples"))
        samples = sample_csvs(sample_root)
        sample_options = [""] + [str(path) for path in samples]
        selected_sample = st.selectbox("Try a sample CSV", sample_options, format_func=lambda value: "None" if value == "" else Path(value).name)
        st.divider()
        st.write("This demo is for research exploration only. It is not a food-safety, allergen, or hazardous-gas detector.")

    uploaded = st.file_uploader("Upload a gas sensor CSV", type=["csv"])
    df, source_name = load_input_dataframe(uploaded, selected_sample)

    if df is None:
        st.info("Upload a CSV or choose a sample from the sidebar to begin.")
        return

    if not model_path.exists():
        st.error(f"Model artifact not found: {model_path}")
        return

    try:
        bundle = load_pipeline(model_path)
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    trained_sensor_cols = list(bundle["sensor_columns"])
    time_col = identify_time_column(df)
    detected_sensor_cols = infer_sensor_columns(df, time_column=time_col)
    _, _, missing_cols = validate_uploaded_schema(df, trained_sensor_cols)

    st.subheader("Input")
    left, right, third = st.columns(3)
    left.metric("Rows", f"{len(df):,}")
    right.metric("Columns", f"{len(df.columns):,}")
    third.metric("Detected sensors", f"{len(detected_sensor_cols):,}")
    st.caption(f"Source: {source_name}")

    with st.expander("CSV preview", expanded=False):
        st.dataframe(df.head(30), use_container_width=True)

    st.subheader("Schema Check")
    if missing_cols:
        st.error(f"Missing expected sensor columns: {missing_cols}")
        st.stop()

    st.success("CSV contains all sensor columns expected by the saved model.")
    schema_df = pd.DataFrame(
        {
            "expected_sensor": trained_sensor_cols,
            "present": [col in df.columns for col in trained_sensor_cols],
        }
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.subheader("Sensor Curves")
    plot_sensor_curves(df, time_col, trained_sensor_cols, title="Raw uploaded signal")

    if st.button("Run prediction", type="primary"):
        try:
            result = predict_dataframe(df, bundle)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        classes = result["class_names"]
        proba = result["probabilities"]
        top_idx = result["top_indices"][: min(5, len(classes))]
        top5_df = pd.DataFrame(
            {
                "class": classes[top_idx],
                "probability": proba[top_idx],
            }
        )

        st.subheader("Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted smell", result["predicted_class"])
        c2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        c3.metric("Windows analyzed", result["n_windows"])
        st.info(confidence_label(result["confidence"]))

        st.subheader("Top-5 Predictions")
        plot_top_predictions(top5_df)
        st.dataframe(
            top5_df.assign(probability=lambda x: (x["probability"] * 100).round(2)),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Preprocessed signal used by the model", expanded=False):
            processed_df = result["processed_df"]
            st.write(f"Shape after preprocessing: {processed_df.shape}")
            st.dataframe(processed_df.head(30), use_container_width=True)
            plot_sensor_curves(processed_df, "time", trained_sensor_cols, title="Preprocessed model input")


if __name__ == "__main__":
    main()
