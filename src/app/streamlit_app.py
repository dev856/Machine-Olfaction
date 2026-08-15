from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.augmentation import add_baseline_drift, add_gaussian_jitter, apply_magnitude_scaling, apply_sensor_dropout
from src.data.preprocess import identify_time_column, infer_sensor_columns
from src.data.semantic_profiles import get_knowledge_base, get_semantic_profile
from src.evaluation.sensor_importance import compute_sensor_importance_scores
from src.models.mixture_predict import deconvolve_mixture_dataframe
from src.models.pipeline_io import load_pipeline
from src.models.predict import predict_dataframe, validate_uploaded_schema


METRIC_EXPLANATIONS: dict[str, str] = {
    "Trial Top-1": "Percent of full CSV trials where the model's first choice is the correct smell class.",
    "Trial Top-5": "Percent of full CSV trials where the correct smell class appears anywhere in the five most likely predictions.",
    "Window Top-1": "Percent of individual time windows classified correctly before windows are combined into one CSV-level prediction.",
    "Macro F1": "Average F1 score across smell classes, treating each class equally.",
    "Weighted F1": "Average F1 score weighted by class support.",
}


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
        .chem-chip {
            display: inline-block;
            border: 1px solid var(--mo-border);
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-size: 0.85rem;
            background: var(--mo-surface-soft);
            color: var(--mo-text);
            margin: 0.15rem 0.25rem 0.15rem 0;
            font-weight: 500;
        }
        .sensory-chip {
            display: inline-block;
            border: 1px solid rgba(235, 152, 78, 0.4);
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-size: 0.85rem;
            background: var(--mo-accent-warm);
            color: var(--mo-text);
            margin: 0.15rem 0.25rem 0.15rem 0;
        }
        .category-badge {
            display: inline-block;
            border-radius: 6px;
            padding: 0.25rem 0.75rem;
            font-size: 0.85rem;
            font-weight: 600;
            background: var(--mo-accent);
            color: #ffffff;
            margin-bottom: 0.4rem;
        }
        .semantic-card {
            border: 1px solid var(--mo-border);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            background: var(--mo-surface);
            box-shadow: 0 2px 6px var(--mo-shadow);
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .kb-card {
            border: 1px solid var(--mo-border);
            border-radius: 8px;
            padding: 0.85rem;
            background: var(--mo-surface);
            box-shadow: 0 1px 3px var(--mo-shadow);
            margin-bottom: 0.75rem;
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


def project_relative_path(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path_obj.as_posix()


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
                "artifact_file": project_relative_path(path),
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


def display_source_name(value: str | None, sample_root: Path) -> str:
    if not value:
        return "No input selected"
    path = Path(value)
    if not path.exists():
        return path.name
    try:
        return str(path.relative_to(sample_root))
    except ValueError:
        return project_relative_path(path)


def load_metrics(model_path: Path) -> dict:
    metrics_path = model_path.parent / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_plotly_template() -> str:
    """Get Plotly template based on Streamlit theme."""
    if st.get_option("theme.base") == "dark":
        return "plotly_dark"
    return "plotly_white"


def plot_sensor_curves_interactive(df: pd.DataFrame, time_col: str | None, sensor_cols: list[str], title: str) -> None:
    """Create interactive Plotly chart for sensor curves with hover tooltips."""
    x = df[time_col] if time_col and time_col in df.columns else np.arange(len(df))
    x_label = time_col if time_col and time_col in df.columns else "Sample Index"
    
    fig = go.Figure()
    
    # Color palette for sensors
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
    
    for idx, col in enumerate(sensor_cols):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                name=col,
                mode='lines',
                line=dict(width=2, color=colors[idx % len(colors)]),
                hovertemplate=f'<b>{col}</b><br>Time: %{{x:.2f}}<br>Response: %{{y:.4f}}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight="bold")),
        xaxis_title=x_label,
        yaxis_title="Sensor Response",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.5)"
        ),
        template=get_plotly_template(),
        height=450,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    
    st.plotly_chart(fig, use_container_width=True)


def confidence_label(confidence: float) -> tuple[str, str]:
    if confidence >= 0.70:
        return "High", "The model is clearly favoring one smell class."
    if confidence >= 0.40:
        return "Medium", "Use the top-5 list and curve shape before trusting the top class."
    return "Low", "The signal is ambiguous for the current model; treat this as exploratory."


def plot_top_predictions_interactive(top5_df: pd.DataFrame) -> None:
    """Create interactive horizontal bar chart for top predictions."""
    ordered = top5_df.iloc[::-1]
    
    # Create colors - highlight the top prediction
    colors = ["#8aa6a9"] * len(ordered)
    if colors:
        colors[-1] = "#2f6f73"  # Highlight top prediction
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            y=ordered["class"],
            x=ordered["probability"],
            orientation='h',
            marker_color=colors,
            hovertemplate='<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title=dict(text="Top 5 Predictions", font=dict(size=16, weight="bold")),
        xaxis_title="Mean Window Probability",
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=False),
        template=get_plotly_template(),
        height=350,
        margin=dict(l=60, r=30, t=50, b=40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_confidence_gauge(confidence: float, predicted_class: str) -> None:
    """Create a gauge chart showing prediction confidence."""
    # Determine color based on confidence level
    if confidence >= 0.70:
        gauge_color = "#27ae60"  # Green - High confidence
        status_text = "High Confidence"
    elif confidence >= 0.40:
        gauge_color = "#f39c12"  # Orange - Medium confidence
        status_text = "Medium Confidence"
    else:
        gauge_color = "#e74c3c"  # Red - Low confidence
        status_text = "Low Confidence"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Prediction Confidence<br><span style='font-size: 0.8em; color: gray'>{predicted_class}</span>", 
               'font': {'size': 14}},
        delta={'reference': 50, 'increasing': {'color': gauge_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': gauge_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(231, 76, 60, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(243, 156, 18, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(39, 174, 96, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        template=get_plotly_template(),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plain_feature_name(feature_name: str) -> str:
    if "__x__" in feature_name:
        parts = feature_name.split("__")
        if len(parts) >= 5:
            return f"relationship between {parts[0]} and {parts[2]} ({parts[-1].replace('_', ' ')})"
    if "__" in feature_name:
        sensor, measure = feature_name.split("__", 1)
        return f"{sensor} sensor {measure.replace('_', ' ')}"
    return feature_name.replace("_", " ")


def linear_explanation_rows(bundle: dict[str, Any], result: dict[str, Any], limit: int = 8) -> pd.DataFrame:
    model = bundle["model"]
    feature_names = list(bundle.get("feature_names") or [])
    features = result.get("feature_matrix")
    if features is None or not feature_names:
        return pd.DataFrame()

    transformed = np.asarray(features)
    active_names = feature_names.copy()
    classifier = model

    if hasattr(model, "steps"):
        for _, step in model.steps[:-1]:
            if hasattr(step, "transform"):
                transformed = step.transform(transformed)
            if hasattr(step, "get_support"):
                support = step.get_support()
                active_names = [name for name, keep in zip(active_names, support) if keep]
        classifier = model.steps[-1][1]

    if not hasattr(classifier, "coef_"):
        return pd.DataFrame()

    predicted_label = result["predicted_class"]
    class_names = list(bundle["label_encoder"].classes_)
    predicted_index = class_names.index(predicted_label)
    classifier_classes = list(getattr(classifier, "classes_", range(len(class_names))))
    class_position = classifier_classes.index(predicted_index) if predicted_index in classifier_classes else predicted_index

    mean_feature_values = np.asarray(transformed).mean(axis=0)
    coefficients = np.asarray(classifier.coef_)[class_position]
    contributions = mean_feature_values * coefficients
    order = np.argsort(contributions)[::-1][:limit]

    rows = []
    for idx in order:
        rows.append(
            {
                "sensor signal clue": plain_feature_name(active_names[idx]),
                "effect": "pushed prediction higher",
                "strength": round(float(contributions[idx]), 3),
            }
        )
    return pd.DataFrame(rows)


def render_semantic_profile_card(smell_class: str) -> None:
    """Render a clean card with botanical category, chemical volatiles, and sensory notes."""
    profile = get_semantic_profile(smell_class)
    volatiles_html = "".join([f'<span class="chem-chip">🧪 {v}</span>' for v in profile.volatiles])
    sensory_html = "".join([f'<span class="sensory-chip">👃 {s}</span>' for s in profile.sensory_notes])

    st.markdown(
        f"""
        <div class="semantic-card">
          <div class="category-badge">{profile.category}</div>
          <div style="font-size: 1.15rem; font-weight: 600; color: var(--mo-text); margin-bottom: 0.35rem;">
            Aroma & Chemical Profile: {smell_class.replace('_', ' ').title()}
          </div>
          <p style="color: var(--mo-muted); font-size: 0.94rem; margin-bottom: 0.6rem;">{profile.full_description}</p>
          <div style="margin-bottom: 0.45rem;">
            <strong style="font-size: 0.9rem; color: var(--mo-text); margin-right: 0.4rem;">Key Chemical Volatiles (VOCs):</strong>
            {volatiles_html}
          </div>
          <div>
            <strong style="font-size: 0.9rem; color: var(--mo-text); margin-right: 0.4rem;">Sensory Aroma Notes:</strong>
            {sensory_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_drift_simulator_tab(df: pd.DataFrame | None, time_col: str | None, sensor_cols: list[str], bundle: dict[str, Any]) -> None:
    """Interactive playground to inject thermal drift, noise, and sensor failures to test model resilience."""
    st.subheader("Hardware Drift & Stress Testing Playground")
    st.write(
        "Real-world electronic noses face sensor aging, temperature drift, and channel disconnection. "
        "Inject artificial distortions below to test how robust the trained model remains under physical stress."
    )
    if df is None:
        st.info("Upload or select a sensor CSV from the sidebar to activate the simulator.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        drift_slope = st.slider("Thermal Baseline Drift", min_value=-0.5, max_value=0.5, value=0.15, step=0.05)
        drift_type = st.selectbox("Drift Type", ["linear", "exponential", "sinusoidal"])
    with c2:
        noise_sigma = st.slider("Gaussian Electrical Noise", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
        scale_factor = st.slider("Concentration / Scale Factor", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    with c3:
        drop_sensor = st.selectbox("Simulate Dead Sensor (Dropout)", ["None"] + sensor_cols)

    raw_values = df[sensor_cols].to_numpy(dtype=float).copy()
    stressed_values = raw_values.copy()

    if drift_slope != 0.0:
        stressed_values = add_baseline_drift(stressed_values, max_slope=drift_slope, drift_type=drift_type)
    if noise_sigma > 0.0:
        stressed_values = add_gaussian_jitter(stressed_values, sigma=noise_sigma)
    if scale_factor != 1.0:
        stressed_values = apply_magnitude_scaling(stressed_values, scale_range=(scale_factor, scale_factor), per_sensor=False)
    if drop_sensor != "None":
        drop_idx = sensor_cols.index(drop_sensor)
        stressed_values[:, drop_idx] = 0.0

    stressed_df = df.copy()
    stressed_df[sensor_cols] = stressed_values

    left_plot, right_plot = st.columns(2)
    with left_plot:
        plot_sensor_curves_interactive(df, time_col, sensor_cols, title="Clean Reference Signal")
    with right_plot:
        plot_sensor_curves_interactive(stressed_df, time_col, sensor_cols, title="Stressed / Distorted Signal")

    try:
        clean_res = predict_dataframe(df, bundle)
        stress_res = predict_dataframe(stressed_df, bundle)

        st.subheader("Model Resilience Comparison")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Clean Prediction", clean_res["predicted_class"], f"{clean_res['confidence']*100:.1f}% conf")
        r2.metric("Stressed Prediction", stress_res["predicted_class"], f"{stress_res['confidence']*100:.1f}% conf")

        match = clean_res["predicted_class"] == stress_res["predicted_class"]
        r3.metric("Prediction Maintained", "✅ Retained" if match else "⚠️ Shifted")
        conf_delta = (stress_res["confidence"] - clean_res["confidence"]) * 100
        r4.metric("Confidence Delta", f"{conf_delta:+.1f}%")

        if match:
            st.success("The model successfully retained the correct smell prediction despite hardware distortion!")
        else:
            st.warning(f"Hardware distortion caused the prediction to shift from `{clean_res['predicted_class']}` to `{stress_res['predicted_class']}`.")
    except Exception as exc:
        st.error(f"Inference under stress failed: {exc}")


def render_mixture_deconvolution_tab() -> None:
    """Interactive tool to deconvolve compound smell mixtures into constituent percentages."""
    st.subheader("Odor Mixture Deconvolution & Ratio Estimation")
    st.write(
        "Real-world odors are frequently combinations of multiple aroma compounds. "
        "This tool uses a multi-output regressor trained on SmellNet mixture data to predict constituent odorants and concentration ratios."
    )

    mix_candidates = list(ROOT.glob("models/mixture*/model.joblib")) + list(ROOT.glob("models/test_mixture/model.joblib"))
    if not mix_candidates:
        st.info("Mixture model artifact not found. Train it with: `uv run python src/models/train_mixture.py`")
        return

    mix_path = mix_candidates[0]
    mix_bundle = cached_load_pipeline(str(mix_path))

    sample_mix_dir = ROOT / "data" / "raw" / "SmellNet" / "mixture_data" / "training_new"
    mix_samples = list(sample_mix_dir.glob("*.csv")) if sample_mix_dir.exists() else []

    c1, c2 = st.columns([1.5, 1.0])
    with c1:
        mix_options = ["None"] + [str(p.name) for p in mix_samples[:50]]
        selected_mix = st.selectbox("Choose a sample mixture recording", mix_options)
    with c2:
        mix_uploaded = st.file_uploader("Or upload a custom mixture CSV", type=["csv"], key="mix_uploader")

    target_df = None
    source_label = ""
    if mix_uploaded is not None:
        target_df = pd.read_csv(mix_uploaded)
        source_label = mix_uploaded.name
    elif selected_mix != "None" and sample_mix_dir.exists():
        chosen_p = sample_mix_dir / selected_mix
        if chosen_p.exists():
            target_df = pd.read_csv(chosen_p)
            source_label = selected_mix

    if target_df is not None:
        try:
            res = deconvolve_mixture_dataframe(target_df, mix_bundle, threshold=0.05)

            st.markdown(
                f"""
                <div class="result-panel">
                  <div class="label">Primary Constituent Odor</div>
                  <div class="value">{res["primary_odor"].replace('_', ' ').title()} ({res["primary_percentage"]:.1f}%)</div>
                  <div class="note">Deconvolved from: {source_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_pie, col_breakdown = st.columns([1.2, 1.0])
            with col_pie:
                active = res["active_components"]
                labels = [c["odor"].replace("_", " ").title() for c in active]
                values = [c["percentage"] for c in active]

                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45, textinfo="label+percent")])
                fig.update_layout(
                    title="Estimated Odor Mixture Proportions",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=340,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_breakdown:
                st.subheader("Constituent Breakdown")
                breakdown_df = pd.DataFrame(active)[["odor", "percentage"]].rename(
                    columns={"odor": "Odor Component", "percentage": "Estimated Share (%)"}
                )
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            st.subheader("Active Component Aroma & Volatile Profiles")
            for comp in active:
                render_semantic_profile_card(comp["odor"])

        except Exception as exc:
            st.error(f"Mixture deconvolution error: {exc}")


def render_hardware_importance_tab(bundle: dict[str, Any], model_path: Path) -> None:
    """Evaluate physical sensor channel predictive power and hardware array pruning."""
    st.subheader("Hardware Sensor Importance & Array Pruning")
    st.write(
        "Electronic nose hardware cost scales with sensor count. "
        "This tool ranks gas sensor channels by their predictive power, helping determine the minimum viable sensor array."
    )

    importance = compute_sensor_importance_scores(bundle)
    if not importance:
        st.info("Importance scores are not available for this model artifact.")
        return

    df_imp = pd.DataFrame(list(importance.items()), columns=["Sensor Channel", "Importance (%)"])

    c_chart, c_recom = st.columns([1.2, 1.0])
    with c_chart:
        fig = px.bar(
            df_imp,
            x="Importance (%)",
            y="Sensor Channel",
            orientation="h",
            color="Importance (%)",
            color_continuous_scale="Viridis",
            title="Sensor Array Predictive Contribution",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c_recom:
        st.subheader("Array Design Recommendations")
        top_sensor = df_imp.iloc[0]["Sensor Channel"]
        top_share = df_imp.iloc[0]["Importance (%)"]
        top_3 = df_imp.head(3)["Sensor Channel"].tolist()
        top_3_share = df_imp.head(3)["Importance (%)"].sum()

        st.markdown(
            f"""
            - **Primary Gas Sensor**: `{top_sensor}` accounts for **{top_share:.1f}%** of classification power.
            - **Top-3 Sensor Core**: `{' + '.join(top_3)}` collectively provide **{top_3_share:.1f}%** of total predictive signal.
            - **Hardware Optimization**: For low-cost embedded e-noses, a reduced {len(top_3)}-sensor array (`{', '.join(top_3)}`) can retain high accuracy while reducing device bill-of-materials (BOM) cost and power consumption.
            """
        )


def render_odor_knowledge_base_tab() -> None:
    """Searchable catalog of all 50 SmellNet smell classes, chemical formulas, and sensory notes."""
    st.subheader("SmellNet Odor & Chemical Knowledge Base")
    st.write(
        "Explore all 50 target smell classes, their botanical classifications, primary chemical volatile compounds (VOCs), and sensory aroma notes."
    )
    kb = get_knowledge_base()
    all_cats = ["All Categories"] + kb.list_categories()

    c_cat, c_search = st.columns([1.0, 1.2])
    with c_cat:
        selected_cat = st.selectbox("Filter by Odor Family", all_cats)
    with c_search:
        search_query = st.text_input("Search smell name, volatile chemical, or flavor note", placeholder="e.g. eugenol, limonene, citrus, spicy...")

    all_classes = kb.list_all_classes()
    filtered_profiles = []
    for name in all_classes:
        prof = kb.get_profile(name)
        if selected_cat != "All Categories" and prof.category != selected_cat:
            continue
        if search_query:
            q = search_query.lower()
            text_match = (
                q in prof.name.lower()
                or q in prof.category.lower()
                or any(q in v.lower() for v in prof.volatiles)
                or any(q in s.lower() for s in prof.sensory_notes)
                or q in prof.full_description.lower()
            )
            if not text_match:
                continue
        filtered_profiles.append(prof)

    st.caption(f"Showing {len(filtered_profiles)} of {len(all_classes)} smell classes.")

    cols = st.columns(2)
    for idx, prof in enumerate(filtered_profiles):
        with cols[idx % 2]:
            render_semantic_profile_card(prof.name)


def render_prediction_explanation(bundle: dict[str, Any], result: dict[str, Any]) -> None:
    st.subheader("Why This Prediction?")
    st.write(
        "For this model, the app looks at the cleaned sensor response windows and identifies which engineered signal clues most supported the predicted smell."
    )
    explanation = linear_explanation_rows(bundle, result)
    if explanation.empty:
        st.info(
            "This selected model does not expose simple linear feature weights. "
            "Use the logistic-regression artifact to see feature-level explanations."
        )
        return
    st.dataframe(explanation, use_container_width=True, hide_index=True)
    st.caption(
        "These are model clues, not chemical proof. They summarize patterns such as sensor slope, response energy, peak timing, and cross-sensor relationships."
    )


def render_signal_analysis(df: pd.DataFrame, time_col: str | None, sensor_cols: list[str]) -> None:
    st.subheader("Sensor Response Analysis")
    st.write(
        "This view treats the uploaded file as a sensor recording over time. The goal is to inspect the response curves before thinking about the classifier."
    )
    plot_sensor_curves_interactive(df, time_col, sensor_cols, title="Raw Gas Sensor Response Curves")

    summary = (
        df[sensor_cols]
        .agg(["mean", "std", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "sensor", "mean": "average response", "std": "variation", "min": "lowest reading", "max": "highest reading"})
    )
    st.dataframe(summary.round(4), use_container_width=True, hide_index=True)
    st.caption("Large variation, slope, timing, and cross-sensor differences are the kind of signals the model turns into features.")


def render_research_summary(metrics: dict) -> None:
    test_metrics = metrics.get("test_metrics", {})
    st.subheader("Research Summary")
    st.markdown(
        """
        **Research question:** Can low-cost gas sensor response curves distinguish smell classes?

        **Method:** Clean each sensor trial, split it into time windows, extract signal-shape features, compare classical baselines, and score full-trial predictions.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Smell classes", f"{len(metrics.get('class_names', []))}")
    c2.metric("Trial Top-1", f"{test_metrics.get('trial_top1_accuracy', 0.0) * 100:.1f}%", help=METRIC_EXPLANATIONS["Trial Top-1"])
    c3.metric("Trial Top-5", f"{test_metrics.get('trial_top5_accuracy', 0.0) * 100:.1f}%", help=METRIC_EXPLANATIONS["Trial Top-5"])


def render_ablation_study(comparison_df: pd.DataFrame) -> None:
    st.subheader("Ablation Study")
    st.write("This table shows how design choices affect full-CSV smell prediction. Higher trial metrics mean better user-facing predictions.")
    if comparison_df.empty:
        st.info("No model metrics are available yet.")
        return
    ablation = comparison_df[["artifact", "feature_mode", "trial_top1", "trial_top5", "window_top1"]].copy()
    ablation = ablation.sort_values("trial_top1", ascending=False, na_position="last")
    for col in ["trial_top1", "trial_top5", "window_top1"]:
        ablation[col] = ablation[col].map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%")
    ablation = ablation.rename(
        columns={
            "artifact": "model setup",
            "feature_mode": "feature style",
            "trial_top1": "Full CSV correct first guess",
            "trial_top5": "Full CSV correct in top 5",
            "window_top1": "Window-level correct first guess",
        }
    )
    st.dataframe(ablation, use_container_width=True, hide_index=True)


def render_error_analysis(metrics: dict, model_path: Path) -> None:
    st.subheader("Error Analysis")
    class_report = metrics.get("class_report", {})
    class_names = metrics.get("class_names", [])
    rows = []
    for name in class_names:
        report = class_report.get(name, {})
        rows.append(
            {
                "smell class": name,
                "recall": report.get("recall"),
                "precision": report.get("precision"),
                "f1": report.get("f1-score"),
            }
        )
    if rows:
        report_df = pd.DataFrame(rows).sort_values("f1", ascending=True, na_position="last")
        st.write("Lowest-F1 classes are where the current sensor-feature baseline struggles most.")
        st.dataframe(report_df.head(12).round(3), use_container_width=True, hide_index=True)

    cm_path = model_path.parent / "confusion_matrix.npy"
    if not cm_path.exists() or not class_names:
        return
    cm = np.load(cm_path)
    confusions = []
    for true_idx, true_name in enumerate(class_names):
        for pred_idx, pred_name in enumerate(class_names):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                confusions.append(
                    {
                        "actual smell": true_name,
                        "predicted as": pred_name,
                        "count": int(cm[true_idx, pred_idx]),
                    }
                )
    if confusions:
        confusion_df = pd.DataFrame(confusions).sort_values("count", ascending=False).head(12)
        st.write("Most common mix-ups on the held-out test split:")
        st.dataframe(confusion_df, use_container_width=True, hide_index=True)


def render_model_metrics(metrics: dict) -> None:
    test_metrics = metrics.get("test_metrics", {})
    if not test_metrics:
        st.caption("No metrics file found beside the model artifact.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Trial Top-1", f"{test_metrics.get('trial_top1_accuracy', 0.0) * 100:.1f}%", help=METRIC_EXPLANATIONS["Trial Top-1"])
    m2.metric("Trial Top-5", f"{test_metrics.get('trial_top5_accuracy', 0.0) * 100:.1f}%", help=METRIC_EXPLANATIONS["Trial Top-5"])
    m3.metric("Macro F1", f"{test_metrics.get('trial_macro_f1', 0.0):.3f}", help=METRIC_EXPLANATIONS["Macro F1"])
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


def render_current_work_description() -> None:
    st.subheader("Current Work")
    st.write(
        "The current version is a complete research baseline for smell recognition from SmellNet gas sensor recordings. "
        "It focuses on the `base_data` task, where each sensor trial belongs to one known smell class such as a fruit, spice, nut, herb, or vegetable."
    )

    current = pd.DataFrame(
        [
            {
                "part": "Dataset focus",
                "what it means": "Uses SmellNet base gas-sensor trials, where one CSV represents one odor exposure over time.",
                "why it matters": "The file is treated as a sensor response curve, not as unrelated spreadsheet rows.",
            },
            {
                "part": "Signal preparation",
                "what it means": "Fills missing readings, trims early warm-up drift, resamples every trial to a common length, and normalizes each sensor.",
                "why it matters": "Gas sensors are noisy and can drift, so the signal must be cleaned before modeling.",
            },
            {
                "part": "Feature extraction",
                "what it means": "Turns each time window into interpretable features such as slope, energy, peak timing, frequency balance, and cross-sensor relationships.",
                "why it matters": "These features describe the shape of the smell response instead of only using raw CSV values.",
            },
            {
                "part": "Current model",
                "what it means": "Uses the `baseline_windowed_trialmax` artifact, a trial-level windowed logistic-regression baseline.",
                "why it matters": "It is easier to explain than a black-box deep model and supports feature-level prediction explanations.",
            },
            {
                "part": "Current result",
                "what it means": "On the held-out SmellNet test split, the saved model reaches 64.0% Trial Top-1 and 92.0% Trial Top-5 across 50 classes.",
                "why it matters": "Trial-level metrics match the app experience because users submit one full sensor trial at a time.",
            },
            {
                "part": "Research demo",
                "what it means": "The app shows sensor curves, top-5 predictions, confidence, schema checks, prediction clues, ablation results, and error analysis.",
                "why it matters": "A reviewer can inspect how the sensor signal is processed and where the baseline succeeds or struggles.",
            },
        ]
    )
    st.dataframe(current, use_container_width=True, hide_index=True)

    st.info(
        "Plain-language summary: this project asks whether low-cost gas sensor response curves can help identify smell classes. "
        "The current answer is a transparent baseline that works reasonably well for research, while clearly showing uncertainty and limitations."
    )


def render_model_comparison(comparison_df: pd.DataFrame) -> None:
    if comparison_df.empty:
        st.info("No saved model artifacts were found under `models/*/model.joblib`.")
        return

    display = comparison_df.copy()
    for col in ["trial_top1", "trial_top5", "macro_f1", "window_top1"]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%" if col != "macro_f1" else f"{value:.3f}")
    display = display.rename(
        columns={
            "artifact": "artifact",
            "best_model": "selected_model",
            "split": "split",
            "feature_mode": "feature_mode",
            "trial_top1": "Trial Top-1",
            "trial_top5": "Trial Top-5",
            "macro_f1": "Trial Macro F1",
            "window_top1": "Window Top-1",
            "classes": "classes",
            "artifact_file": "artifact file",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_metric_glossary() -> None:
    st.subheader("Metric Glossary")
    glossary = pd.DataFrame(
        [
            {"metric": name, "plain meaning": meaning}
            for name, meaning in METRIC_EXPLANATIONS.items()
        ]
    )
    st.dataframe(glossary, use_container_width=True, hide_index=True)


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
            {"option": "Feature-selected logistic baseline", "command flag": "--feature-select-k 80", "tradeoff": "Keeps the strongest engineered signal features before logistic regression, which can reduce noisy feature effects."},
            {"option": "Cross-validated logistic baseline", "command flag": "--include-logistic-cv", "tradeoff": "Tunes logistic-regression strength automatically; slower, but useful for accuracy experiments."},
            {"option": "Include SVM", "command flag": "--include-svm", "tradeoff": "Adds an RBF SVM and a larger soft-voting ensemble; slower but can improve accuracy."},
            {"option": "Tune preprocessing", "command flag": "--target-points, --warmup-ratio", "tradeoff": "Changes signal shape seen by every model; rerun evaluation after changing."},
            {"option": "Optional sequence model", "command flag": "src/models/train_timeseries.py", "tradeoff": "Experimental PyTorch path; classical baseline is currently stronger."},
        ]
    )
    st.dataframe(options, use_container_width=True, hide_index=True)
    st.code(
        "uv run python src/models/train_baseline.py --data-root data/raw/SmellNet "
        "--output-dir models/baseline_windowed_trialmax --window-size 100 --window-stride 25 --include-svm --include-logistic-cv --feature-select-k 80 --trial-aggregation max",
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


def render_uniqueness_playbook() -> None:
    st.subheader("What Makes This Project Specific")
    st.write(
        "This project is specific to gas sensor time-series data. "
        "The CSV file stores the recording, but the useful information is in how sensor readings change during one smell exposure."
    )

    angles = pd.DataFrame(
        [
            {
                "part": "Sensor response patterns",
                "what the system does": "Represents each smell trial using curve shape, response energy, peak timing, drift, and sensor-to-sensor relationships.",
                "plain meaning": "Different smells can create different patterns across the sensor array, even when one sensor alone is not enough.",
            },
            {
                "part": "Window-based prediction",
                "what the system does": "Splits a trial into time windows, predicts probabilities for each window, and combines them into one CSV-level result.",
                "plain meaning": "The model can use the strongest parts of the sensor response instead of flattening the whole file into one table row.",
            },
            {
                "part": "Interpretable baseline",
                "what the system does": "Uses named signal features with classical models before relying on more complex sequence models.",
                "plain meaning": "It is easier to inspect which sensor clues contributed to a prediction.",
            },
            {
                "part": "Uncertainty shown in the app",
                "what the system does": "Shows top-5 alternatives, confidence labels, schema checks, and model metrics.",
                "plain meaning": "The interface shows when the signal is ambiguous instead of pretending every answer is certain.",
            },
            {
                "part": "Modular workflow",
                "what the system does": "Separates preprocessing, features, training, evaluation, and app inference.",
                "plain meaning": "Future work can improve one part of the workflow without rewriting the whole project.",
            },
        ]
    )
    st.dataframe(angles, use_container_width=True, hide_index=True)


def render_presentation_script() -> None:
    st.subheader("Simple Project Explanation")
    st.markdown(
        """
        **Short version:** This project helps a computer work with smell-related sensor data. A gas sensor array records how its readings change during one smell exposure. The software cleans that signal, looks for patterns in the sensor curves, and predicts which known smell class the recording most resembles.

        **Developer version:** The pipeline detects sensor columns, handles missing values, trims early warm-up behavior, resamples each trial to a common length, normalizes each sensor, creates windows, extracts statistical, derivative, timing, frequency, and cross-sensor features, and trains classical classifiers. At inference time, the app reuses the same preprocessing and combines window probabilities into one full-trial prediction.

        **Why it matters:** Smell is hard to describe consistently, and low-cost gas sensors can be noisy. This project turns sensor response curves into a repeatable workflow that can be inspected, compared, and explained.
        """
    )


def render_future_uniqueness() -> None:
    st.subheader("What Still Needs Work")
    st.write(
        "These are practical improvements that would make the project clearer, more reliable, and easier to use."
    )
    future = pd.DataFrame(
        [
            {
                "addition": "Confidence calibration",
                "why it helps": "Turns raw model probabilities into better reliability estimates.",
                "deliverable": "Reliability plot, expected calibration error, and calibrated confidence labels.",
            },
            {
                "addition": "Drift stress test",
                "why it helps": "Gas sensors drift across time and environments, so robustness matters more than a single accuracy number.",
                "deliverable": "Evaluate performance by collection split/session if metadata supports it, then document failure modes.",
            },
            {
                "addition": "Sensor fingerprint report",
                "why it helps": "Gives each prediction a human-readable signal summary instead of only a class name.",
                "deliverable": "Peak timing, strongest sensors, slope direction, cross-sensor similarity, and top supporting features.",
            },
            {
                "addition": "Mixture recognition branch",
                "why it helps": "Mixtures are closer to real smells than single ingredients.",
                "deliverable": "Separate multi-label or ingredient-composition experiment using `mixture_data`, with its own metrics.",
            },
            {
                "addition": "GC-MS prior comparison",
                "why it helps": "Connects sensor behavior with chemical-profile information available in SmellNet.",
                "deliverable": "Compare sensor-only retrieval against a chemistry-informed ranking experiment.",
            },
        ]
    )
    st.dataframe(future, use_container_width=True, hide_index=True)


def render_project_guide() -> None:
    st.markdown(
        """
        <div class="app-hero">
          <h1>Project Guide</h1>
          <p>A plain-language overview of what this machine olfaction project does, how it works, what is working now, and what still needs improvement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_usecase_overview()
    st.divider()
    render_current_work_description()
    st.divider()
    render_uniqueness_playbook()
    st.divider()
    render_presentation_script()
    st.divider()
    render_research_workflow()
    st.divider()
    render_metric_glossary()
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

    st.info(
        "Plain explanation: the CSV is only the file format. The model learns from response curves, window behavior, and cross-sensor patterns."
    )

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
            {"where": "Prediction Demo > Signal Analysis", "what you do there": "Inspect the gas sensor response curves and sensor summary before looking at the model."},
            {"where": "Prediction Demo > Research Evidence", "what you do there": "Read the research question, ablation table, and error analysis in plain language."},
            {"where": "Prediction Demo > CSV Details", "what you do there": "Inspect detected sensors, missing columns, and raw CSV rows."},
            {"where": "Prediction Demo > Models", "what you do there": "Compare saved model artifacts and see the model-changing training options."},
            {"where": "Prediction Demo > Method", "what you do there": "Understand preprocessing, feature extraction, evaluation, and limitations."},
        ]
    )
    st.dataframe(navigation, use_container_width=True, hide_index=True)

    st.divider()
    render_future_uniqueness()

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
        with st.expander("Advanced local settings", expanded=False):
            st.caption("For deployed demos, keep this as the bundled sample folder unless you are running locally with a different dataset layout.")
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
            st.caption(f"{len(samples)} bundled demo CSVs available.")
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
        st.error(f"Model artifact not found: {project_relative_path(model_path)}")
        return

    try:
        bundle = cached_load_pipeline(str(model_path))
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    metrics = load_metrics(model_path)
    trained_sensor_cols = list(bundle["sensor_columns"])

    overview_tab, predict_tab, signal_tab, mixture_tab, drift_tab, hardware_tab, kb_tab, research_tab, details_tab, models_tab, method_tab = st.tabs(
        [
            "Overview",
            "Prediction Results",
            "Signal Analysis",
            "Mixture Deconvolution",
            "Stress Test & Drift",
            "Hardware Importance",
            "Odor Knowledge Base",
            "Research Evidence",
            "CSV Details",
            "Models",
            "Method",
        ]
    )

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
                st.caption(f"Source: {display_source_name(source_name, sample_root)}")
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
                  <div class="value">{result["predicted_class"].replace('_', ' ').title()}</div>
                  <div class="note">{confidence_name} confidence: {result["confidence"] * 100:.1f}% across {result["n_windows"]} analyzed window(s), using {result["aggregation"]} aggregation.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Create visual metrics row with confidence gauge
            col_gauge, col_metrics = st.columns([1, 2])
            
            with col_gauge:
                plot_confidence_gauge(result["confidence"], result["predicted_class"])
            
            with col_metrics:
                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted Smell", result["predicted_class"].replace('_', ' ').title())
                r2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                r3.metric("Analyzed Windows", result["n_windows"])
                st.info(confidence_text)

            # Render rich aroma and chemical volatile profile card
            render_semantic_profile_card(result["predicted_class"])

            left, right = st.columns([1.0, 1.0])
            with left:
                st.subheader("Top-5 Predictions")
                plot_top_predictions_interactive(top5_df)
            with right:
                display_df = top5_df.assign(probability=lambda x: (x["probability"] * 100).round(2))
                st.subheader("Probability Table")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            render_prediction_explanation(bundle, result)

            with st.expander("Preprocessed Signal Used by the Model", expanded=False):
                processed_df = result["processed_df"]
                st.caption(f"Shape after preprocessing: {processed_df.shape}")
                st.dataframe(processed_df.head(30), use_container_width=True)
                plot_sensor_curves_interactive(processed_df, "time", trained_sensor_cols, title="Preprocessed Model Input")
        else:
            st.info("Click `Analyze Smell Signal` to run the selected model on this CSV.")

        st.subheader("Sensor Curves")
        plot_sensor_curves_interactive(df, time_col, trained_sensor_cols, title="Raw Sensor Response")

    with signal_tab:
        render_signal_analysis(df, time_col, trained_sensor_cols)

    with mixture_tab:
        render_mixture_deconvolution_tab()

    with drift_tab:
        render_drift_simulator_tab(df, time_col, trained_sensor_cols, bundle)

    with hardware_tab:
        render_hardware_importance_tab(bundle, model_path)

    with kb_tab:
        render_odor_knowledge_base_tab()

    with research_tab:
        render_research_summary(metrics)
        st.divider()
        render_ablation_study(comparison_df)
        st.divider()
        render_error_analysis(metrics, model_path)

    with details_tab:
        st.subheader("Schema and CSV Preview")
        render_schema_panel(df, trained_sensor_cols, detected_sensor_cols, missing_cols)
        st.dataframe(df.head(40), use_container_width=True)

    with models_tab:
        st.subheader("Saved Model Comparison")
        render_model_comparison(comparison_df)
        render_metric_glossary()
        render_training_options()
        if metrics.get("val_results"):
            st.subheader("Validation Candidates Inside Selected Artifact")
            val_df = pd.DataFrame(metrics["val_results"]).T.reset_index(names="candidate_model")
            st.dataframe(val_df, use_container_width=True, hide_index=True)

    with method_tab:
        render_method_notes()


if __name__ == "__main__":
    main()
