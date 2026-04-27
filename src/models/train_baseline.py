from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import PreprocessConfig, preprocess_trial, window_array
from src.features.extract_features import extract_window_feature_vector, feature_names_for_sensors
from src.models.pipeline_io import save_pipeline



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline smell classifier from feature windows.")
    parser.add_argument("--data-root", default="data/raw/SmellNet", help="SmellNet root directory")
    parser.add_argument("--output-dir", default="models/baseline", help="Output directory")
    parser.add_argument("--target-points", type=int, default=300, help="Resample length for each trial")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of early rows to drop")
    parser.add_argument("--window-size", type=int, default=0, help="Window size; 0 uses full sequence")
    parser.add_argument("--window-stride", type=int, default=50, help="Stride for window extraction")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation fraction from total dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-svm", action="store_true", help="Also train SVM baseline")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap for fast debugging")
    parser.add_argument(
        "--use-folder-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SmellNet base_data/training and base_data/testing folders when present",
    )
    return parser.parse_args()



def discover_base_files(data_root: Path) -> list[Path]:
    base_dir = data_root / "base_data"
    if not base_dir.exists():
        raise FileNotFoundError(f"base_data folder not found under: {data_root}")

    files = [
        p
        for p in sorted(base_dir.rglob("*.csv"))
        if ".cache" not in p.parts
    ]
    return files



def label_from_filename(csv_path: Path) -> str:
    stem = csv_path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def split_name_from_path(csv_path: Path) -> str:
    parts = {part.lower() for part in csv_path.parts}
    if "testing" in parts:
        return "test"
    if "training" in parts:
        return "train"
    return "unknown"



def compute_metrics(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> dict[str, float]:
    y_pred = np.argmax(proba, axis=1)
    top1 = float(accuracy_score(y_true, y_pred))

    k = min(5, n_classes)
    top5 = float(top_k_accuracy_score(y_true, proba, k=k, labels=np.arange(n_classes)))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_trial_metrics(y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray, n_classes: int) -> dict[str, float]:
    trial_true: list[int] = []
    trial_proba: list[np.ndarray] = []

    for group in pd.unique(groups):
        mask = groups == group
        group_labels = y_true[mask]
        if len(np.unique(group_labels)) != 1:
            raise ValueError(f"Cannot aggregate trial with multiple labels: {group}")
        trial_true.append(int(group_labels[0]))
        trial_proba.append(np.mean(proba[mask], axis=0))

    metrics = compute_metrics(np.array(trial_true), np.vstack(trial_proba), n_classes=n_classes)
    return {f"trial_{key}": value for key, value in metrics.items()}


def label_aware_group_holdout(labels: np.ndarray, groups: np.ndarray, holdout_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    group_df = pd.DataFrame({"group": groups, "label": labels}).drop_duplicates("group")

    holdout_groups: list[str] = []
    for _, class_df in group_df.groupby("label", sort=True):
        class_groups = class_df["group"].to_numpy()
        rng.shuffle(class_groups)
        if len(class_groups) <= 1:
            continue
        n_holdout = int(round(len(class_groups) * holdout_size))
        n_holdout = min(max(1, n_holdout), len(class_groups) - 1)
        holdout_groups.extend(class_groups[:n_holdout].tolist())

    holdout_group_set = set(holdout_groups)
    holdout_mask = np.array([group in holdout_group_set for group in groups])
    return np.flatnonzero(~holdout_mask), np.flatnonzero(holdout_mask)



def build_feature_dataset(csv_files: list[Path], cfg: PreprocessConfig, window_size: int, window_stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    source_splits: list[str] = []
    sample_ids: list[str] = []

    sensor_columns_ref: list[str] | None = None

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            proc_df, _, sensor_cols = preprocess_trial(df, sensor_columns=sensor_columns_ref, config=cfg)

            if sensor_columns_ref is None:
                sensor_columns_ref = sensor_cols

            values = proc_df[sensor_columns_ref].to_numpy(dtype=float)
            windows = window_array(values, window_size=window_size, stride=window_stride)

            label = label_from_filename(csv_path)
            group_id = str(csv_path)

            for window_idx, window in enumerate(windows):
                vec = extract_window_feature_vector(window, sensor_columns_ref)
                features.append(vec)
                labels.append(label)
                groups.append(group_id)
                source_splits.append(split_name_from_path(csv_path))
                sample_ids.append(f"{csv_path.name}::w{window_idx}")
        except Exception as exc:
            print(f"Skipping {csv_path} due to error: {exc}")

    if not features or sensor_columns_ref is None:
        raise RuntimeError("No training samples were created from base_data files.")

    return (
        np.vstack(features),
        np.array(labels),
        np.array(groups),
        np.array(source_splits),
        sensor_columns_ref,
        sample_ids,
    )



def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_base_files(data_root)
    if args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    print(f"Found {len(csv_files)} base_data CSV files")

    cfg = PreprocessConfig(
        target_points=args.target_points,
        warmup_ratio=args.warmup_ratio,
        fill_strategy="ffill_bfill",
        normalize=True,
    )

    X, y_labels, groups, source_splits, sensor_columns, sample_ids = build_feature_dataset(
        csv_files,
        cfg,
        window_size=args.window_size,
        window_stride=args.window_stride,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    n_classes = len(label_encoder.classes_)

    has_folder_split = args.use_folder_split and {"train", "test"}.issubset(set(source_splits.tolist()))
    if has_folder_split:
        train_val_idx = np.flatnonzero(source_splits == "train")
        test_idx = np.flatnonzero(source_splits == "test")
    else:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    train_idx_rel, val_idx_rel = label_aware_group_holdout(
        y_labels[train_val_idx],
        groups[train_val_idx],
        holdout_size=args.val_size,
        seed=args.seed,
    )
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Dataset windows: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    models: dict[str, object] = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            l2_regularization=0.01,
            random_state=args.seed,
        ),
    }

    if args.include_svm:
        models["svm_rbf"] = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=args.seed),
        )

    val_results: dict[str, dict[str, float]] = {}
    best_name: str | None = None
    best_top1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        proba_val = model.predict_proba(X_val)
        metrics = compute_metrics(y_val, proba_val, n_classes=n_classes)
        metrics.update(compute_trial_metrics(y_val, proba_val, groups[val_idx], n_classes=n_classes))
        val_results[name] = metrics
        print(f"{name} val metrics: {metrics}")

        selection_score = metrics["trial_top1_accuracy"]
        if selection_score > best_top1:
            best_top1 = selection_score
            best_name = name

    if best_name is None:
        raise RuntimeError("No model was trained.")

    best_model = models[best_name]
    best_model.fit(X[train_val_idx], y[train_val_idx])
    proba_test = best_model.predict_proba(X_test)

    test_metrics = compute_metrics(y_test, proba_test, n_classes=n_classes)
    test_metrics.update(compute_trial_metrics(y_test, proba_test, groups[test_idx], n_classes=n_classes))
    y_pred_test = np.argmax(proba_test, axis=1)

    class_report = classification_report(
        y_test,
        y_pred_test,
        labels=np.arange(n_classes),
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred_test, labels=np.arange(n_classes))

    feature_names = feature_names_for_sensors(sensor_columns)
    split_manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group": groups,
            "label": y_labels,
            "source_split": source_splits,
            "split": "train",
        }
    )
    split_manifest.loc[val_idx, "split"] = "val"
    split_manifest.loc[test_idx, "split"] = "test"

    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    pipeline_bundle = {
        "model": best_model,
        "best_model_name": best_name,
        "label_encoder": label_encoder,
        "sensor_columns": sensor_columns,
        "feature_names": feature_names,
        "preprocess_config": asdict(cfg),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_split_used": "folder" if has_folder_split else "random_group",
    }
    save_pipeline(pipeline_bundle, output_dir / "model.joblib")

    np.savez(
        output_dir / "eval_data.npz",
        X_test=X_test,
        y_test=y_test,
        y_label_test=y_labels[test_idx],
        groups_test=groups[test_idx],
        sample_ids_test=np.array(sample_ids, dtype=object)[test_idx],
        proba_test=proba_test,
    )
    np.save(output_dir / "confusion_matrix.npy", cm)

    metrics_payload = {
        "val_results": val_results,
        "best_model": best_name,
        "test_metrics": test_metrics,
        "class_report": class_report,
        "class_names": label_encoder.classes_.tolist(),
        "n_samples": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "n_trials": {
            "train": int(pd.Series(groups[train_idx]).nunique()),
            "val": int(pd.Series(groups[val_idx]).nunique()),
            "test": int(pd.Series(groups[test_idx]).nunique()),
        },
        "split_strategy": "folder" if has_folder_split else "random_group",
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Saved model artifact and metrics to:", output_dir)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
