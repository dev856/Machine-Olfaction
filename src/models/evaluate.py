from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pipeline_io import load_pipeline



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved baseline model on held-out data.")
    parser.add_argument("--model-path", default="models/baseline/model.joblib")
    parser.add_argument("--eval-data", default="models/baseline/eval_data.npz")
    parser.add_argument("--output-dir", default="models/baseline")
    parser.add_argument(
        "--trial-aggregation",
        choices=["artifact", "mean", "max", "median"],
        default="artifact",
        help="How to aggregate window probabilities for trial-level metrics.",
    )
    return parser.parse_args()


def aggregate_probabilities(proba: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        out = np.mean(proba, axis=0)
    elif method == "max":
        out = np.max(proba, axis=0)
    elif method == "median":
        out = np.median(proba, axis=0)
    else:
        raise ValueError(f"Unknown trial aggregation method: {method}")

    total = float(np.sum(out))
    return out / total if total > 0 else out



def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    eval_path = Path(args.eval_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_pipeline(model_path)
    model = bundle["model"]
    class_names = bundle["label_encoder"].classes_
    trial_aggregation = bundle.get("trial_aggregation", "mean") if args.trial_aggregation == "artifact" else args.trial_aggregation

    data = np.load(eval_path)
    X_test = data["X_test"]
    y_test = data["y_test"]

    proba = model.predict_proba(X_test)
    if "proba_test" in data:
        proba = data["proba_test"]
    y_pred = np.argmax(proba, axis=1)

    top1 = float(accuracy_score(y_test, y_pred))
    top5 = float(top_k_accuracy_score(y_test, proba, k=min(5, len(class_names)), labels=np.arange(len(class_names))))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    metrics = {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    trial_report = None
    if "groups_test" in data:
        groups = data["groups_test"]
        trial_true: list[int] = []
        trial_proba: list[np.ndarray] = []
        for group in pd.unique(groups):
            mask = groups == group
            labels = y_test[mask]
            if len(np.unique(labels)) != 1:
                raise ValueError(f"Cannot aggregate trial with multiple labels: {group}")
            trial_true.append(int(labels[0]))
            trial_proba.append(aggregate_probabilities(proba[mask], method=trial_aggregation))

        trial_y = np.array(trial_true)
        trial_p = np.vstack(trial_proba)
        trial_pred = np.argmax(trial_p, axis=1)
        metrics.update(
            {
                "trial_top1_accuracy": float(accuracy_score(trial_y, trial_pred)),
                "trial_top5_accuracy": float(
                    top_k_accuracy_score(trial_y, trial_p, k=min(5, len(class_names)), labels=np.arange(len(class_names)))
                ),
                "trial_macro_f1": float(f1_score(trial_y, trial_pred, average="macro", zero_division=0)),
                "trial_weighted_f1": float(f1_score(trial_y, trial_pred, average="weighted", zero_division=0)),
            }
        )
        trial_report = classification_report(
            trial_y,
            trial_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names.tolist(),
            output_dict=True,
            zero_division=0,
        )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names.tolist(),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "class_report": report,
                "trial_class_report": trial_report,
                "trial_aggregation": trial_aggregation,
            },
            f,
            indent=2,
        )

    per_class = pd.DataFrame(report).T
    per_class.to_csv(output_dir / "per_class_report.csv", index=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)

    print("Evaluation metrics:", metrics)
    print("Trial aggregation:", trial_aggregation)
    print("Saved:")
    print(f"- {output_dir / 'evaluation_metrics.json'}")
    print(f"- {output_dir / 'per_class_report.csv'}")
    print(f"- {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
