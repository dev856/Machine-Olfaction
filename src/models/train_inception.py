"""Train the SOTA InceptionTime neural network on SmellNet time-series data.

Features:
- Multi-scale Inception 1D convolutions with residual shortcuts
- On-the-fly sensor signal augmentation (thermal drift, jitter, scaling, dropout)
- Leakage-aware grouped / folder splits
- Full metrics reporting (Top-1, Top-5, Macro F1, Weighted F1, Confusion Matrix)
- Unified artifact saving compatible with the Streamlit app and FastAPI
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.augmentation import AugmentationConfig, SensorAugmenter
from src.data.preprocess import PreprocessConfig, preprocess_trial
from src.models.pipeline_io import save_pipeline
from src.models.timeseries_models import InceptionTime, ResNet1D

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


class SensorDataset(Dataset):
    """PyTorch Dataset for multivariate sensor time-series with optional augmentation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augmenter: SensorAugmenter | None = None) -> None:
        self.X = X  # (N, seq_len, n_sensors)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr = self.X[idx]
        if self.augmenter is not None:
            arr = self.augmenter.augment(arr)
        # Transpose to (n_sensors, seq_len)
        tensor_x = torch.from_numpy(arr.T).float()
        return tensor_x, self.y[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train InceptionTime neural network on SmellNet.")
    parser.add_argument("--data-root", default="data/raw/SmellNet", help="SmellNet root directory")
    parser.add_argument("--output-dir", default="models/inception_time", help="Output directory")
    parser.add_argument("--target-points", type=int, default=300, help="Resample length per trial")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup trim ratio")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model-type", choices=["inception", "resnet"], default="inception", help="Architecture")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable sensor data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-files", type=int, default=0, help="Cap for fast debugging")
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
    return [p for p in sorted(base_dir.rglob("*.csv")) if ".cache" not in p.parts]


def label_from_filename(csv_path: Path) -> str:
    parts = csv_path.stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return csv_path.stem


def build_dataset(csv_files: list[Path], cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    samples: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    sensor_cols_ref: list[str] | None = None

    for p in csv_files:
        try:
            df = pd.read_csv(p)
            proc_df, _, cols = preprocess_trial(df, sensor_columns=sensor_cols_ref, config=cfg)
            if sensor_cols_ref is None:
                sensor_cols_ref = cols
            arr = proc_df[sensor_cols_ref].to_numpy(dtype=np.float32)
            samples.append(arr)
            labels.append(label_from_filename(p))
            groups.append(str(p))
        except Exception as exc:
            print(f"Skipping {p}: {exc}")

    if not samples or sensor_cols_ref is None:
        raise RuntimeError("No time-series samples generated.")

    return np.stack(samples, axis=0), np.array(labels), np.array(groups), sensor_cols_ref


def evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    n_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probas: list[np.ndarray] = []

    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probas, axis=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(y_b.numpy().tolist())
            all_probas.append(probas)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probas_matrix = np.vstack(all_probas)

    acc = float(accuracy_score(y_true, y_pred))
    k = min(5, n_classes)
    top5 = float(top_k_accuracy_score(y_true, probas_matrix, k=k, labels=np.arange(n_classes)))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "top5_accuracy": top5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not installed in the current environment.")

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_base_files(data_root)
    if args.max_files > 0:
        files = files[: args.max_files]

    prep_cfg = PreprocessConfig(target_points=args.target_points, warmup_ratio=args.warmup_ratio)
    print(f"Loading and preprocessing {len(files)} files...")
    X, y_labels, groups, sensor_columns = build_dataset(files, prep_cfg)

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_labels)
    n_classes = len(label_enc.classes_)
    n_sensors = len(sensor_columns)

    # Train / Val / Test splitting
    training_mask = np.array(["training" in Path(g).parts for g in groups])
    testing_mask = np.array(["testing" in Path(g).parts for g in groups])

    if args.use_folder_split and np.any(training_mask) and np.any(testing_mask):
        print("Using SmellNet base_data folder split...")
        train_val_idx = np.where(training_mask)[0]
        test_idx = np.where(testing_mask)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        train_sub, val_sub = next(gss.split(train_val_idx, groups=groups[train_val_idx]))
        train_idx = train_val_idx[train_sub]
        val_idx = train_val_idx[val_sub]
    else:
        print("Using grouped shuffle split...")
        gss1 = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_val_idx, test_idx = next(gss1.split(X, y, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_size / (1.0 - args.test_size), random_state=args.seed)
        train_sub, val_sub = next(gss2.split(train_val_idx, groups=groups[train_val_idx]))
        train_idx = train_val_idx[train_sub]
        val_idx = train_val_idx[val_sub]

    print(f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} trials. Classes={n_classes}")

    augmenter = SensorAugmenter(seed=args.seed) if args.augment else None
    train_ds = SensorDataset(X[train_idx], y[train_idx], augmenter=augmenter)
    val_ds = SensorDataset(X[val_idx], y[val_idx], augmenter=None)
    test_ds = SensorDataset(X[test_idx], y[test_idx], augmenter=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.model_type == "inception":
        model = InceptionTime(n_sensors=n_sensors, n_classes=n_classes)
    else:
        model = ResNet1D(n_sensors=n_sensors, n_classes=n_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None

    print(f"Training {args.model_type.title()} for {args.epochs} epochs on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)

        scheduler.step()
        train_loss = total_loss / len(train_idx)
        val_eval = evaluate_torch_model(model, val_loader, n_classes, device)

        if val_eval["accuracy"] > best_val_acc:
            best_val_acc = val_eval["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_eval['accuracy']:.3f} | Val Top-5: {val_eval['top5_accuracy']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_eval = evaluate_torch_model(model, test_loader, n_classes, device)
    print("\n--- Final Test Results ---")
    print(f"Test Top-1:     {test_eval['accuracy']:.4f}")
    print(f"Test Top-5:     {test_eval['top5_accuracy']:.4f}")
    print(f"Test Macro F1:  {test_eval['macro_f1']:.4f}")
    print(f"Test W-F1:      {test_eval['weighted_f1']:.4f}")

    # Build and save full artifact bundle
    model.eval()
    model.cpu()
    bundle = {
        "framework": "pytorch",
        "architecture": f"InceptionTime_{args.model_type}",
        "best_model_name": f"PyTorch_{args.model_type.capitalize()}",
        "model": model,
        "label_encoder": label_enc,
        "sensor_columns": sensor_columns,
        "preprocess_config": asdict(prep_cfg),
        "target_points": args.target_points,
        "feature_mode": "sequence",
        "trial_aggregation": "sequence",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_pipeline(bundle, output_dir / "model.joblib")

    # Save metrics JSON
    metrics = {
        "best_model_name": f"PyTorch_{args.model_type.capitalize()}",
        "trial_top1": test_eval["accuracy"],
        "trial_top5": test_eval["top5_accuracy"],
        "macro_f1": test_eval["macro_f1"],
        "weighted_f1": test_eval["weighted_f1"],
        "test_accuracy": test_eval["accuracy"],
        "test_top5": test_eval["top5_accuracy"],
        "n_classes": n_classes,
        "n_sensors": n_sensors,
        "target_points": args.target_points,
        "epochs": args.epochs,
        "model_type": args.model_type,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel artifact and metrics saved to: {output_dir}")


if __name__ == "__main__":
    main()
