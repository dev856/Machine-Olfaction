"""Train an LSTM-based smell classifier for time-series sensor data.

This model captures long-term temporal dependencies in sensor response curves,
which can improve accuracy over the basic 1D CNN baseline.
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import PreprocessConfig, preprocess_trial


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM smell classifier.")
    parser.add_argument("--data-root", default="data/raw/SmellNet")
    parser.add_argument("--output-dir", default="models/lstm")
    parser.add_argument("--target-points", type=int, default=300)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args()


def discover_base_files(data_root: Path) -> list[Path]:
    base_dir = data_root / "base_data"
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
    sensor_columns_ref: list[str] | None = None

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            proc_df, _, sensor_cols = preprocess_trial(df, sensor_columns=sensor_columns_ref, config=cfg)
            if sensor_columns_ref is None:
                sensor_columns_ref = sensor_cols

            arr = proc_df[sensor_columns_ref].to_numpy(dtype=np.float32)
            samples.append(arr)
            labels.append(label_from_filename(csv_path))
            groups.append(str(csv_path))
        except Exception as exc:
            print(f"Skipping {csv_path} due to error: {exc}")

    if not samples or sensor_columns_ref is None:
        raise RuntimeError("No time-series samples were generated.")

    X = np.stack(samples, axis=0)
    y_labels = np.array(labels)
    group_arr = np.array(groups)
    return X, y_labels, group_arr, sensor_columns_ref


if torch is not None and nn is not None:
    class SmellLSTM(nn.Module):
        def __init__(self, n_sensors: int, hidden_size: int, n_classes: int, num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = True) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_sensors,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * (2 if bidirectional else 1), n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(lstm_out * attention_weights, dim=1)
            return self.classifier(context)


def topk_accuracy(logits: torch.Tensor, y_true: torch.Tensor, k: int = 5) -> float:
    k = min(k, logits.shape[1])
    topk = torch.topk(logits, k=k, dim=1).indices
    y_expanded = y_true.view(-1, 1).expand_as(topk)
    return float((topk == y_expanded).any(dim=1).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_true: list[int] = []
    all_logits: list[torch.Tensor] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(yb.cpu().numpy().tolist())
            all_logits.append(logits.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    y_true_tensor = torch.tensor(all_true)
    top1 = float((logits_cat.argmax(dim=1) == y_true_tensor).float().mean().item())
    top5 = topk_accuracy(logits_cat, y_true_tensor, k=5)
    return {"top1_accuracy": top1, "top5_accuracy": top5}


def main() -> None:
    if torch is None:
        raise SystemExit("PyTorch is not installed. Install it first:\nuv add torch")

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_base_files(data_root)
    if args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    cfg = PreprocessConfig(
        target_points=args.target_points,
        warmup_ratio=args.warmup_ratio,
        fill_strategy="ffill_bfill",
        normalize=True,
    )

    X, y_labels, groups, sensor_columns = build_dataset(csv_files, cfg)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    val_size_adjusted = args.val_size / max(1e-9, 1.0 - args.test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=args.seed)
    train_rel, val_rel = next(gss_val.split(X[train_val_idx], y[train_val_idx], groups=groups[train_val_idx]))

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmellLSTM(
        n_sensors=X_train.shape[2],
        hidden_size=args.hidden_size,
        n_classes=len(label_encoder.classes_),
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_top1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item())

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        mean_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:02d} | loss={mean_loss:.4f} | val={val_metrics}")

        if val_metrics["top1_accuracy"] > best_val_top1:
            best_val_top1 = val_metrics["top1_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    model_path = output_dir / "lstm_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": {
                "n_sensors": X_train.shape[2],
                "hidden_size": args.hidden_size,
                "n_classes": len(label_encoder.classes_),
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "bidirectional": args.bidirectional,
            },
        },
        model_path,
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sensor_columns": sensor_columns,
        "class_names": label_encoder.classes_.tolist(),
        "preprocess_config": asdict(cfg),
        "model": "SmellLSTM",
        "model_config": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
        },
        "test_metrics": test_metrics,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved LSTM model to:", model_path)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
