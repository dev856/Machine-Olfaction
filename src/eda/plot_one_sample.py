from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocess import identify_time_column, infer_sensor_columns



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one smell sample as sensor curves.")
    parser.add_argument("--csv-path", required=True, help="Path to one sensor CSV file")
    parser.add_argument("--max-sensors", type=int, default=12, help="Maximum number of sensors to plot")
    parser.add_argument("--figsize", type=float, nargs=2, default=(11, 6), help="Figure size, e.g., --figsize 11 6")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV path not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    time_col = identify_time_column(df)
    sensor_cols = infer_sensor_columns(df, time_column=time_col)

    if not sensor_cols:
        raise ValueError("No sensor columns detected. Check CSV schema.")

    if len(sensor_cols) > args.max_sensors:
        sensor_cols = sensor_cols[: args.max_sensors]

    if time_col and time_col in df.columns:
        time_axis = df[time_col]
        x_label = time_col
    else:
        time_axis = pd.RangeIndex(start=0, stop=len(df), step=1)
        x_label = "sample_index"

    print(f"Time axis: {time_col if time_col else 'row index'}")
    print(f"Sensor columns: {sensor_cols}")

    plt.figure(figsize=tuple(args.figsize))
    for col in sensor_cols:
        plt.plot(time_axis, df[col], label=col, linewidth=1.2, alpha=0.85)

    plt.title(f"Sensor Curves - {csv_path.name}")
    plt.xlabel(x_label)
    plt.ylabel("Sensor Reading")
    plt.legend(loc="best", ncol=2, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
