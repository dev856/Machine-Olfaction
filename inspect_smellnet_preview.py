import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Quick SmellNet CSV structure preview")
    parser.add_argument("--data-dir", required=True, help="Path to SmellNet root folder")
    parser.add_argument(
        "--subset",
        default=None,
        choices=["base_data", "mixture_data", "gcms_data", "gcms_processed", "text_data"],
        help="Optional subset folder under data-dir",
    )
    parser.add_argument("--max-files", type=int, default=10, help="How many CSV files to preview")
    parser.add_argument("--head", type=int, default=5, help="How many top rows per file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Dataset path not found or not a folder: {data_dir}")
        return

    scan_root = data_dir / args.subset if args.subset else data_dir
    if not scan_root.exists() or not scan_root.is_dir():
        print(f"Scan path not found or not a folder: {scan_root}")
        return

    csv_files = [
        path
        for path in sorted(scan_root.rglob("*.csv"))
        if ".cache" not in path.parts
    ]
    print(f"Dataset root: {data_dir}")
    print(f"Scan root: {scan_root}")
    print(f"Total CSV files found: {len(csv_files)}")

    if not csv_files:
        return

    for i, csv_path in enumerate(csv_files[: args.max_files], start=1):
        print("\n" + "=" * 90)
        rel_path = csv_path.relative_to(scan_root)
        print(f"[{i}] File: {rel_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Could not read file: {exc}")
            continue

        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print("First rows:")
        print(df.head(args.head).to_string(index=False))


if __name__ == "__main__":
    main()