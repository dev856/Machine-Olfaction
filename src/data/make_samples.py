"""Copy a few real SmellNet CSV files into `data/samples` for the demo app.

The script does not synthesize sensor readings. It only copies lightweight
examples from a local SmellNet checkout so Streamlit can be tested without
asking every user to browse the full dataset tree.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create small demo CSV samples for Streamlit users.")
    parser.add_argument("--data-root", default="data/raw/SmellNet")
    parser.add_argument("--output-dir", default="data/samples")
    parser.add_argument("--base-class-limit", type=int, default=8, help="Number of base classes to sample")
    parser.add_argument("--mixture-count", type=int, default=4, help="Number of mixture files to sample")
    return parser.parse_args()



def label_from_filename(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return path.stem



def collect_base_samples(base_dir: Path, class_limit: int) -> list[Path]:
    """Pick one base-data trial from the first N labels in sorted order."""

    files = [p for p in sorted(base_dir.rglob("*.csv")) if ".cache" not in p.parts]
    by_label: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_label[label_from_filename(path)].append(path)

    selected: list[Path] = []
    for label in sorted(by_label.keys())[:class_limit]:
        selected.append(by_label[label][0])
    return selected



def collect_mixture_samples(mix_dir: Path, count: int) -> list[Path]:
    """Pick mixture examples when SmellNet's mixture index file is available."""

    index_file = mix_dir / "test_index_seen.csv"
    if not index_file.exists():
        return []

    df = pd.read_csv(index_file)
    selected: list[Path] = []

    for raw_path in df["filepath"].tolist():
        suffix = str(raw_path).split("Data0819/")[-1]
        local_path = mix_dir / suffix
        if local_path.exists():
            selected.append(local_path)
        if len(selected) >= count:
            break

    return selected



def copy_files(paths: list[Path], source_root: Path, dest_root: Path) -> None:
    for src in paths:
        rel = src.relative_to(source_root)
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)



def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)

    base_dir = data_root / "base_data"
    mix_dir = data_root / "mixture_data"

    base_samples = collect_base_samples(base_dir, class_limit=args.base_class_limit)
    mixture_samples = collect_mixture_samples(mix_dir, count=args.mixture_count)

    copy_files(base_samples, data_root, out_root)
    copy_files(mixture_samples, data_root, out_root)

    print(f"Copied {len(base_samples)} base samples and {len(mixture_samples)} mixture samples to {out_root}")


if __name__ == "__main__":
    main()
