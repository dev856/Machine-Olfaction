from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib



def save_pipeline(bundle: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)



def load_pipeline(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    loaded = joblib.load(model_path)
    if not isinstance(loaded, dict):
        raise TypeError("Model artifact must be a dictionary bundle.")
    return loaded
