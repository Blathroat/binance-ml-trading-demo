from __future__ import annotations

from pathlib import Path
from typing import Any

from joblib import dump, load


def save_model_artifact(model: Any, model_path: str | Path) -> Path:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    return path


def load_model_artifact(model_path: str | Path) -> dict[str, Any]:
    artifact = load(Path(model_path))
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "feature_columns": None, "best_threshold": 0.5}

