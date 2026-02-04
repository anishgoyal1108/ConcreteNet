"""Thin wrapper around detect_rebars.RebarDetector with model path resolution."""

from pathlib import Path
from typing import Literal

# Import from project root (detect_rebars is at root)
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detect_rebars import RebarDetector, DEFAULT_MODELS

ModelType = Literal["gp8000", "gssi"]


def get_model_path(model_type: ModelType, project_root: Path | None = None) -> Path:
    """Resolve model path relative to project root."""
    root = project_root or PROJECT_ROOT
    rel = DEFAULT_MODELS.get(model_type)
    if not rel:
        raise ValueError(f"Unknown model type: {model_type}")
    return root / rel


def create_detector(
    model_type: ModelType = "gp8000",
    model_path: str | Path | None = None,
    confidence: float = 0.25,
    project_root: Path | None = None,
) -> RebarDetector:
    """Create a RebarDetector instance."""
    root = project_root or PROJECT_ROOT
    if model_path is None:
        path = get_model_path(model_type, root)
    else:
        path = Path(model_path)
    return RebarDetector(str(path), confidence_threshold=confidence)
