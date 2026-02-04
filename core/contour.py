"""ContourEngine: interpolate anchors to smooth contour curve."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline


def compute_contour(
    anchors: list[list[float]],
    image_width: Optional[float] = None,
    num_samples: int = 500,
) -> list[list[float]]:
    """
    Compute smooth contour from anchor points via cubic spline.
    Anchors are (x, y) in image coords. Returns list of [x, y] points.
    """
    if len(anchors) < 2:
        return [[a[0], a[1]] for a in anchors]

    anchors = sorted(anchors, key=lambda p: p[0])
    xs = np.array([a[0] for a in anchors], dtype=float)
    ys = np.array([a[1] for a in anchors], dtype=float)

    # CubicSpline needs unique x values; if duplicates, add tiny offset
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-6

    if xs[-1] - xs[0] < 1e-9:
        return [[a[0], a[1]] for a in anchors]

    try:
        cs = CubicSpline(xs, ys)
    except Exception:
        return [[a[0], a[1]] for a in anchors]

    x_min, x_max = float(xs.min()), float(xs.max())
    if image_width is not None:
        x_max = min(x_max, image_width)
        x_min = max(x_min, 0)
    x_samples = np.linspace(x_min, x_max, num_samples)
    y_samples = cs(x_samples)
    points = [[float(x), float(y)] for x, y in zip(x_samples, y_samples)]
    return points


def save_contour(
    path: Path, points: list[list[float]], metadata: Optional[dict] = None
):
    """Save contour points to JSON."""
    data = {"points": points, **(metadata or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_contour(path: Path) -> list[list[float]]:
    """Load contour points from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data.get("points", [])
