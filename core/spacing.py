"""SpacingAnalyzer: median distance, outlier detection using anchor points."""

import numpy as np


def _distances_from_positions(positions: list[tuple[float, float]]) -> list[float]:
    """Pairwise horizontal distances between consecutive positions (sorted by x)."""
    if len(positions) < 2:
        return []
    sorted_mp = sorted(positions, key=lambda p: p[0])
    dists = []
    for i in range(len(sorted_mp) - 1):
        dx = sorted_mp[i + 1][0] - sorted_mp[i][0]
        dists.append(dx)
    return dists


class SpacingAnalyzer:
    """Compute median spacing and flag outliers (>2 SD from median)."""

    def __init__(self, outlier_sigma: float = 2.0):
        self.outlier_sigma = outlier_sigma

    def analyze(
        self, boxes: list[list[float]], anchors: list[list[float]] | None = None
    ) -> dict:
        """Use anchor points (x,y) for spacing. Fall back to box centers if anchors missing."""
        if anchors is not None and len(anchors) == len(boxes):
            positions = [(a[0], a[1]) for a in anchors]
        else:
            positions = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        if len(positions) < 2:
            return {
                "median": 0.0,
                "outlier_count": 0,
                "outlier_box_count": 0,
                "distances": [],
                "outliers": [],
                "outlier_box_indices": set(),
            }
        sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i][0])
        distances = _distances_from_positions(positions)
        arr = np.array(distances)
        median = float(np.median(arr))
        std = float(np.std(arr)) if len(arr) > 1 else 0.0
        # Use at least 6 px so gaps within ~6 px of median are not flagged
        threshold = max(self.outlier_sigma * std, 6.0)
        outlier_box_indices = set()
        for i, d in enumerate(distances):
            if abs(d - median) > threshold:
                outlier_box_indices.add(sorted_indices[i])
                outlier_box_indices.add(sorted_indices[i + 1])
        outliers = [d for d in distances if abs(d - median) > threshold]
        return {
            "median": median,
            "outlier_count": len(outliers),
            "outlier_box_count": len(outlier_box_indices),
            "distances": distances,
            "outliers": outliers,
            "outlier_box_indices": outlier_box_indices,
        }
