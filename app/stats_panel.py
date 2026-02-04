"""Stats panel - image stats, spacing, outliers, export button."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
)
from PyQt6.QtCore import pyqtSignal


class StatsPanel(QWidget):
    """Shows image and spacing statistics and export button."""

    export_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(220)
        layout = QVBoxLayout(self)

        group = QGroupBox("Stats")
        gl = QVBoxLayout(group)
        self.labels = {}
        for key in [
            "image_name",
            "image_size",
            "rebar_count",
            "median_spacing",
            "outlier_count",
            "outlier_gaps",
        ]:
            lbl = QLabel("--")
            self.labels[key] = lbl
            gl.addWidget(lbl)
        self.export_btn = QPushButton("Export Stats")
        self.export_btn.setToolTip("Export all stats to CSV or JSON")
        self.export_btn.clicked.connect(self.export_clicked.emit)
        gl.addWidget(self.export_btn)
        layout.addWidget(group)
        layout.addStretch()

        self._current_stats = {}
        self._image_path = ""
        self._image_size = (0, 0)

    def set_stats(
        self,
        stats: dict,
        image_path: str = "",
        image_size: tuple = (0, 0),
    ):
        self._current_stats = stats or {}
        self._image_path = image_path
        self._image_size = image_size

        if not stats:
            for k in self.labels:
                self.labels[k].setText("--")
            return

        from pathlib import Path

        if image_path:
            self.labels["image_name"].setText(f"Image: {Path(image_path).name}")
        else:
            self.labels["image_name"].setText("Image: --")

        if image_size and image_size[0] and image_size[1]:
            self.labels["image_size"].setText(
                f"Size: {image_size[0]} x {image_size[1]}"
            )
        else:
            self.labels["image_size"].setText("Size: --")

        rebar = stats.get("rebar_count", 0)
        if not rebar and "distances" in stats:
            rebar = len(stats["distances"]) + 1
        self.labels["rebar_count"].setText(f"Rebars: {rebar}")

        med = stats.get("median")
        self.labels["median_spacing"].setText(
            f"Median spacing: {med:.1f} px" if med is not None else "Median spacing: --"
        )

        out_gaps = stats.get("outlier_count", 0)
        out_boxes = stats.get(
            "outlier_box_count", len(stats.get("outlier_box_indices", []))
        )
        self.labels["outlier_count"].setText(
            f"Outlier gaps: {out_gaps} ({out_boxes} boxes)"
        )

        outliers = stats.get("outliers", [])
        if outliers:
            s = ", ".join(f"{x:.0f}" for x in outliers[:5])
            if len(outliers) > 5:
                s += "..."
            self.labels["outlier_gaps"].setText(f"Outlier gaps (px): {s}")
        else:
            self.labels["outlier_gaps"].setText("Outlier gaps: --")
