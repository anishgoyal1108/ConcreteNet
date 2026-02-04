"""Upload dialog and file pickers."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def get_image_paths(parent=None) -> list[Path]:
    """Open file dialog to select multiple images."""
    paths, _ = QFileDialog.getOpenFileNames(
        parent,
        "Select Images",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)",
    )
    return [Path(p) for p in paths if Path(p).suffix.lower() in IMAGE_EXTENSIONS]


def get_save_path(parent=None, default_name: str = "project.json") -> Path | None:
    """Open save dialog for project file."""
    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Project",
        default_name,
        "JSON (*.json);;All Files (*)",
    )
    return Path(path) if path else None


def get_open_path(parent=None) -> Path | None:
    """Open file dialog for project file."""
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Open Project",
        "",
        "JSON (*.json);;All Files (*)",
    )
    return Path(path) if path else None


def get_contour_import_path(parent=None) -> Path | None:
    """Open file dialog to import contour map (JSON or PNG)."""
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Import Contour Map",
        "",
        "JSON (*.json);;PNG (*.png);;All Files (*)",
    )
    return Path(path) if path else None


def get_export_png_path(parent=None, default_name: str = "export.png") -> Path | None:
    """Open save dialog for PNG export."""
    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Export PNG",
        default_name,
        "PNG (*.png);;All Files (*)",
    )
    return Path(path) if path else None


def get_export_path(parent=None, default_name: str = "stats.csv") -> Path | None:
    """Open save dialog for stats export."""
    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Export Stats",
        default_name,
        "CSV (*.csv);;JSON (*.json);;All Files (*)",
    )
    return Path(path) if path else None
