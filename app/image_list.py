"""Sidebar list of images."""

from pathlib import Path

from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from PyQt6.QtCore import pyqtSignal, Qt


class ImageList(QListWidget):
    """List of images in the project. Emits selection changes."""

    selection_changed = pyqtSignal(int)  # index or -1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setMaximumWidth(250)
        self.currentRowChanged.connect(self._on_row_changed)

    def _on_row_changed(self, row: int):
        self.selection_changed.emit(row if row >= 0 else -1)

    def set_images(self, paths: list[str]):
        """Populate list from image paths."""
        self.clear()
        for path in paths:
            name = Path(path).name
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.addItem(item)

    def get_selected_index(self) -> int:
        return self.currentRow()

    def get_selected_path(self) -> str | None:
        item = self.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None
