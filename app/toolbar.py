"""Toolbar with Upload, model selector, and action buttons."""

from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QCheckBox,
)
from PyQt6.QtCore import pyqtSignal


class Toolbar(QWidget):
    """Main toolbar."""

    upload_clicked = pyqtSignal()
    import_contour_clicked = pyqtSignal()
    toggle_boxes_changed = pyqtSignal(bool)
    toggle_contour_changed = pyqtSignal(bool)
    toggle_anchors_changed = pyqtSignal(bool)
    save_clicked = pyqtSignal()
    add_mode_clicked = pyqtSignal()
    select_mode_clicked = pyqtSignal()
    model_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["GP8000", "GSSI"])
        self.model_combo.setCurrentText("GP8000")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.model_combo)

        upload_btn = QPushButton("Upload Images")
        upload_btn.setToolTip("Add images and run rebar detection")
        upload_btn.clicked.connect(self.upload_clicked.emit)
        layout.addWidget(upload_btn)

        import_contour_btn = QPushButton("Import Contour")
        import_contour_btn.setToolTip(
            "Load contour map from JSON or PNG (PNG shown at 20% opacity)"
        )
        import_contour_btn.clicked.connect(self.import_contour_clicked.emit)
        layout.addWidget(import_contour_btn)

        self.show_boxes = QCheckBox("Show Boxes")
        self.show_boxes.setToolTip("Toggle bounding box visibility")
        self.show_boxes.setChecked(True)
        self.show_boxes.stateChanged.connect(
            lambda s: self.toggle_boxes_changed.emit(s == 2)
        )
        layout.addWidget(self.show_boxes)

        self.show_contour = QCheckBox("Show Contour")
        self.show_contour.setToolTip("Show interpolated contour from anchors")
        self.show_contour.setChecked(False)
        self.show_contour.stateChanged.connect(
            lambda s: self.toggle_contour_changed.emit(s == 2)
        )
        layout.addWidget(self.show_contour)

        self.show_anchors = QCheckBox("Show Anchors")
        self.show_anchors.setToolTip("Toggle anchor point visibility")
        self.show_anchors.setChecked(True)
        self.show_anchors.stateChanged.connect(
            lambda s: self.toggle_anchors_changed.emit(s == 2)
        )
        layout.addWidget(self.show_anchors)

        add_btn = QPushButton("Add Box (S)")
        add_btn.setToolTip("Add rebar - click twice to draw a box (live preview)")
        add_btn.setCheckable(True)
        add_btn.clicked.connect(self.add_mode_clicked.emit)
        layout.addWidget(add_btn)
        self.add_btn = add_btn

        select_btn = QPushButton("Select")
        select_btn.setToolTip("Drag to select multiple boxes/anchors, Del to delete")
        select_btn.setCheckable(True)
        select_btn.clicked.connect(self.select_mode_clicked.emit)
        layout.addWidget(select_btn)
        self.select_btn = select_btn

        save_btn = QPushButton("Save Changes")
        save_btn.setToolTip("Save project and export contour maps")
        save_btn.clicked.connect(self.save_clicked.emit)
        layout.addWidget(save_btn)

        layout.addStretch()
        hint = QLabel("Del: Remove | Arrows: Move selected | Ctrl+Z/Y: Undo/Redo")
        hint.setToolTip(
            "S: Add new rebar boxes | Select: Drag to select | Del: Remove selected"
        )
        layout.addWidget(hint)

    def _on_model_changed(self, text: str):
        self.model_changed.emit(text.lower())

    def set_model(self, model_type: str):
        """Set model combo from 'gp8000' or 'gssi'."""
        label = model_type.upper()
        idx = self.model_combo.findText(label)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
