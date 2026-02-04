"""Main window for ConcreteNet."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QMessageBox,
    QProgressDialog,
    QFileDialog,
    QMenuBar,
    QMenu,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut

from app.toolbar import Toolbar
from app.image_list import ImageList
from app.canvas import Canvas, MODE_NORMAL, MODE_ADD, MODE_SELECT
from app.stats_panel import StatsPanel
from app.dialogs import (
    get_image_paths,
    get_save_path,
    get_open_path,
    get_contour_import_path,
    get_export_path,
    get_export_png_path,
)
from core.project import ProjectManager, effective_boxes, default_anchors_from_boxes
from core.detector import create_detector
from core.spacing import SpacingAnalyzer
from core.contour import compute_contour, save_contour


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ConcreteNet - Rebar Detection")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 900)

        self.project = ProjectManager(Path(__file__).resolve().parent.parent)
        self.project.new()
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        open_act = file_menu.addAction("&Open Project...")
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open)
        save_act = file_menu.addAction("&Save Project...")
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._on_save)
        file_menu.addSeparator()
        undo_act = file_menu.addAction("&Undo")
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        undo_act.triggered.connect(self._on_undo)
        redo_act = file_menu.addAction("&Redo")
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        redo_act.triggered.connect(self._on_redo)
        file_menu.addSeparator()
        export_png_act = file_menu.addAction("Export PNG...")
        export_png_act.triggered.connect(self._on_export_png)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.toolbar = Toolbar()
        layout.addWidget(self.toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.image_list = ImageList()
        splitter.addWidget(self.image_list)

        self.canvas = Canvas()
        splitter.addWidget(self.canvas)

        self.stats_panel = StatsPanel()
        splitter.addWidget(self.stats_panel)

        layout.addWidget(splitter)

        splitter.setSizes([200, 600, 200])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._connect_signals()

    def _connect_signals(self):
        self.toolbar.upload_clicked.connect(self._on_upload)
        self.toolbar.import_contour_clicked.connect(self._on_import_contour)
        self.toolbar.toggle_boxes_changed.connect(self.canvas.set_show_boxes)
        self.toolbar.toggle_contour_changed.connect(self._on_toggle_contour)
        self.toolbar.toggle_anchors_changed.connect(self.canvas.set_show_anchors)
        self.toolbar.save_clicked.connect(self._on_save)
        self.toolbar.add_mode_clicked.connect(self._on_add_mode_clicked)
        self.toolbar.select_mode_clicked.connect(self._on_select_mode_clicked)
        self.toolbar.model_changed.connect(self._on_model_changed)

        self.image_list.selection_changed.connect(self._on_image_selected)
        self.stats_panel.export_clicked.connect(self._on_export_stats)
        self.canvas.boxes_delete_requested.connect(self._on_boxes_delete_requested)
        self.canvas.box_added.connect(self._on_box_added)
        self.canvas.anchor_removed.connect(self._on_anchor_removed)
        self.canvas.anchor_drag_started.connect(lambda _: self._push_undo())
        self.canvas.anchor_moved.connect(self._on_anchor_moved)

        # Global shortcuts
        QShortcut(QKeySequence("s"), self, self._toggle_add_mode)
        QShortcut(QKeySequence("S"), self, self._toggle_add_mode)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._on_delete_selected)
        QShortcut(
            QKeySequence(Qt.Key.Key_Left), self, lambda: self._on_move_selected(-1, 0)
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Right), self, lambda: self._on_move_selected(1, 0)
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Up), self, lambda: self._on_move_selected(0, -1)
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Down), self, lambda: self._on_move_selected(0, 1)
        )

    def _toggle_add_mode(self):
        mode = self.canvas.get_mode()
        new_mode = MODE_ADD if mode != MODE_ADD else MODE_NORMAL
        self.canvas.set_mode(new_mode)
        self._sync_mode_buttons()

    def _on_add_mode_clicked(self):
        self._toggle_add_mode()

    def _sync_mode_buttons(self):
        mode = self.canvas.get_mode()
        self.toolbar.add_btn.setChecked(mode == MODE_ADD)
        self.toolbar.select_btn.setChecked(mode == MODE_SELECT)

    def _on_select_mode_clicked(self):
        mode = self.canvas.get_mode()
        new_mode = MODE_SELECT if mode != MODE_SELECT else MODE_NORMAL
        self.canvas.set_mode(new_mode)
        self._sync_mode_buttons()

    def _on_delete_selected(self):
        self.canvas.delete_selected()

    def _on_move_selected(self, dx: int, dy: int):
        """Move selected boxes and anchors by (dx, dy) with arrow keys."""
        indices = self.canvas._selected_boxes | self.canvas._selected_anchors
        if not indices:
            return
        idx = self.image_list.get_selected_index()
        if idx < 0:
            return
        self._push_undo()
        self.project.move_pairs(idx, indices, float(dx), float(dy))
        self._refresh_canvas()

    def _on_boxes_delete_requested(self, indices: list):
        """Batch delete pairs (called from Select mode Del)."""
        idx = self.image_list.get_selected_index()
        if idx < 0:
            return
        self._push_undo()
        for i in indices:
            self.project.remove_pair(idx, i)
        self._refresh_canvas()

    def _push_undo(self):
        """Save current image state before modification."""
        idx = self.image_list.get_selected_index()
        if idx < 0:
            return
        img = self.project.get_image(idx)
        if not img:
            return
        state = {
            "image_index": idx,
            "boxes": [list(b) for b in img["boxes"]],
            "confidences": list(img["confidences"]),
            "removed_indices": list(img["removed_indices"]),
            "added_boxes": [list(b) for b in img["added_boxes"]],
            "anchors": [list(a) for a in img.get("anchors", [])],
        }
        self._undo_stack.append(state)
        self._redo_stack.clear()

    def _on_undo(self):
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        # Push current (modified) state to redo before restoring
        self._redo_stack.append(self._snapshot_image(state["image_index"]))
        img = self.project.get_image(state["image_index"])
        if img:
            img["boxes"] = [list(b) for b in state["boxes"]]
            img["confidences"] = state["confidences"]
            img["removed_indices"] = list(state["removed_indices"])
            img["added_boxes"] = [list(b) for b in state["added_boxes"]]
            img["anchors"] = [list(a) for a in state["anchors"]]
        self.image_list.setCurrentRow(state["image_index"])
        self._refresh_canvas()

    def _on_redo(self):
        if not self._redo_stack:
            return
        state = self._redo_stack.pop()
        # Push current (undone) state to undo before restoring
        self._undo_stack.append(self._snapshot_image(state["image_index"]))
        img = self.project.get_image(state["image_index"])
        if img:
            img["boxes"] = [list(b) for b in state["boxes"]]
            img["confidences"] = state["confidences"]
            img["removed_indices"] = list(state["removed_indices"])
            img["added_boxes"] = [list(b) for b in state["added_boxes"]]
            img["anchors"] = [list(a) for a in state["anchors"]]
        self.image_list.setCurrentRow(state["image_index"])
        self._refresh_canvas()

    def _snapshot_image(self, idx: int) -> dict:
        img = self.project.get_image(idx)
        if not img:
            return {
                "image_index": idx,
                "boxes": [],
                "confidences": [],
                "removed_indices": [],
                "added_boxes": [],
                "anchors": [],
            }
        return {
            "image_index": idx,
            "boxes": [list(b) for b in img["boxes"]],
            "confidences": list(img["confidences"]),
            "removed_indices": list(img["removed_indices"]),
            "added_boxes": [list(b) for b in img["added_boxes"]],
            "anchors": [list(a) for a in img.get("anchors", [])],
        }

    def _on_anchor_removed(self, anchor_index: int):
        idx = self.image_list.get_selected_index()
        if idx >= 0:
            img = self.project.get_image(idx)
            if img:
                anchors = img.get("anchors", [])
                if 0 <= anchor_index < len(anchors):
                    anchors = anchors[:anchor_index] + anchors[anchor_index + 1 :]
                    self.project.set_anchors(idx, anchors)
                    self.canvas.set_anchors(anchors)
                if self.toolbar.show_contour.isChecked():
                    self._update_contour_for_current_image()

    def _on_anchor_moved(self, anchor_index: int, x: float, y: float):
        idx = self.image_list.get_selected_index()
        if idx >= 0:
            img = self.project.get_image(idx)
            if img:
                anchors = img.get("anchors", [])
                if 0 <= anchor_index < len(anchors):
                    anchors[anchor_index] = [x, y]
                    self.project.set_anchors(idx, anchors)
                self._refresh_canvas()

    def _on_upload(self):
        paths = get_image_paths(self)
        if not paths:
            return

        model_type = self.toolbar.model_combo.currentText().lower()
        self.project.data["model_type"] = model_type

        try:
            detector = create_detector(model_type=model_type)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Detection Error",
                f"Could not load model: {e}\n\nEnsure the model weights exist at runs/detect/.",
            )
            return

        progress = QProgressDialog("Running detection...", None, 0, len(paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        for i, path in enumerate(paths):
            progress.setValue(i)
            progress.setLabelText(f"Processing {path.name}...")
            if progress.wasCanceled():
                break
            try:
                result = detector.detect(str(path))
                self.project.add_image(
                    str(path),
                    result["boxes"],
                    result["confidences"],
                )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Detection Failed",
                    f"Failed to process {path.name}: {e}",
                )

        progress.setValue(len(paths))

        paths_str = [img["path"] for img in self.project.images]
        self.image_list.set_images(paths_str)
        if paths_str:
            self.image_list.setCurrentRow(len(paths_str) - 1)
            self._on_image_selected(len(paths_str) - 1)

    def _on_image_selected(self, index: int):
        if index < 0:
            self.canvas.load_image(None)
            self.canvas.set_boxes([])
            self.canvas.set_anchors([])
            self.canvas.set_contour_points([])
            self.stats_panel.set_stats({})
            return

        img = self.project.get_image(index)
        if not img:
            return

        path = Path(img["path"])
        if not path.exists():
            QMessageBox.warning(
                self,
                "Image Not Found",
                f"Image not found: {path}\nIt may have been moved or deleted.",
            )
            self.canvas.load_image(None)
            self.canvas.set_boxes([])
            self.canvas.set_anchors([])
            self.canvas.set_contour_points([])
            self.stats_panel.set_stats({})
            return

        self.canvas.load_image(img["path"])
        boxes = effective_boxes(img)
        anchors = img.get("anchors") or default_anchors_from_boxes(boxes)
        stats = SpacingAnalyzer().analyze(boxes, anchors)
        stats["rebar_count"] = len(boxes)
        self.canvas.set_boxes(
            boxes, outlier_indices=stats.get("outlier_box_indices", set())
        )
        self.canvas.set_anchors(img.get("anchors", []))
        self._update_contour_for_current_image()

        try:
            import cv2

            im = cv2.imread(img["path"])
            size = (im.shape[1], im.shape[0]) if im is not None else (0, 0)
        except Exception:
            size = (0, 0)
        self.stats_panel.set_stats(stats, image_path=img["path"], image_size=size)

    def _on_import_contour(self):
        path = get_contour_import_path(self)
        if not path or not path.exists():
            return
        try:
            if path.suffix.lower() == ".png":
                from PyQt6.QtGui import QPixmap

                pix = QPixmap(str(path))
                if not pix.isNull():
                    self.canvas.set_contour_overlay(pix, opacity=0.2)
                    self.canvas.set_show_contour(True)
                    self.toolbar.show_contour.setChecked(True)
                else:
                    QMessageBox.warning(self, "Import Error", "Could not load PNG.")
            else:
                import json

                with open(path) as f:
                    data = json.load(f)
                points = data.get("points", [])
                if points:
                    self.canvas.set_contour_points(points)
                self.canvas.set_contour_overlay(None)
                self.toolbar.show_contour.setChecked(True)
                self.canvas.set_show_contour(True)
        except Exception as e:
            QMessageBox.warning(self, "Import Error", f"Could not load contour: {e}")

    def _on_open(self):
        path = get_open_path(self)
        if path and path.exists():
            try:
                self.project.load(path)
                self.toolbar.set_model(self.project.model_type)
                paths_str = [img["path"] for img in self.project.images]
                self.image_list.set_images(paths_str)
                if paths_str:
                    self.image_list.setCurrentRow(0)
                    self._on_image_selected(0)
                QMessageBox.information(self, "Opened", f"Project loaded from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Open Error", f"Could not load project: {e}")

    def _on_save(self):
        path = get_save_path(self, "project.json")
        if path:
            self.project.save(path)
            # Export contour maps
            export_dir = path.parent / "contour_exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(self.project.images):
                anchors = img.get("anchors", [])
                if len(anchors) >= 2:
                    try:
                        import cv2

                        im = cv2.imread(img["path"])
                        w = im.shape[1] if im is not None else None
                    except Exception:
                        w = None
                    points = compute_contour(anchors, image_width=w)
                    stem = Path(img["path"]).stem
                    contour_path = export_dir / f"{stem}_contour.json"
                    save_contour(contour_path, points, {"source": img["path"]})
            QMessageBox.information(
                self,
                "Saved",
                f"Project saved to {path}\nContour maps exported to {export_dir}/",
            )

    def _on_model_changed(self, model_type: str):
        self.project.data["model_type"] = model_type

    def _on_box_added(self, box: list[float]):
        idx = self.image_list.get_selected_index()
        if idx >= 0:
            self._push_undo()
            self.project.add_box(idx, box)
            img = self.project.get_image(idx)
            if img:
                x1, y1, x2, y2 = box
                anchor = [(x1 + x2) / 2, y1]
                anchors = img.get("anchors", []) + [anchor]
                self.project.set_anchors(idx, anchors)
            self._refresh_canvas()

    def _refresh_canvas(self):
        """Update annotations only (preserves zoom/pan)."""
        idx = self.image_list.get_selected_index()
        if idx < 0:
            return
        img = self.project.get_image(idx)
        if not img:
            return
        boxes = effective_boxes(img)
        anchors = img.get("anchors") or default_anchors_from_boxes(boxes)
        stats = SpacingAnalyzer().analyze(boxes, anchors)
        stats["rebar_count"] = len(boxes)
        self.canvas.set_boxes(
            boxes, outlier_indices=stats.get("outlier_box_indices", set())
        )
        self.canvas.set_anchors(img.get("anchors", []))
        self._update_contour_for_current_image()
        try:
            import cv2

            im = cv2.imread(img["path"])
            size = (im.shape[1], im.shape[0]) if im is not None else (0, 0)
        except Exception:
            size = (0, 0)
        self.stats_panel.set_stats(stats, image_path=img["path"], image_size=size)
        self.canvas.update()

    def _update_contour_for_current_image(self):
        """Compute contour from current image's anchors and set on canvas."""
        idx = self.image_list.get_selected_index()
        if idx < 0:
            self.canvas.set_contour_points([])
            return
        img = self.project.get_image(idx)
        if not img or not img.get("anchors"):
            self.canvas.set_contour_points([])
            return
        # Get image dimensions for interpolation range
        try:
            import cv2

            im = cv2.imread(img["path"])
            w = im.shape[1] if im is not None else None
        except Exception:
            w = None
        points = compute_contour(img["anchors"], image_width=w)
        self.canvas.set_contour_points(points)

    def _on_toggle_contour(self, show: bool):
        self.canvas.set_show_contour(show)
        if show:
            self._update_contour_for_current_image()

    def _on_export_stats(self):
        path = get_export_path(self, "stats.csv")
        if not path:
            return
        idx = self.image_list.get_selected_index()
        if idx < 0:
            QMessageBox.warning(self, "Export", "No image selected.")
            return
        img = self.project.get_image(idx)
        if not img:
            return
        boxes = effective_boxes(img)
        anchors = img.get("anchors") or default_anchors_from_boxes(boxes)
        stats = SpacingAnalyzer().analyze(boxes, anchors)
        stats["rebar_count"] = len(boxes)
        stats["image_name"] = Path(img["path"]).name
        try:
            import cv2

            im = cv2.imread(img["path"])
            stats["image_width"] = im.shape[1] if im is not None else 0
            stats["image_height"] = im.shape[0] if im is not None else 0
            stats["image_path"] = img["path"]
        except Exception:
            stats["image_width"] = stats["image_height"] = 0
            stats["image_path"] = img["path"]
        # Box coordinates (x1,y1,x2,y2), widths, and anchors for CSV/JSON
        stats["boxes"] = [
            [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)]
            for b in boxes
        ]
        stats["boxes_widths"] = [round(b[2] - b[0], 2) for b in boxes]
        stats["anchors"] = [[round(a[0], 2), round(a[1], 2)] for a in anchors]
        obi = stats.get("outlier_box_indices", set())
        exportable = {k: v for k, v in stats.items() if k != "outlier_box_indices"}
        exportable["outlier_box_indices"] = list(obi)
        if path.suffix.lower() == ".json":
            import json

            with open(path, "w") as f:
                json.dump(exportable, f, indent=2)
        else:
            import csv

            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                keys = list(exportable.keys())
                w.writerow(keys)

                def _csv_val(v):
                    if isinstance(v, list):
                        if v and isinstance(v[0], (list, tuple)):
                            return "|".join(",".join(str(y) for y in x) for x in v)
                        return ",".join(str(x) for x in v)
                    if isinstance(v, set):
                        return ",".join(str(x) for x in v)
                    return v

                row = [_csv_val(exportable.get(k, "")) for k in keys]
                w.writerow(row)
        QMessageBox.information(self, "Exported", f"Stats saved to {path}")

    def _on_export_png(self):
        path = get_export_png_path(self, "export.png")
        if not path:
            return
        if self.canvas._pixmap is None:
            QMessageBox.warning(self, "Export", "No image loaded.")
            return
        reply = QMessageBox.question(
            self,
            "Export PNG",
            "Export annotations only (transparent background)?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return
        annot_only = reply == QMessageBox.StandardButton.Yes
        pix = self.canvas.render_to_pixmap(annotations_only=annot_only)
        if pix and not pix.isNull():
            pix.save(str(path))
            QMessageBox.information(self, "Exported", f"PNG saved to {path}")
        else:
            QMessageBox.warning(self, "Export", "Could not render image.")
