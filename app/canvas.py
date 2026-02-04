"""Interactive canvas for image display with zoom/pan, boxes, anchors."""

from pathlib import Path

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QColor,
    QPen,
    QBrush,
    QWheelEvent,
    QMouseEvent,
    QCursor,
)

import cv2


MODE_NORMAL = "normal"
MODE_ADD = "add"
MODE_SELECT = "select"


class Canvas(QWidget):
    """Displays image with optional bounding boxes, anchors, zoom/pan."""

    boxes_delete_requested = pyqtSignal(list)  # list of pair indices to delete
    box_added = pyqtSignal(list)  # [x1,y1,x2,y2]
    anchor_added = pyqtSignal(float, float)
    anchor_removed = pyqtSignal(int)
    anchor_moved = pyqtSignal(int, float, float)
    anchor_drag_started = pyqtSignal(int)
    boxes_selected = pyqtSignal(set)  # box indices
    anchors_selected = pyqtSignal(set)  # anchor indices

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self._mode = MODE_NORMAL
        self._add_first_point: tuple[float, float] | None = None
        self._add_preview_point: tuple[float, float] | None = None
        self._selected_anchor: int | None = None
        self._selected_boxes: set[int] = set()
        self._selected_anchors: set[int] = set()
        self._selection_rect_start: tuple[float, float] | None = None
        self._selection_rect_end: tuple[float, float] | None = None
        self._anchor_drag_start: tuple[float, float] | None = None
        self._hovered_box: int | None = None
        self._hovered_anchor: int | None = None
        self._pixmap: QPixmap | None = None
        self._image_path: str | None = None
        self._boxes: list[list[float]] = []
        self._confidences: list[float] = []
        self._outlier_indices: set[int] = set()
        self._anchors: list[list[float]] = []
        self._show_boxes = True
        self._show_contour = False
        self._show_anchors = True
        self._contour_points: list[list[float]] = []
        self._contour_overlay_pixmap: QPixmap | None = None
        self._contour_overlay_opacity = 0.2
        self._scale = 1.0
        self._user_zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._draw_offset = (0, 0)
        self._last_pan_pos: QPointF | None = None

    def set_mode(self, mode: str):
        self._mode = mode
        self._add_first_point = None
        self._add_preview_point = None
        self._selection_rect_start = None
        self._selection_rect_end = None
        if mode != MODE_SELECT:
            self._selected_boxes.clear()
            self._selected_anchors.clear()
        if mode == MODE_ADD:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        elif mode == MODE_SELECT:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.update()

    def get_mode(self) -> str:
        return self._mode

    def load_image(self, path: str | Path | None):
        if path is None:
            self._pixmap = None
            self._image_path = None
            self._contour_overlay_pixmap = None
        else:
            path = Path(path)
            if not path.exists():
                self._pixmap = None
                self._image_path = None
            else:
                img = cv2.imread(str(path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img.shape[:2]
                    bytes_per_line = w * 3
                    qimg = QImage(
                        img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                    )
                    self._pixmap = QPixmap.fromImage(qimg)
                    self._image_path = str(path)
                    self._user_zoom = 1.0
                    self._pan_offset = QPointF(0, 0)
                else:
                    self._pixmap = None
                    self._image_path = None
        self.update()

    def set_boxes(
        self,
        boxes: list[list[float]],
        confidences: list[float] | None = None,
        outlier_indices: set[int] | None = None,
    ):
        self._boxes = [list(b) for b in boxes]
        self._confidences = confidences or []
        self._outlier_indices = outlier_indices or set()
        self._selected_boxes = {i for i in self._selected_boxes if i < len(self._boxes)}

    def set_anchors(self, anchors: list[list[float]]):
        self._anchors = [list(a) for a in anchors]
        self._selected_anchors = {
            i for i in self._selected_anchors if i < len(self._anchors)
        }

    def set_contour_points(self, points: list[list[float]]):
        self._contour_points = [list(p) for p in points]

    def set_contour_overlay(self, pixmap: QPixmap | None, opacity: float = 0.2):
        """Set PNG overlay for contour (e.g. from imported contour map)."""
        self._contour_overlay_pixmap = pixmap
        self._contour_overlay_opacity = opacity
        self.update()

    def set_show_boxes(self, show: bool):
        self._show_boxes = show
        self.update()

    def set_show_contour(self, show: bool):
        self._show_contour = show
        self.update()

    def set_show_anchors(self, show: bool):
        self._show_anchors = show
        self.update()

    def _base_scale_and_offset(self) -> tuple[float, float, float]:
        """Return (scale, draw_x, draw_y) for fitting image in widget."""
        if self._pixmap is None:
            return 1.0, 0, 0
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        base_scale = min(
            scaled.width() / self._pixmap.width(),
            scaled.height() / self._pixmap.height(),
        )
        draw_x = (self.width() - scaled.width()) / 2
        draw_y = (self.height() - scaled.height()) / 2
        return base_scale, draw_x, draw_y

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._pixmap is None:
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded"
            )
            return

        base_scale, draw_x, draw_y = self._base_scale_and_offset()
        self._scale = base_scale * self._user_zoom
        draw_x += self._pan_offset.x()
        draw_y += self._pan_offset.y()
        self._draw_offset = (draw_x, draw_y)

        scaled_w = self._pixmap.width() * self._scale
        scaled_h = self._pixmap.height() * self._scale
        painter.drawPixmap(
            int(draw_x),
            int(draw_y),
            int(scaled_w),
            int(scaled_h),
            self._pixmap,
        )

        if (
            self._show_contour
            and self._contour_overlay_pixmap
            and not self._contour_overlay_pixmap.isNull()
        ):
            painter.setOpacity(self._contour_overlay_opacity)
            overlay_scaled = self._contour_overlay_pixmap.scaled(
                int(scaled_w),
                int(scaled_h),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(int(draw_x), int(draw_y), overlay_scaled)
            painter.setOpacity(1.0)

        if self._scale <= 0:
            return

        def to_widget(ix: float, iy: float) -> tuple[float, float]:
            wx = draw_x + ix * self._scale
            wy = draw_y + iy * self._scale
            return (wx, wy)

        if self._show_boxes and self._boxes:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i, box in enumerate(self._boxes):
                x1, y1, x2, y2 = box
                wx1, wy1 = to_widget(x1, y1)
                wx2, wy2 = to_widget(x2, y2)
                if i == self._hovered_box:
                    painter.setOpacity(0.5)
                if i in self._outlier_indices:
                    pen = QPen(QColor(255, 0, 0), 3)
                else:
                    pen = QPen(QColor(0, 255, 0), 2)
                painter.setPen(pen)
                if i in self._selected_boxes:
                    painter.setBrush(QBrush(QColor(0, 255, 0, 40)))
                painter.drawRect(int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                if i == self._hovered_box:
                    painter.setOpacity(1.0)

        if self._show_contour and len(self._contour_points) >= 2:
            pen = QPen(QColor(255, 100, 100, 180), 2)
            painter.setPen(pen)
            for i in range(len(self._contour_points) - 1):
                p1 = self._contour_points[i]
                p2 = self._contour_points[i + 1]
                w1 = to_widget(p1[0], p1[1])
                w2 = to_widget(p2[0], p2[1])
                painter.drawLine(int(w1[0]), int(w1[1]), int(w2[0]), int(w2[1]))

        if self._show_anchors and self._anchors:
            r = 6
            for i, (ax, ay) in enumerate(self._anchors):
                wx, wy = to_widget(ax, ay)
                alpha = (
                    200
                    if (i == self._hovered_anchor or i in self._selected_anchors)
                    else 255
                )
                if i == self._selected_anchor or i in self._selected_anchors:
                    painter.setBrush(QBrush(QColor(255, 100, 0, alpha)))
                    painter.setPen(QPen(QColor(255, 50, 0), 2))
                else:
                    painter.setBrush(QBrush(QColor(255, 200, 0, int(alpha * 0.9))))
                    painter.setPen(QPen(QColor(255, 150, 0), 1))
                painter.drawEllipse(int(wx - r), int(wy - r), r * 2, r * 2)

        if self._add_first_point and self._add_preview_point:
            x1, y1 = self._add_first_point
            x2, y2 = self._add_preview_point
            wx1, wy1 = to_widget(x1, y1)
            wx2, wy2 = to_widget(x2, y2)
            x_min, x_max = min(wx1, wx2), max(wx1, wx2)
            y_min, y_max = min(wy1, wy2), max(wy1, wy2)
            painter.setPen(QPen(QColor(0, 255, 255), 2))
            painter.setBrush(QBrush(QColor(0, 255, 255, 30)))
            painter.drawRect(
                int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
            )

        if self._selection_rect_start and self._selection_rect_end:
            sx, sy = self._selection_rect_start
            ex, ey = self._selection_rect_end
            painter.setPen(QPen(QColor(100, 150, 255), 2))
            painter.setBrush(QBrush(QColor(100, 150, 255, 50)))
            painter.drawRect(
                int(min(sx, ex)), int(min(sy, ey)), int(abs(ex - sx)), int(abs(ey - sy))
            )

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        base_scale, draw_x, draw_y = self._base_scale_and_offset()
        self._scale = base_scale * self._user_zoom
        ox = draw_x + self._pan_offset.x()
        oy = draw_y + self._pan_offset.y()
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 1 / 1.1
        pos = event.position()
        ix = (pos.x() - ox) / self._scale if self._scale > 0 else 0
        iy = (pos.y() - oy) / self._scale if self._scale > 0 else 0
        self._user_zoom *= factor
        self._user_zoom = max(0.2, min(10.0, self._user_zoom))
        self._scale = base_scale * self._user_zoom
        new_wx = ox + ix * self._scale
        new_wy = oy + iy * self._scale
        self._pan_offset += QPointF(pos.x() - new_wx, pos.y() - new_wy)
        self.update()

    def _box_at_point(self, ix: float, iy: float) -> int | None:
        """Return box index if (ix,iy) is inside a box, else None."""
        for i, box in enumerate(self._boxes):
            x1, y1, x2, y2 = box
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return i
        return None

    def _anchor_at_point(
        self, ix: float, iy: float, tolerance: float = 15
    ) -> int | None:
        """Return anchor index if (ix,iy) is near an anchor (image coords)."""
        for i, (ax, ay) in enumerate(self._anchors):
            if abs(ix - ax) <= tolerance and abs(iy - ay) <= tolerance:
                return i
        return None

    def _anchor_at_widget(
        self, wx: float, wy: float, tolerance: float = 12
    ) -> int | None:
        """Return anchor index if (wx,wy) is near an anchor (widget coords)."""
        for i, (ax, ay) in enumerate(self._anchors):
            twx, twy = self.image_to_widget(ax, ay)
            if (wx - twx) ** 2 + (wy - twy) ** 2 <= tolerance**2:
                return i
        return None

    def _boxes_in_rect(self, x1: float, y1: float, x2: float, y2: float) -> set[int]:
        """Return box indices whose center falls inside rect (image coords)."""
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        result = set()
        for i, box in enumerate(self._boxes):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                result.add(i)
        return result

    def _anchors_in_rect(self, x1: float, y1: float, x2: float, y2: float) -> set[int]:
        """Return anchor indices inside rect (image coords)."""
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        result = set()
        for i, (ax, ay) in enumerate(self._anchors):
            if x_min <= ax <= x_max and y_min <= ay <= y_max:
                result.add(i)
        return result

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._last_pan_pos = event.position()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            wx, wy = event.position().x(), event.position().y()
            ix, iy = self.widget_to_image(wx, wy)
            if self._mode == MODE_ADD:
                if self._add_first_point is None:
                    self._add_first_point = (ix, iy)
                    self._add_preview_point = (ix, iy)
                    event.accept()
                    return
                else:
                    x1, y1 = self._add_first_point
                    x2, y2 = ix, iy
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    if x_max - x_min > 5 and y_max - y_min > 5:
                        self.box_added.emit([x_min, y_min, x_max, y_max])
                    self._add_first_point = None
                    self._add_preview_point = None
                    event.accept()
                    return
            elif self._mode == MODE_SELECT:
                self._selection_rect_start = (wx, wy)
                self._selection_rect_end = (wx, wy)
                event.accept()
                return
            elif self._mode == MODE_NORMAL:
                aidx = self._anchor_at_widget(wx, wy)
                if aidx is not None:
                    self._selected_anchor = aidx
                    self._anchor_drag_start = (ix, iy)
                    self.anchor_drag_started.emit(aidx)
                else:
                    self._selected_anchor = None
                    self._anchor_drag_start = None
                event.accept()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        wx, wy = event.position().x(), event.position().y()
        ix, iy = self.widget_to_image(wx, wy)
        if (
            self._last_pan_pos is not None
            and event.buttons() & Qt.MouseButton.MiddleButton
        ):
            delta = event.position() - self._last_pan_pos
            self._pan_offset += delta
            self._last_pan_pos = event.position()
            self.update()
            event.accept()
            return
        if (
            self._selection_rect_start is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and self._mode == MODE_SELECT
        ):
            self._selection_rect_end = (wx, wy)
            self.update()
            event.accept()
            return
        if self._add_first_point is not None and self._mode == MODE_ADD:
            self._add_preview_point = (ix, iy)
            self.update()
            event.accept()
            return
        if (
            self._selected_anchor is not None
            and self._anchor_drag_start is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._anchors[self._selected_anchor] = [ix, iy]
            self.anchor_moved.emit(self._selected_anchor, ix, iy)
            self._anchor_drag_start = (ix, iy)
            self.update()
            event.accept()
            return
        hbox = self._box_at_point(ix, iy) if self._mode == MODE_SELECT else None
        hanch = (
            self._anchor_at_widget(wx, wy)
            if (self._mode == MODE_NORMAL or self._mode == MODE_SELECT)
            else None
        )
        if hbox != self._hovered_box or hanch != self._hovered_anchor:
            self._hovered_box = hbox
            self._hovered_anchor = hanch
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._last_pan_pos = None
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if (
                self._selection_rect_start
                and self._selection_rect_end
                and self._mode == MODE_SELECT
            ):
                wx, wy = event.position().x(), event.position().y()
                ix, iy = self.widget_to_image(wx, wy)
                sx, sy = self._selection_rect_start
                ex, ey = self._selection_rect_end
                if abs(ex - sx) < 5 and abs(ey - sy) < 5:
                    b = self._box_at_point(ix, iy)
                    a = self._anchor_at_widget(wx, wy)
                    idx = b if b is not None else a
                    if idx is not None:
                        if idx in self._selected_boxes:
                            self._selected_boxes.discard(idx)
                            self._selected_anchors.discard(idx)
                        else:
                            self._selected_boxes.add(idx)
                            self._selected_anchors.add(idx)
                else:
                    ix1, iy1 = self.widget_to_image(sx, sy)
                    ix2, iy2 = self.widget_to_image(ex, ey)
                    new_boxes = self._boxes_in_rect(ix1, iy1, ix2, iy2)
                    new_anchors = self._anchors_in_rect(ix1, iy1, ix2, iy2)
                    new_pairs = new_boxes | new_anchors
                    self._selected_boxes |= new_pairs
                    self._selected_anchors |= new_pairs
                self._selection_rect_start = None
                self._selection_rect_end = None
                self.boxes_selected.emit(self._selected_boxes)
                self.anchors_selected.emit(self._selected_anchors)
                self.update()
            self._anchor_drag_start = None
        super().mouseReleaseEvent(event)

    def delete_selected(self):
        """Remove selected pairs (box+anchor). Emits boxes_delete_requested for batch delete."""
        to_remove = sorted(self._selected_boxes | self._selected_anchors, reverse=True)
        if to_remove:
            self.boxes_delete_requested.emit(to_remove)
        self._selected_boxes.clear()
        self._selected_anchors.clear()
        self._selected_anchor = None
        if to_remove:
            self.update()

    def delete_selected_anchor(self):
        """Remove selected anchor(s) and boxes (call from main window on Del key)."""
        self.delete_selected()

    def image_to_widget(self, ix: float, iy: float) -> tuple[float, float]:
        ox, oy = self._draw_offset
        return (ox + ix * self._scale, oy + iy * self._scale)

    def widget_to_image(self, wx: float, wy: float) -> tuple[float, float]:
        ox, oy = self._draw_offset
        if self._scale <= 0:
            return (wx, wy)
        return ((wx - ox) / self._scale, (wy - oy) / self._scale)

    def render_to_pixmap(self, annotations_only: bool = False) -> QPixmap | None:
        """Render current view to pixmap. If annotations_only, draw on transparent background at image size."""
        if self._pixmap is None:
            return None
        if annotations_only:
            w, h = int(self._pixmap.width()), int(self._pixmap.height())
            img = QImage(w, h, QImage.Format.Format_ARGB32)
            img.fill(0)
            painter = QPainter(img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            scale = 1.0
            draw_x, draw_y = 0, 0

            def to_xy(ix: float, iy: float):
                return (draw_x + ix * scale, draw_y + iy * scale)

            if self._show_boxes and self._boxes:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for i, box in enumerate(self._boxes):
                    x1, y1, x2, y2 = box
                    if i in self._outlier_indices:
                        painter.setPen(QPen(QColor(255, 0, 0), 3))
                    else:
                        painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if self._show_contour and len(self._contour_points) >= 2:
                painter.setPen(QPen(QColor(255, 100, 100, 180), 2))
                for j in range(len(self._contour_points) - 1):
                    p1, p2 = self._contour_points[j], self._contour_points[j + 1]
                    painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            if (
                self._contour_overlay_pixmap
                and not self._contour_overlay_pixmap.isNull()
            ):
                painter.setOpacity(self._contour_overlay_opacity)
                overlay = self._contour_overlay_pixmap.scaled(
                    w,
                    h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(0, 0, overlay)
                painter.setOpacity(1.0)
            if self._show_anchors and self._anchors:
                painter.setBrush(QBrush(QColor(255, 200, 0, 200)))
                painter.setPen(QPen(QColor(255, 150, 0), 1))
                for ax, ay in self._anchors:
                    painter.drawEllipse(int(ax - 6), int(ay - 6), 12, 12)
            painter.end()
            return QPixmap.fromImage(img)
        return self.grab()
