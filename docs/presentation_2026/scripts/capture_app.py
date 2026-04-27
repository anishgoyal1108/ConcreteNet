"""Qt-internal screenshot harness for ConcreteNet GUI.

Drives MainWindow via its public Python API and renders via widget.grab() so
everything is deterministic and works under Wayland without input injection.

Produces feature-specific screenshots + crops of individual panels.

Usage:
    python scripts/capture_app.py
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESENTATION_ROOT = HERE.parent
PROJECT_ROOT = PRESENTATION_ROOT.parent.parent
OUT = PRESENTATION_ROOT / "img" / "app"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QRect

from app.main_window import MainWindow
from app.canvas import MODE_NORMAL, MODE_ADD, MODE_SELECT
from core.detector import create_detector


def pick_sample(model: str, offset: int = 0) -> Path:
    if model == "gp8000":
        root = PROJECT_ROOT / "data" / "GP8000" / "images" / "val"
        if not root.exists():
            root = PROJECT_ROOT / "data" / "GP8000" / "images" / "train"
    else:
        root = PROJECT_ROOT / "data" / "GSSI" / "images" / "val"
        if not root.exists():
            root = PROJECT_ROOT / "data" / "GSSI" / "images" / "train"
    imgs = sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))
    if not imgs:
        raise FileNotFoundError(f"No images in {root}")
    return imgs[(len(imgs) // 2 + offset) % len(imgs)]


def run_detection(model: str, img_path: Path):
    det = create_detector(model_type=model, project_root=PROJECT_ROOT)
    r = det.detect(str(img_path))
    return r["boxes"], r["confidences"]


def save(widget, name: str, rect: QRect = None):
    if rect is not None:
        pix = widget.grab(rect)
    else:
        pix = widget.grab()
    path = OUT / name
    pix.save(str(path))
    s = pix.size()
    print(f"  + {name}  ({s.width()}x{s.height()})")


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    app.processEvents()

    # ── 1. Auto-detection on GP8000 ───────────────────────────────────────────
    win.toolbar.model_combo.setCurrentText("GP8000")
    app.processEvents()

    sample = pick_sample("gp8000")
    print(f"GP8000 detection on {sample.name}...")
    boxes, confs = run_detection("gp8000", sample)
    win.project.add_image(str(sample), boxes, confs)
    paths = [img["path"] for img in win.project.images]
    win.image_list.set_images(paths)
    win.image_list.setCurrentRow(0)
    win._on_image_selected(0)
    app.processEvents()
    save(win, "01_full_window.png")

    # ── 2. Toolbar strip (crop top ~60 px) ────────────────────────────────────
    tb = win.toolbar
    save(tb, "02_toolbar.png")

    # ── 3. Image list (left sidebar) ─────────────────────────────────────────
    save(win.image_list, "03_image_list_single.png")

    # ── 4. Stats panel (right sidebar) ───────────────────────────────────────
    save(win.stats_panel, "04_stats_panel.png")

    # ── 5. Canvas close-up with detections + anchors ─────────────────────────
    save(win.canvas, "05_canvas_detections.png")

    # ── 6. Add-Box mode ──────────────────────────────────────────────────────
    win.canvas.set_mode(MODE_ADD)
    win._sync_mode_buttons()
    app.processEvents()
    save(win, "06_add_box_mode.png")
    save(win.toolbar, "07_toolbar_add_active.png")
    win.canvas.set_mode(MODE_NORMAL)
    win._sync_mode_buttons()

    # ── 7. Select mode + multi-select ────────────────────────────────────────
    win.canvas.set_mode(MODE_SELECT)
    win._sync_mode_buttons()
    img = win.project.get_image(0)
    if img and img["boxes"]:
        win.canvas._selected_boxes = set(range(min(4, len(img["boxes"]))))
    app.processEvents()
    save(win, "08_select_mode_multi.png")
    save(win.canvas, "09_canvas_selected.png")
    win.canvas._selected_boxes = set()
    win.canvas.set_mode(MODE_NORMAL)
    win._sync_mode_buttons()

    # ── 8. Contour overlay + anchors on ──────────────────────────────────────
    win.toolbar.show_contour.setChecked(True)
    win.toolbar.show_anchors.setChecked(True)
    win._on_toggle_contour(True)
    app.processEvents()
    save(win, "10_contour_overlay.png")
    save(win.canvas, "11_canvas_contour.png")
    win.toolbar.show_contour.setChecked(False)
    win._on_toggle_contour(False)

    # ── 9. Hide boxes, show only anchors ─────────────────────────────────────
    win.toolbar.show_boxes.setChecked(False)
    win.canvas.set_show_boxes(False)
    app.processEvents()
    save(win, "12_anchors_only.png")
    win.toolbar.show_boxes.setChecked(True)
    win.canvas.set_show_boxes(True)

    # ── 10. GSSI model on a GSSI scan ────────────────────────────────────────
    win.project.new()
    win.toolbar.model_combo.setCurrentText("GSSI")
    win.project.data["model_type"] = "gssi"
    try:
        gssi_sample = pick_sample("gssi")
        print(f"GSSI detection on {gssi_sample.name}...")
        gboxes, gconfs = run_detection("gssi", gssi_sample)
        win.project.add_image(str(gssi_sample), gboxes, gconfs)
        paths = [i["path"] for i in win.project.images]
        win.image_list.set_images(paths)
        win.image_list.setCurrentRow(0)
        win._on_image_selected(0)
        app.processEvents()
        save(win, "13_gssi_window.png")
        save(win.canvas, "14_gssi_canvas.png")
    except Exception as e:
        print(f"skipping GSSI: {e}")

    # ── 11. Multi-image project (3 images) ──────────────────────────────────
    win.project.new()
    win.toolbar.model_combo.setCurrentText("GP8000")
    for offset in (-2, 0, 2, 4):
        p = pick_sample("gp8000", offset=offset)
        b, c = run_detection("gp8000", p)
        win.project.add_image(str(p), b, c)
    paths = [i["path"] for i in win.project.images]
    win.image_list.set_images(paths)
    win.image_list.setCurrentRow(1)
    win._on_image_selected(1)
    app.processEvents()
    save(win, "15_multi_image.png")
    save(win.image_list, "16_image_list_multi.png")

    print(f"\nAll screenshots written to: {OUT}")


if __name__ == "__main__":
    main()
