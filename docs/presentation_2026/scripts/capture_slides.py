"""Render canvas-only annotated images for the deformity, human-in-the-loop,
and contour-over-years slides.

Outputs to img/app/ (crop-free, canvas-only, native image resolution).

Produces:
  hitl_add_boxes.png        — existing detections (green) + two newly added boxes (cyan)
  hitl_select_highlight.png — existing detections with a subset selected/highlighted
  deformity_example.png     — detections with one box deleted → flanking boxes red,
                              plus a matplotlib overlay showing "gap > μ+2σ"
  contour_years.png         — one B-scan with three overlaid depth contours
                              (Year 1 / Year 2 / Year 3)
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

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.detector import create_detector
from core.spacing import SpacingAnalyzer
from core.project import default_anchors_from_boxes
from core.contour import compute_contour


# ── BGR colour palette (cv2) ───────────────────────────────────────────────
GREEN    = (0, 220, 40)
RED      = (0, 0, 235)
CYAN     = (255, 215, 0)     # BGR for strong cyan — new user-added boxes
GOLD     = (0, 200, 255)
GOLD_DK  = (0, 150, 220)
PINK     = (100, 100, 255)


def pick(model: str, offset: int = 0) -> Path:
    root = PROJECT_ROOT / "data" / model.upper() / "images" / "val"
    if not root.exists():
        root = PROJECT_ROOT / "data" / model.upper() / "images" / "train"
    imgs = sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))
    return imgs[(len(imgs) // 2 + offset) % len(imgs)]


def run_detect(model: str, path: Path):
    det = create_detector(model_type=model, project_root=PROJECT_ROOT)
    r = det.detect(str(path))
    return r["boxes"], r["confidences"]


def draw_box(img, b, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in b]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_anchor(img, pt, r=6, fill=GOLD, outline=GOLD_DK):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img, (x, y), r, fill, -1)
    cv2.circle(img, (x, y), r, outline, 1)


def draw_contour(img, points, color, thickness=2):
    if len(points) < 2:
        return
    pts = np.array([[int(p[0]), int(p[1])] for p in points], np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)


def render_annotations(
    image_path: Path,
    boxes: list,
    anchors: list | None = None,
    outlier_indices: set | None = None,
    extra_boxes: list | None = None,
    highlighted: set | None = None,
    contour_points_list: list | None = None,
):
    """
    Bake annotations onto the native-resolution image (no Qt chrome).
    - boxes: green normally, red if index in outlier_indices
    - extra_boxes: drawn in CYAN (user-added)
    - highlighted: indices drawn with translucent green fill + thicker border
    - contour_points_list: list of (pts, color) tuples
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cv2 could not load {image_path}")

    outlier_indices = outlier_indices or set()
    highlighted = highlighted or set()

    for i, b in enumerate(boxes):
        if i in outlier_indices:
            draw_box(img, b, RED, thickness=4)
        else:
            draw_box(img, b, GREEN, thickness=2)
        if i in highlighted:
            # Orange fill + thick outline so selection stands out against green
            x1, y1, x2, y2 = [int(v) for v in b]
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 120, 255), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 100, 255), 4)

    if extra_boxes:
        for b in extra_boxes:
            draw_box(img, b, CYAN, thickness=3)

    if anchors:
        for a in anchors:
            draw_anchor(img, a, r=7)

    if contour_points_list:
        for pts, color in contour_points_list:
            draw_contour(img, pts, color, thickness=3)

    return img


def save_png(img, name: str):
    path = OUT / name
    cv2.imwrite(str(path), img)
    print(f"  + {name}  ({img.shape[1]}x{img.shape[0]})")


def crop_rebar_row(img, boxes, pad_top=15, pad_bottom=60):
    """Crop vertically to the rebar row + a little padding.
    Returns (cropped_img, y_offset_removed) so caller can adjust overlays.
    """
    ys_top = [b[1] for b in boxes]
    ys_bot = [b[3] for b in boxes]
    y_top = max(0, int(min(ys_top)) - pad_top)
    y_bot = min(img.shape[0], int(max(ys_bot)) + pad_bottom)
    return img[y_top:y_bot, :, :].copy(), y_top


# ═════════════════════════════════════════════════════════════════════════════
# 1. Add-boxes HITL screenshot — green (auto) + cyan (user-added)
# ═════════════════════════════════════════════════════════════════════════════
def cap_add_boxes():
    p = pick("gp8000", offset=2)
    boxes, _ = run_detect("gp8000", p)
    anchors = default_anchors_from_boxes(boxes)
    # Place two user-added cyan boxes in the upper-right quadrant where we
    # delete a couple of existing ones to avoid overlap
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    # Synthesize two plausible user-added cyan boxes placed BELOW the rebar row
    # so they stand out against the detected set (clearly new, not overlapping).
    median_w = np.median([b[2] - b[0] for b in boxes])
    median_h = np.median([b[3] - b[1] for b in boxes])
    # Pick reference x positions from boxes ~1/3 and ~2/3 across
    ref1 = sorted_boxes[len(sorted_boxes) // 3]
    ref2 = sorted_boxes[2 * len(sorted_boxes) // 3]
    # Place extras just below the rebar layer
    y_top = ref1[3] + 10
    extras = [
        [ref1[0], y_top, ref1[0] + median_w * 1.1, y_top + median_h * 1.2],
        [ref2[0], y_top, ref2[0] + median_w * 1.1, y_top + median_h * 1.2],
    ]
    img = render_annotations(p, boxes, anchors=anchors, extra_boxes=extras)
    cropped, _ = crop_rebar_row(img, boxes + extras, pad_top=25, pad_bottom=90)
    save_png(cropped, "hitl_add_boxes.png")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Select-mode HITL screenshot — a subset highlighted
# ═════════════════════════════════════════════════════════════════════════════
def cap_select_highlight():
    p = pick("gp8000", offset=4)
    boxes, _ = run_detect("gp8000", p)
    anchors = default_anchors_from_boxes(boxes)
    # Highlight 4 contiguous boxes near the middle
    sorted_idx = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    mid = len(sorted_idx) // 2
    highlighted = set(sorted_idx[mid - 2 : mid + 2])
    img = render_annotations(p, boxes, anchors=anchors, highlighted=highlighted)
    cropped, _ = crop_rebar_row(img, boxes, pad_top=25, pad_bottom=60)
    save_png(cropped, "hitl_select_highlight.png")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Deformity example — delete middle box, flanking outliers turn red
# ═════════════════════════════════════════════════════════════════════════════
def cap_deformity():
    p = pick("gp8000", offset=0)
    boxes, _ = run_detect("gp8000", p)

    # Sort by x and drop ONE box in the middle to produce a large gap
    sorted_idx = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    delete_at = len(sorted_idx) // 2
    deleted_orig_idx = sorted_idx[delete_at]

    kept_boxes = [b for i, b in enumerate(boxes) if i != deleted_orig_idx]
    anchors = default_anchors_from_boxes(kept_boxes)

    stats = SpacingAnalyzer().analyze(kept_boxes, anchors)
    outliers = stats["outlier_box_indices"]
    median = stats["median"]
    std = float(np.std(stats["distances"])) if len(stats["distances"]) > 1 else 0.0

    # Render annotated image (green everywhere, red on flanking boxes)
    img = render_annotations(p, kept_boxes, anchors=anchors, outlier_indices=outliers)

    # Crop to rebar row for readability
    img, y_off = crop_rebar_row(img, kept_boxes, pad_top=85, pad_bottom=60)

    # Adjust anchor y's and gap label y positions to cropped coord system
    anchors = [[a[0], a[1] - y_off] for a in anchors]

    # Also save the raw annotated PNG for reuse
    save_png(img, "deformity_raw.png")

    # Now overlay a gap indicator with matplotlib → vector PDF for LaTeX
    # Identify the gap location: find the outlier distance and its x span
    gap_x_start = gap_x_end = None
    gap_y = None
    sorted_anchor_idx = sorted(range(len(anchors)), key=lambda i: anchors[i][0])
    for j in range(len(sorted_anchor_idx) - 1):
        a_idx = sorted_anchor_idx[j]
        b_idx = sorted_anchor_idx[j + 1]
        d = anchors[b_idx][0] - anchors[a_idx][0]
        if abs(d - median) > max(2 * std, 6.0):
            gap_x_start = anchors[a_idx][0]
            gap_x_end = anchors[b_idx][0]
            gap_y = (anchors[a_idx][1] + anchors[b_idx][1]) / 2
            break

    fig, ax = plt.subplots(figsize=(11, 11 * img.shape[0] / img.shape[1]))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_axis_off()

    if gap_x_start is not None:
        # Draw a gold double-headed arrow above the gap with label
        y_label = max(20, gap_y - 80)
        ax.annotate(
            "",
            xy=(gap_x_end, y_label),
            xytext=(gap_x_start, y_label),
            arrowprops=dict(arrowstyle="<->", color="#C9A84C", lw=3),
        )
        label_text = (
            f"gap = {gap_x_end - gap_x_start:.0f} px  >  "
            rf"$\mu + 2\sigma$  =  {median + 2 * std:.0f} px"
        )
        ax.text(
            (gap_x_start + gap_x_end) / 2,
            y_label - 20,
            label_text,
            ha="center", va="bottom",
            fontsize=16, color="#0D1B2A",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#FFF8E1", edgecolor="#C9A84C", lw=1.5),
        )
        # Vertical guides
        ax.plot([gap_x_start, gap_x_start], [y_label, gap_y + 20],
                color="#C9A84C", lw=1.2, linestyle=":")
        ax.plot([gap_x_end, gap_x_end], [y_label, gap_y + 20],
                color="#C9A84C", lw=1.2, linestyle=":")

    out = OUT / "deformity_example.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=200)
    # Also PNG at screen res
    out_png = OUT / "deformity_example.png"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)
    print(f"  + deformity_example.pdf  /  .png")
    print(f"     median = {median:.1f} px,  std = {std:.1f} px,  "
          f"threshold = {max(2 * std, 6.0):.1f} px")
    print(f"     outliers flagged: {len(outliers)} boxes")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Contour-over-years overlay
# ═════════════════════════════════════════════════════════════════════════════
def cap_contour_years():
    p = pick("gp8000", offset=0)
    boxes, _ = run_detect("gp8000", p)
    anchors0 = default_anchors_from_boxes(boxes)
    anchors0 = sorted(anchors0, key=lambda a: a[0])

    img_w = cv2.imread(str(p)).shape[1]

    # Simulate rebar-level drift over three years:
    # Year 1 = baseline
    # Year 2 = +15 px downshift on the right half (simulating settling)
    # Year 3 = +28 px downshift on the right 2/3 + 8 px on middle
    def shifted(anchors, dy_fn):
        return [[a[0], a[1] + dy_fn(a[0])] for a in anchors]

    def noise(seed, amplitude=3.0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, amplitude, len(anchors0))

    n1 = noise(1, 2.5)
    y1 = [[a[0], a[1] + n1[i]] for i, a in enumerate(anchors0)]

    def dy2(x, xmax=img_w):
        frac = max(0.0, (x / xmax) - 0.2) / 0.8
        return 55.0 * frac

    n2 = noise(2, 4.0)
    y2 = [[a[0], a[1] + dy2(a[0]) + n2[i]] for i, a in enumerate(anchors0)]

    def dy3(x, xmax=img_w):
        frac = max(0.0, (x / xmax) - 0.1) / 0.9
        return 110.0 * frac ** 1.25

    n3 = noise(3, 5.0)
    y3 = [[a[0], a[1] + dy3(a[0]) + n3[i]] for i, a in enumerate(anchors0)]

    c1 = compute_contour(y1, image_width=img_w, num_samples=400)
    c2 = compute_contour(y2, image_width=img_w, num_samples=400)
    c3 = compute_contour(y3, image_width=img_w, num_samples=400)

    # BGR for three years — blue/green/red
    COL_1 = (230, 180, 60)    # calm blue-teal
    COL_2 = (40, 200, 220)    # amber/gold
    COL_3 = (60, 60, 230)     # red

    img = cv2.imread(str(p))
    draw_contour(img, c1, COL_1, thickness=3)
    draw_contour(img, c2, COL_2, thickness=3)
    draw_contour(img, c3, COL_3, thickness=3)

    # Add legend in top-right corner
    legend_x = img.shape[1] - 260
    legend_y = 30
    pad = 6
    cv2.rectangle(
        img,
        (legend_x - pad, legend_y - 22),
        (legend_x + 250, legend_y + 100),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        img,
        (legend_x - pad, legend_y - 22),
        (legend_x + 250, legend_y + 100),
        (120, 120, 120),
        1,
    )
    for i, (lbl, col) in enumerate([
        ("Year 1  (baseline)", COL_1),
        ("Year 2  (+1 yr)",     COL_2),
        ("Year 3  (+2 yr)",     COL_3),
    ]):
        y = legend_y + i * 32
        cv2.line(img, (legend_x, y), (legend_x + 40, y), col, 4)
        cv2.putText(img, lbl, (legend_x + 50, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)

    # Crop to rebar row so the contours dominate the view
    cropped, _ = crop_rebar_row(img, boxes, pad_top=40, pad_bottom=180)
    save_png(cropped, "contour_years.png")


if __name__ == "__main__":
    print("Generating canvas-only slide images...")
    cap_add_boxes()
    cap_select_highlight()
    cap_deformity()
    cap_contour_years()
    print(f"\nAll outputs in: {OUT}")
