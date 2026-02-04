"""ProjectManager: load, save, add image, update boxes/anchors."""

import json
import uuid
from pathlib import Path
from typing import Literal

ModelType = Literal["gp8000", "gssi"]


def _default_image_data(path: str) -> dict:
    return {
        "path": path,
        "boxes": [],
        "confidences": [],
        "removed_indices": [],
        "added_boxes": [],
        "anchors": [],
        "contour_map_id": None,
    }


def effective_boxes(img: dict) -> list[list[float]]:
    """boxes minus removed_indices, plus added_boxes."""
    boxes = [b for i, b in enumerate(img["boxes"]) if i not in img["removed_indices"]]
    return boxes + img["added_boxes"]


def default_anchors_from_boxes(boxes: list[list[float]]) -> list[list[float]]:
    """Top-center of each box."""
    anchors = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x = (x1 + x2) / 2
        y = y1
        anchors.append([x, y])
    return anchors


class ProjectManager:
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.project_path: Path | None = None
        self.data: dict = self._empty_project()

    def _empty_project(self) -> dict:
        return {
            "version": 1,
            "model_type": "gp8000",
            "images": [],
            "contour_maps": {},
        }

    def new(self, model_type: ModelType = "gp8000"):
        """Start a new project."""
        self.data = self._empty_project()
        self.data["model_type"] = model_type
        self.project_path = None

    def add_image(
        self,
        path: str | Path,
        boxes: list[list[float]],
        confidences: list[float],
        anchors: list[list[float]] | None = None,
    ) -> dict:
        """Add an image with detection results."""
        path_str = str(Path(path).resolve())
        img = _default_image_data(path_str)
        img["boxes"] = [list(b) for b in boxes]
        img["confidences"] = list(confidences)
        if anchors is not None:
            img["anchors"] = [list(a) for a in anchors]
        else:
            img["anchors"] = default_anchors_from_boxes(boxes)
        self.data["images"].append(img)
        return img

    def get_image(self, index: int) -> dict | None:
        if 0 <= index < len(self.data["images"]):
            return self.data["images"][index]
        return None

    def get_image_by_path(self, path: str | Path) -> dict | None:
        path_str = str(Path(path).resolve())
        for img in self.data["images"]:
            if str(Path(img["path"]).resolve()) == path_str:
                return img
        return None

    def update_image(
        self,
        index: int,
        *,
        removed_indices: list[int] | None = None,
        added_boxes: list[list[float]] | None = None,
        anchors: list[list[float]] | None = None,
    ):
        img = self.get_image(index)
        if not img:
            return
        if removed_indices is not None:
            img["removed_indices"] = removed_indices
        if added_boxes is not None:
            img["added_boxes"] = [list(b) for b in added_boxes]
        if anchors is not None:
            img["anchors"] = [list(a) for a in anchors]

    def remove_box(self, image_index: int, box_index: int):
        """Mark a detected box as false positive or remove added box."""
        img = self.get_image(image_index)
        if not img:
            return
        visible_orig = [
            i for i in range(len(img["boxes"])) if i not in img["removed_indices"]
        ]
        orig_count = len(visible_orig)
        if box_index < orig_count:
            img["removed_indices"] = sorted(
                set(img["removed_indices"]) | {visible_orig[box_index]}
            )
        else:
            idx = box_index - orig_count
            if 0 <= idx < len(img["added_boxes"]):
                img["added_boxes"].pop(idx)

    def move_pairs(
        self, image_index: int, pair_indices: set[int], dx: float, dy: float
    ) -> None:
        """Move boxes and anchors by (dx, dy) for given effective indices."""
        img = self.get_image(image_index)
        if not img:
            return
        visible_orig = [
            i for i in range(len(img["boxes"])) if i not in img["removed_indices"]
        ]
        orig_count = len(visible_orig)
        anchors = img.get("anchors", [])
        for i in pair_indices:
            if i < orig_count:
                oi = visible_orig[i]
                b = img["boxes"][oi]
                b[0] += dx
                b[1] += dy
                b[2] += dx
                b[3] += dy
            else:
                add_idx = i - orig_count
                if 0 <= add_idx < len(img["added_boxes"]):
                    b = img["added_boxes"][add_idx]
                    b[0] += dx
                    b[1] += dy
                    b[2] += dx
                    b[3] += dy
            if 0 <= i < len(anchors):
                anchors[i][0] += dx
                anchors[i][1] += dy
        img["anchors"] = anchors

    def remove_pair(self, image_index: int, pair_index: int):
        """Remove box and corresponding anchor at effective index (boxes/anchors come as pairs)."""
        self.remove_box(image_index, pair_index)
        img = self.get_image(image_index)
        if img and 0 <= pair_index < len(img.get("anchors", [])):
            anchors = img["anchors"]
            anchors.pop(pair_index)
            img["anchors"] = anchors

    def unremove_box(self, image_index: int, removed_index: int):
        """Undo a false positive removal."""
        img = self.get_image(image_index)
        if not img or removed_index not in img["removed_indices"]:
            return
        img["removed_indices"] = [
            i for i in img["removed_indices"] if i != removed_index
        ]

    def add_box(self, image_index: int, box: list[float]):
        """Add a user-drawn box."""
        img = self.get_image(image_index)
        if img:
            img["added_boxes"].append(list(box))

    def set_anchors(self, image_index: int, anchors: list[list[float]]):
        img = self.get_image(image_index)
        if img:
            img["anchors"] = [list(a) for a in anchors]

    def add_contour_map(self, points: list[list[float]], source_image_path: str) -> str:
        cid = str(uuid.uuid4())
        self.data["contour_maps"][cid] = {
            "points": [list(p) for p in points],
            "source_image": source_image_path,
        }
        return cid

    def get_contour_map(self, contour_map_id: str) -> dict | None:
        return self.data["contour_maps"].get(contour_map_id)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        self.project_path = path

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path) as f:
            self.data = json.load(f)
        self.project_path = path

    @property
    def images(self) -> list[dict]:
        return self.data["images"]

    @property
    def model_type(self) -> str:
        return self.data.get("model_type", "gp8000")
