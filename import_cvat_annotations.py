#!/usr/bin/env python3
"""
Import CVAT annotations.xml and set up YOLO dataset with 80-20 train/val split.
Converts Pascal VOC (xtl, ytl, xbr, ybr) to YOLO format (normalized x_center, y_center, width, height).
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Class name to YOLO class ID
CLASS_MAP = {"rebar": 0}

random.seed(42)


def parse_cvat_xml(xml_path: Path) -> list[dict]:
    """
    Parse CVAT annotations.xml. Returns list of dicts:
    [{"name": "image1.png", "width": 2445, "height": 1485, "boxes": [(class_id, x_center, y_center, w, h), ...]}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = []
    for image_elem in root.findall("image"):
        name = image_elem.get("name")
        width = int(float(image_elem.get("width", 1)))
        height = int(float(image_elem.get("height", 1)))

        boxes = []
        for box in image_elem.findall("box"):
            label = box.get("label", "rebar")
            class_id = CLASS_MAP.get(label, 0)

            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))

            # Convert to YOLO format (normalized 0-1)
            x_center = (xtl + xbr) / 2 / width
            y_center = (ytl + ybr) / 2 / height
            w = (xbr - xtl) / width
            h = (ybr - ytl) / height

            # Clamp to valid range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0.001, min(1, w))
            h = max(0.001, min(1, h))

            boxes.append((class_id, x_center, y_center, w, h))

        result.append({"name": name, "width": width, "height": height, "boxes": boxes})

    return result


def find_image_path(image_name: str, search_dirs: list[Path]) -> Path | None:
    """Find image file by name in search directories (searches recursively)."""
    for d in search_dirs:
        if not d.exists():
            continue
        # Check exact path first
        direct = d / image_name
        if direct.exists():
            return direct
        # Search recursively
        for p in d.rglob(image_name):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Import CVAT annotations.xml, convert to YOLO format, 80-20 train/val split"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("annotations.xml"),
        help="Path to CVAT annotations.xml",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/GP8000/images"),
        help="Directory containing images (searches train/ and val/ recursively)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/GP8000"),
        help="Output dataset directory (creates images/train, images/val, labels/train, labels/val)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8 = 80%% train, 20%% val)",
    )
    args = parser.parse_args()

    # Parse annotations (only include images that have at least one box)
    print(f"Parsing {args.annotations}...")
    all_images = parse_cvat_xml(args.annotations)
    images_data = [img for img in all_images if img["boxes"]]
    print(
        f"  Found {len(images_data)} images with annotations (skipped {len(all_images) - len(images_data)} empty)"
    )

    # Search directories for images (check train/val subdirs and Cleaned as fallback)
    search_dirs = [
        args.images,
        args.images / "train",
        args.images / "val",
        args.output.parent / "Cleaned" / "Training" / args.output.name,
        args.output.parent / "Cleaned" / "Validation" / args.output.name,
    ]

    # Build list of (image_data, image_path) for images we can find
    found = []
    missing = []
    for img in images_data:
        path = find_image_path(img["name"], search_dirs)
        if path:
            found.append((img, path))
        else:
            missing.append(img["name"])

    if missing:
        print(f"\nWarning: Could not find {len(missing)} images:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if not found:
        print("No images found. Check --images path.")
        return 1

    print(f"  Matched {len(found)} images")

    # 80-20 split
    random.shuffle(found)
    split_idx = int(len(found) * args.split)
    train_data = found[:split_idx]
    val_data = found[split_idx:]

    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val")

    train_names = {img["name"] for img, _ in train_data}
    val_names = {img["name"] for img, _ in val_data}

    # Create output dirs
    for split in ["train", "val"]:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy images and write labels first (sources may be in these dirs)
    for split_name, data in [("train", train_data), ("val", val_data)]:
        img_dir = args.output / "images" / split_name
        lbl_dir = args.output / "labels" / split_name

        for img, src_path in data:
            name = img["name"]
            # Copy image (skip if already in place)
            dst_img = img_dir / name
            if src_path.resolve() != dst_img.resolve():
                shutil.copy2(src_path, dst_img)

            # Write YOLO label (.txt, same base name)
            label_path = lbl_dir / Path(name).with_suffix(".txt")
            with open(label_path, "w") as f:
                for box in img["boxes"]:
                    line = " ".join(f"{x:.6g}" for x in box) + "\n"
                    f.write(line)

    # Remove unannotated images/labels (files not in our 80-20 split)
    for split_name, keep_names in [("train", train_names), ("val", val_names)]:
        img_dir = args.output / "images" / split_name
        lbl_dir = args.output / "labels" / split_name
        keep_stems = {Path(n).stem for n in keep_names}
        for d in (img_dir, lbl_dir):
            for f in d.iterdir():
                if f.stem not in keep_stems:
                    f.unlink()

    print(f"\nDone! Dataset written to {args.output}")
    print(f"  Train: {args.output}/images/train, {args.output}/labels/train")
    print(f"  Val:   {args.output}/images/val,   {args.output}/labels/val")
    print("\nTrain with: python train_models.py --model gp8000")
    return 0


if __name__ == "__main__":
    exit(main())
