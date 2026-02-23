#!/usr/bin/env python3
"""
Import CVAT YOLO 1.1 export into the YOLOv8 dataset structure.

CVAT YOLO 1.1 export places images and .txt labels together in obj_train_data/.
This script splits them into data/{DATASET}/images/{train,val} and
data/{DATASET}/labels/{train,val} for use with train_models.py.

Usage:
    python import_yolo11_annotations.py --dataset gssi
    python import_yolo11_annotations.py --dataset gp8000 --source some_other_dir
"""

import argparse
import random
import shutil
from pathlib import Path

random.seed(42)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find (image, label) pairs in source_dir.
    Each image must have a corresponding .txt file with the same stem.
    """
    pairs = []
    for img_path in sorted(source_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"  Warning: no label for {img_path.name}, skipping")
            continue
        pairs.append((img_path, label_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Import CVAT YOLO 1.1 annotations into YOLOv8 dataset structure"
    )
    parser.add_argument(
        "--dataset",
        choices=["gssi", "gp8000"],
        default="gssi",
        help="Target dataset (default: gssi)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("obj_train_data"),
        help="Directory containing images + .txt labels (default: obj_train_data)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train fraction (default: 0.8 = 80%% train, 20%% val)",
    )
    args = parser.parse_args()

    source_dir = args.source
    dataset_name = args.dataset.upper()
    output_dir = Path("data") / dataset_name

    print(f"Source:  {source_dir}")
    print(f"Dataset: {dataset_name}  ->  {output_dir}")

    if not source_dir.exists():
        print(f"Error: source directory '{source_dir}' not found.")
        return 1

    pairs = collect_pairs(source_dir)
    if not pairs:
        print(
            f"No image+label pairs found in '{source_dir}'.\n"
            "Make sure the images (e.g. image124.png) are in that directory alongside the .txt files."
        )
        return 1

    print(f"Found {len(pairs)} image+label pairs")

    # Train / val split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Split:   {len(train_pairs)} train, {len(val_pairs)} val")

    # Create output dirs
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        for img_src, lbl_src in split_pairs:
            dst_img = img_dir / img_src.name
            dst_lbl = lbl_dir / lbl_src.name
            if img_src.resolve() != dst_img.resolve():
                shutil.copy2(img_src, dst_img)
            if lbl_src.resolve() != dst_lbl.resolve():
                shutil.copy2(lbl_src, dst_lbl)

    print(f"\nDone. Dataset written to {output_dir}/")
    print(f"  {output_dir}/images/train  ({len(train_pairs)} images)")
    print(f"  {output_dir}/images/val    ({len(val_pairs)} images)")
    print(f"\nNext: python train_models.py --model {args.dataset}")
    return 0


if __name__ == "__main__":
    exit(main())
