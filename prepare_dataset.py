#!/usr/bin/env python3
"""
Script to reorganize cleaned GPR images into YOLO-compatible directory structure.
Splits data into training and validation sets.
"""

import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Base directories
BASE_DIR = Path("/home/anish/Documents/Anish Goyal's Vault/02. Projects/ConcreteNet")
CLEANED_DIR = BASE_DIR / "data" / "Cleaned"
DATA_DIR = BASE_DIR / "data"

# Train/val split ratio (for the Training folder images)
TRAIN_SPLIT = 0.8


def create_yolo_structure():
    """Create YOLO-compatible directory structure."""
    for dataset in ["GP8000", "GSSI"]:
        for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
            (DATA_DIR / dataset / subdir).mkdir(parents=True, exist_ok=True)
            print(f"Created: data/{dataset}/{subdir}")


def copy_images(source_dir: Path, dest_dir: Path, file_list: list = None):
    """Copy images from source to destination."""
    if file_list is None:
        file_list = list(source_dir.glob("*.png"))
    
    for img_path in file_list:
        dest_path = dest_dir / img_path.name
        shutil.copy2(img_path, dest_path)
    
    return len(file_list)


def organize_dataset(dataset_name: str):
    """Organize a single dataset (GP8000 or GSSI) into YOLO structure."""
    print(f"\n{'='*50}")
    print(f"Organizing {dataset_name} dataset")
    print('='*50)
    
    # Source directories
    training_src = CLEANED_DIR / "Training" / dataset_name
    validation_src = CLEANED_DIR / "Validation" / dataset_name
    
    # Destination directories
    train_dest = DATA_DIR / dataset_name / "images" / "train"
    val_dest = DATA_DIR / dataset_name / "images" / "val"
    
    # Get all training images and shuffle
    training_images = list(training_src.glob("*.png"))
    random.shuffle(training_images)
    
    # Split training images
    split_idx = int(len(training_images) * TRAIN_SPLIT)
    train_images = training_images[:split_idx]
    val_images_from_train = training_images[split_idx:]
    
    # Copy training split to train folder
    count = copy_images(training_src, train_dest, train_images)
    print(f"Copied {count} images to train/ (from Training/)")
    
    # Copy validation split from training to val folder
    count = copy_images(training_src, val_dest, val_images_from_train)
    print(f"Copied {count} images to val/ (split from Training/)")
    
    # Copy all validation images to val folder
    validation_images = list(validation_src.glob("*.png"))
    count = copy_images(validation_src, val_dest, validation_images)
    print(f"Copied {count} images to val/ (from Validation/)")
    
    # Summary
    total_train = len(list(train_dest.glob("*.png")))
    total_val = len(list(val_dest.glob("*.png")))
    print(f"\nTotal: {total_train} training, {total_val} validation images")


def main():
    print("="*60)
    print("YOLO Dataset Preparation Tool")
    print("="*60)
    print(f"\nSource: {CLEANED_DIR}")
    print(f"Output: {DATA_DIR}/GP8000 and {DATA_DIR}/GSSI")
    print(f"Train/Val split: {TRAIN_SPLIT*100:.0f}% / {(1-TRAIN_SPLIT)*100:.0f}%")
    
    # Create directory structure
    print("\n--- Creating directory structure ---")
    create_yolo_structure()
    
    # Organize each dataset
    organize_dataset("GP8000")
    organize_dataset("GSSI")
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install labelImg: pip install labelImg")
    print("2. Annotate GP8000: labelimg data/GP8000/images/train data/GP8000/labels/train")
    print("3. Annotate GSSI: labelimg data/GSSI/images/train data/GSSI/labels/train")
    print("4. Remember to also annotate validation images!")


if __name__ == "__main__":
    main()
