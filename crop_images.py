#!/usr/bin/env python3
"""
Script to crop GPR scan images, removing UI elements and keeping only the main scan area.
Handles both GP8000 and GSSI image formats.
Saves cropped images to data/Cropped/ while preserving originals in data/Raw/
"""

from PIL import Image
import os
from pathlib import Path

# Base directories
DATA_DIR = Path("/home/anish/Documents/Anish Goyal's Vault/02. Projects/ConcreteNet/data")
RAW_DIR = DATA_DIR / "Raw"
CROPPED_DIR = DATA_DIR / "Cropped"

# Crop coordinates (left, top, right, bottom)
# GP8000 images are 2732 x 2048
# Main scan area is between the left depth scale and right buttons, 
# below the header/color bar and above the bottom distance scale
GP8000_CROP = (95, 255, 2540, 1740)

# GSSI images are 843 x 874
# Main scan area is between the left depth scale and right tick marks,
# below top labels and above bottom status bar
GSSI_CROP = (65, 15, 790, 835)


def crop_image(input_path: Path, crop_box: tuple, output_path: Path):
    """Crop an image to the specified box and save it."""
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with Image.open(input_path) as img:
        cropped = img.crop(crop_box)
        cropped.save(output_path)
        return cropped.size


def process_directory(input_dir: Path, output_dir: Path, crop_box: tuple, device_name: str):
    """Process all PNG images in a directory, saving to output directory."""
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return
    
    png_files = list(input_dir.glob("*.png"))
    print(f"\nProcessing {len(png_files)} {device_name} images")
    print(f"  From: {input_dir}")
    print(f"  To:   {output_dir}")
    
    for img_path in sorted(png_files):
        output_path = output_dir / img_path.name
        try:
            new_size = crop_image(img_path, crop_box, output_path)
            print(f"  Cropped: {img_path.name} -> {new_size[0]}x{new_size[1]}")
        except Exception as e:
            print(f"  ERROR: {img_path.name} - {e}")


def main():
    # Directories to process: (input_subpath, crop_box, device_name)
    directories = [
        ("Training/GP8000", GP8000_CROP, "GP8000"),
        ("Training/GSSI", GSSI_CROP, "GSSI"),
        ("Validation/GP8000", GP8000_CROP, "GP8000"),
        ("Validation/GSSI", GSSI_CROP, "GSSI"),
    ]
    
    print("=" * 60)
    print("GPR Image Cropping Tool")
    print("=" * 60)
    print(f"\nSource: {RAW_DIR}")
    print(f"Output: {CROPPED_DIR}")
    print(f"\nGP8000 crop box: {GP8000_CROP}")
    print(f"GSSI crop box: {GSSI_CROP}")
    
    for subpath, crop_box, device_name in directories:
        input_dir = RAW_DIR / subpath
        output_dir = CROPPED_DIR / subpath
        process_directory(input_dir, output_dir, crop_box, device_name)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Cropped images saved to: {CROPPED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
