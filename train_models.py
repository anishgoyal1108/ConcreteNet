#!/usr/bin/env python3
"""
Training script for YOLOv8 rebar detection models.
Trains two separate models: one for GP8000 and one for GSSI format images.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(
    dataset_yaml: str,
    model_name: str,
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = None,
):
    """
    Train a YOLOv8 model for rebar detection.
    
    Args:
        dataset_yaml: Path to dataset YAML configuration file
        model_name: Name for the training run (e.g., 'gp8000_rebar')
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device to use ('cpu', '0', '0,1', etc.). None for auto-detect.
    
    Returns:
        Trained model
    """
    # Load pretrained model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Training arguments
    train_args = {
        "data": dataset_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "name": model_name,
        "patience": 20,  # Early stopping patience
        "save": True,
        "plots": True,
        # Data augmentation (helps with small datasets)
        "augment": True,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "degrees": 0.0,  # No rotation for GPR images
        "translate": 0.1,
        "scale": 0.2,
        "flipud": 0.0,  # No vertical flip for GPR
        "fliplr": 0.5,  # Horizontal flip is ok
    }
    
    if device is not None:
        train_args["device"] = device
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} model")
    print(f"Dataset: {dataset_yaml}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
    print(f"{'='*60}\n")
    
    # Train the model
    results = model.train(**train_args)
    
    return model, results


def train_gp8000(model_size: str = "n", epochs: int = 100, batch: int = 16, device: str = None):
    """Train the GP8000 rebar detection model."""
    return train_model(
        dataset_yaml="gp8000_dataset.yaml",
        model_name="gp8000_rebar",
        model_size=model_size,
        epochs=epochs,
        batch=batch,
        device=device,
    )


def train_gssi(model_size: str = "n", epochs: int = 100, batch: int = 16, device: str = None):
    """Train the GSSI rebar detection model."""
    return train_model(
        dataset_yaml="gssi_dataset.yaml",
        model_name="gssi_rebar",
        model_size=model_size,
        epochs=epochs,
        batch=batch,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 rebar detection models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gp8000", "gssi", "both"],
        default="both",
        help="Which model(s) to train",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cpu', '0' for GPU 0, '0,1' for multi-GPU)",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ConcreteNet - Rebar Detection Training")
    print("="*60)
    
    if args.model in ["gp8000", "both"]:
        print("\n>>> Training GP8000 model...")
        train_gp8000(
            model_size=args.size,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
        )
    
    if args.model in ["gssi", "both"]:
        print("\n>>> Training GSSI model...")
        train_gssi(
            model_size=args.size,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
        )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nTrained models saved to: runs/detect/")
    print("  - GP8000: runs/detect/gp8000_rebar/weights/best.pt")
    print("  - GSSI: runs/detect/gssi_rebar/weights/best.pt")


if __name__ == "__main__":
    main()
