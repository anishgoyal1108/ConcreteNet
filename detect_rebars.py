#!/usr/bin/env python3
"""
Inference script for rebar detection using trained YOLOv8 models.
Supports both GP8000 and GSSI formats with visualization.
"""

import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


# Default model paths (after training)
DEFAULT_MODELS = {
    "gp8000": "runs/detect/gp8000_rebar/weights/best.pt",
    "gssi": "runs/detect/gssi_rebar/weights/best.pt",
}

# Visualization colors (BGR format for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (0, 255, 0)  # Green
BOX_THICKNESS = 2
FONT_SCALE = 0.6


class RebarDetector:
    """Rebar detection class for GPR images."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        """
        Initialize the detector with a trained model.
        
        Args:
            model_path: Path to the trained YOLOv8 weights (.pt file)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect(self, image_path: str) -> dict:
        """
        Detect rebars in an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing:
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - confidences: List of confidence scores
                - count: Number of detected rebars
        """
        results = self.model(image_path, conf=self.confidence_threshold)
        result = results[0]
        
        boxes = []
        confidences = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                boxes.append(box.xyxy[0].tolist())
                confidences.append(float(box.conf[0]))
        
        return {
            "boxes": boxes,
            "confidences": confidences,
            "count": len(boxes),
        }
    
    def detect_and_visualize(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Detect rebars and visualize results.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the annotated image (optional)
            show: Whether to display the image
            
        Returns:
            Tuple of (annotated_image, detection_results)
        """
        # Run detection
        detections = self.detect(image_path)
        
        # Load image for visualization
        image = cv2.imread(str(image_path))
        
        # Draw bounding boxes
        for i, (box, conf) in enumerate(zip(detections["boxes"], detections["confidences"])):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            
            # Draw label
            label = f"rebar {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
            )
            
            # Background for text
            cv2.rectangle(
                image,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                BOX_COLOR,
                -1,
            )
            
            # Text
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),  # Black text on green background
                1,
            )
        
        # Add detection count
        count_text = f"Detected: {detections['count']} rebars"
        cv2.putText(
            image,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            TEXT_COLOR,
            2,
        )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"Saved annotated image to: {output_path}")
        
        # Show if requested
        if show:
            cv2.imshow("Rebar Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image, detections


def detect_single_image(
    image_path: str,
    model_type: str = "gp8000",
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    confidence: float = 0.25,
    show: bool = False,
) -> dict:
    """
    Convenience function to detect rebars in a single image.
    
    Args:
        image_path: Path to the input image
        model_type: 'gp8000' or 'gssi'
        model_path: Custom model path (overrides model_type)
        output_path: Path to save annotated image
        confidence: Confidence threshold
        show: Whether to display the result
        
    Returns:
        Detection results dictionary
    """
    if model_path is None:
        model_path = DEFAULT_MODELS.get(model_type)
        if model_path is None:
            raise ValueError(f"Unknown model type: {model_type}")
    
    detector = RebarDetector(model_path, confidence_threshold=confidence)
    
    if output_path or show:
        _, results = detector.detect_and_visualize(image_path, output_path, show)
    else:
        results = detector.detect(image_path)
    
    return results


def detect_batch(
    input_dir: str,
    output_dir: str,
    model_type: str = "gp8000",
    model_path: Optional[str] = None,
    confidence: float = 0.25,
) -> List[dict]:
    """
    Detect rebars in all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save annotated images
        model_type: 'gp8000' or 'gssi'
        model_path: Custom model path (overrides model_type)
        confidence: Confidence threshold
        
    Returns:
        List of detection results for each image
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if model_path is None:
        model_path = DEFAULT_MODELS.get(model_type)
    
    detector = RebarDetector(model_path, confidence_threshold=confidence)
    
    results = []
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    
    for img_file in sorted(input_path.iterdir()):
        if img_file.suffix.lower() in image_extensions:
            out_file = output_path / f"detected_{img_file.name}"
            _, detection = detector.detect_and_visualize(
                str(img_file), str(out_file), show=False
            )
            detection["image"] = img_file.name
            results.append(detection)
            print(f"Processed {img_file.name}: {detection['count']} rebars detected")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect rebars in GPR images")
    parser.add_argument(
        "input",
        type=str,
        help="Input image path or directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for annotated image(s)",
    )
    parser.add_argument(
        "--model-type", "-t",
        type=str,
        choices=["gp8000", "gssi"],
        default="gp8000",
        help="Model type to use",
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Custom model path (overrides --model-type)",
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (0-1)",
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display results (single image only)",
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all images in input directory",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ConcreteNet - Rebar Detection")
    print("="*60)
    
    if args.batch:
        # Batch processing
        if args.output is None:
            args.output = "detection_results"
        
        results = detect_batch(
            args.input,
            args.output,
            model_type=args.model_type,
            model_path=args.model_path,
            confidence=args.confidence,
        )
        
        # Summary
        total_rebars = sum(r["count"] for r in results)
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"Processed {len(results)} images")
        print(f"Total rebars detected: {total_rebars}")
        print(f"Results saved to: {args.output}/")
        
    else:
        # Single image processing
        results = detect_single_image(
            args.input,
            model_type=args.model_type,
            model_path=args.model_path,
            output_path=args.output,
            confidence=args.confidence,
            show=args.show,
        )
        
        print(f"\nImage: {args.input}")
        print(f"Rebars detected: {results['count']}")
        
        if results["count"] > 0:
            print("\nDetections:")
            for i, (box, conf) in enumerate(zip(results["boxes"], results["confidences"])):
                print(f"  {i+1}. Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}], "
                      f"Confidence: {conf:.3f}")


if __name__ == "__main__":
    main()
