# ConcreteNet: A Deep Convolutional Neural Network for Deformity Detection and Classification in Ground-Penetrating Radar (GPR) Images

<p align="center">
  <img src="docs/award proposal/img/Figure1.jpg">
</p>

## Overview

ConcreteNet is a specialized convolutional neural network (CNN) designed to detect and classify structural anomalies in ground-penetrating radar (GPR) images, or radargrams. This repository contains all the source code, training data, and experimental results used in the research paper titled *"ConcreteNet: A Deep Convolutional Neural Network for Deformity Detection and Classification in Ground-Penetrating Radar Images"*. The project leverages a modified AlexNet architecture with a "Network in Network" approach for enhanced feature extraction and classification of structural defects in concrete.

ConcreteNet is built to tackle the challenges of interpreting noisy and complex GPR radargram data for assessing concrete structures. The dataset used for training and validation is sourced from radar scans of the Georgia Southern Engineering Research Building (ERB) and pre-existing GPR data from the Georgia Department of Transportation (GDOT). Additionally, the project includes a publicly accessible dataset and benchmarks state-of-the-art models like ResNet, MobileNet, and others.

---

## Getting Started

### Prerequisites

To run this project, you'll need to have the following software installed:

- Python 3.8+
- TensorFlow or PyTorch (depending on the model you want to run)
- CUDA (optional for GPU acceleration)
- Other dependencies (install via `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anishgoyal1108/ConcreteNet.git
   cd ConcreteNet
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download the GPR radargram dataset (if not included):
   - The dataset is available in the `data/` directory. If you do not have access to the dataset, please contact the repository owner for permission to download.

### Running ConcreteNet

You can train ConcreteNet by running the provided training script with the appropriate configuration:

```bash
python experiments/train_concretenet.py --config config/concretenet.yaml
```

Refer to the `config/` folder for configuration files that specify hyperparameters, dataset paths, and model settings.

### Quick Start

1. Run the desktop app to load images and detect rebars:
   ```bash
   python run.py
   ```

2. Or run detection from the command line:
   ```bash
   python detect_rebars.py image.png --model-type gp8000 --output result.png
   ```

---

## ConcreteNet Desktop App

The desktop application provides an interactive workflow for rebar detection, annotation refinement, and contour analysis.

### Features

- Upload images and run rebar detection (GP8000 or GSSI models)
- Import and overlay contour maps (JSON or PNG)
- **Select mode**: Drag to select boxes/anchors; Del to remove; arrow keys to move
- **Add Box mode (S)**: Draw new boxes for missed rebars
- Anchor points for contour interpolation (default: top-center of each box)
- Spacing analysis with outlier highlighting (6 px deviation threshold)
- Export: project JSON, stats CSV/JSON, PNG (full or annotations only)
- Undo/Redo (Ctrl+Z / Ctrl+Y)

### Running the App

```bash
python run.py
```

Requires PyQt6 and scipy (see `requirements.txt`).

---

## Dataset

The dataset used in this project consists of GPR radargram images collected from two main sources:

1. **Georgia Southern University Engineering Research Building (ERB)**: Recent scans from the building’s floors.
2. **Georgia Department of Transportation (GDOT)**: Pre-existing GPR data used for validation and testing.

### Data Structure

Each GPR scan is processed into a set of radargrams labeled with various structural anomalies (e.g., delamination, corrosion, honeycombing, voids). The dataset is split into training, validation, and testing sets for model evaluation.

The dataset includes images from two GPR devices:
- **GP8000**: Shows rebars as dark/faded spike patterns
- **GSSI**: Shows rebars as hyperbolic signatures

```
data/
├── Raw/                    # Original uncropped images
├── Cleaned/                # Cropped images (main scan area only)
├── GP8000/                 # YOLO-formatted GP8000 dataset
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   └── labels/val/
└── GSSI/                   # YOLO-formatted GSSI dataset
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    └── labels/val/
```

### Public Access

The dataset is publicly accessible to facilitate future research in GPR-based anomaly detection. Benchmarking data can be found in the `data/` folder. Researchers can also use this dataset for comparisons with state-of-the-art models.

---

## Model Architecture

ConcreteNet is built on the AlexNet architecture but employs the "Network in Network" approach to enhance the detection of subtle features in GPR radargrams. The architecture is optimized for classifying various structural defects within concrete structures.

Key Features:

- **Convolutional Layers**: Modified AlexNet backbone with custom convolutional layers for better feature extraction from noisy radargrams.
- **Network in Network**: Incorporates smaller, more specialized convolutional networks within the larger model for increased discriminative power.

### Training

Training ConcreteNet involves GPR images from the ERB building and data augmentation techniques to account for variability in radargram noise and defects.

To train the model from scratch:
```bash
python experiments/train_concretenet.py --config config/concretenet.yaml
```

---

## Rebar Detection (YOLOv8)

This project includes YOLOv8-based object detection for locating rebars in GPR images. Two separate models are trained for the different GPR device formats. The dataset is pre-annotated (from `annotations.xml`).

### Training Rebar Detection Models

```bash
# Train both models
python train_models.py

# Train specific model
python train_models.py --model gp8000
python train_models.py --model gssi

# Customize training
python train_models.py --model both --size s --epochs 150 --batch 8
```

### Running Inference

```bash
# Detect rebars in a single image
python detect_rebars.py image.png --model-type gp8000 --output result.png

# Batch processing
python detect_rebars.py input_folder/ --batch --output results/ --model-type gssi
```

---

## Benchmarking

This repository includes benchmarking scripts to evaluate various state-of-the-art models against the GPR dataset. The following models are included:

- **Image Classification**: MobileNet, DenseNet121, ResNet50
- **Object Detection**: YOLOv11, RetinaNet, Faster-RCNN
- **Weakly Supervised Object Detection (WSOD)**: Wetectron, C-MIL, PCL

To run benchmarking experiments:
```bash
python experiments/benchmark.py --config config/benchmark.yaml
```

---

## Results (TBD)

ConcreteNet achieves state-of-the-art performance on the GPR radargram dataset, with the following key metrics:

- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1-Score**: XX%

Benchmark results show that ConcreteNet outperforms existing models in both classification and anomaly detection tasks, particularly when dealing with noisy radargram data.

Detailed results, including graphs and logs, can be found in the `results/` directory.

---

## Future Work

- **Autoencoders**: Exploring the use of autoencoders for unsupervised feature learning.
- **LSTM Integration**: Incorporating RNNs with Long Short-Term Memory (LSTM) layers to improve classification accuracy using raw waveform data.
- **Dataset Expansion**: Continuing to expand the publicly available GPR radargram dataset to support further research in anomaly detection.

---

## Contributors

- **Anish Goyal**
- **Dr. Hossein Taheri** - [htaheri@georgiasouthern.edu](mailto:htaheri@georgiasouthern.edu)

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
