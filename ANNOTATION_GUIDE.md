# Rebar Annotation Guide

This guide explains how to annotate GPR images for training the rebar detection models.

## Prerequisites

Install the annotation tool:

```bash
pip install labelImg
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Annotating GP8000 Images

```bash
# Training images
labelimg data/GP8000/images/train data/GP8000/labels/train

# Validation images
labelimg data/GP8000/images/val data/GP8000/labels/val
```

### Annotating GSSI Images

```bash
# Training images
labelimg data/GSSI/images/train data/GSSI/labels/train

# Validation images
labelimg data/GSSI/images/val data/GSSI/labels/val
```

## LabelImg Setup

1. **Launch LabelImg** with the commands above
2. **Set save format to YOLO**:
   - View → Change Save Format → YOLO
3. **Enable Auto Save** (recommended):
   - View → Auto Save Mode
4. **Create the class**:
   - The first time you draw a box, enter `rebar` as the class name
   - This will be saved to `classes.txt` in the labels directory

## Annotation Guidelines

### GP8000 Images (Dark/Faded Spike Patterns)

GP8000 images show rebars as **dark vertical spike patterns** that fade downward.

```
What to annotate:
┌─────────────────────────────────────────┐
│                                         │
│   ▼▼▼  ▼▼▼  ▼▼▼  ▼▼▼  ▼▼▼  ▼▼▼        │
│   ││   ││   ││   ││   ││   ││          │
│   ▽    ▽    ▽    ▽    ▽    ▽           │
│   │    │    │    │    │    │           │
│   ·    ·    ·    ·    ·    ·           │
│                                         │
└─────────────────────────────────────────┘
     ↑
     Each of these spike patterns is ONE rebar
```

**Bounding box placement:**
- **Top**: Start at the top of the dark spike
- **Bottom**: Extend to where the fade pattern becomes indistinguishable from background
- **Left/Right**: Tight around the spike width (include immediate diffraction pattern)

### GSSI Images (Hyperbolic Patterns)

GSSI images show rebars as **hyperbolic signatures** (inverted V or parabola shapes).

```
What to annotate:
┌─────────────────────────────────────────┐
│                                         │
│  \    /   \    /   \    /   \    /     │
│   \  /     \  /     \  /     \  /      │
│    \/       \/       \/       \/       │ ← Apex
│    /\       /\       /\       /\       │
│   /  \     /  \     /  \     /  \      │
│  /    \   /    \   /    \   /    \     │
│                                         │
└─────────────────────────────────────────┘
       ↑
       Each hyperbola is ONE rebar
```

**Bounding box placement:**
- **Top**: Start above where the hyperbola tails begin to fade
- **Bottom**: Include the full apex of the hyperbola
- **Left/Right**: Extend to where the tails fade into the background

## Tips for Accurate Annotation

### General Tips

1. **Be consistent**: Use the same criteria for all images
2. **When in doubt, include it**: Slightly larger boxes are better than missing parts
3. **Overlapping rebars**: Create separate boxes for each rebar, even if they overlap
4. **Faint rebars**: Annotate them if you can distinguish them from noise
5. **Edge rebars**: Annotate rebars that are partially visible at image edges

### Handling Difficult Cases

**Overlapping hyperbolas (GSSI):**
- Draw separate boxes for each identifiable hyperbola
- It's okay if boxes overlap significantly

**Fading patterns (GP8000):**
- Include the fade tail even if it's subtle
- Better to be slightly too large than to miss information

**Very close rebars:**
- If you can distinguish two separate patterns, draw two boxes
- If they're merged into one pattern, draw one box

## YOLO Label Format

LabelImg will automatically create `.txt` files in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1) relative to image dimensions.

Example `image1.txt`:
```
0 0.234 0.456 0.089 0.234
0 0.456 0.423 0.078 0.267
0 0.678 0.489 0.091 0.245
```

Each line represents one rebar (class 0).

## Workflow Summary

1. **Open LabelImg** with the appropriate image and label directories
2. **Set format to YOLO** and enable Auto Save
3. **For each image**:
   - Press `W` to create a new box
   - Draw the bounding box around the rebar
   - Select/type `rebar` as the class
   - Press `D` to go to the next image
4. **Annotate all training images** (most important)
5. **Annotate all validation images** (for evaluation)

## Keyboard Shortcuts (LabelImg)

| Key | Action |
|-----|--------|
| W | Create a rect box |
| D | Next image |
| A | Previous image |
| Del | Delete selected box |
| Ctrl+S | Save |
| Ctrl+D | Duplicate selected box |

## After Annotation

Once you've annotated all images, you can train the models:

```bash
# Train both models
python train_models.py

# Train only GP8000
python train_models.py --model gp8000

# Train only GSSI
python train_models.py --model gssi
```

## Annotation Statistics

| Dataset | Training Images | Validation Images |
|---------|-----------------|-------------------|
| GP8000  | 48              | 72                |
| GSSI    | 24              | 36                |

**Estimated annotation time**: 2-5 minutes per image depending on rebar density.
