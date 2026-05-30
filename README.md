# voxel-nn: 3D Object Detection for Industrial Assemblies

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen.svg)]()

> **3D single-stage object detector for automated component detection in mechanical assemblies.**  
> Detects components (isolators, fasteners, and other parts) in voxelized 3D models. Full pipeline: SolidWorks assembly → voxelization → neural network inference → Excel report with component specifications.

---

## Quick Start

### Installation
```bash
git clone https://github.com/Gezisufa/voxel-nn.git
cd voxel-nn
pip install -r requirements.txt
```

### Minimal Example (Inference)
```bash
cd single-stage
python infer.py
# Outputs: detections_report.xlsx + infer_output.png
```

### Full Pipeline (SolidWorks → Inference → Report)
```bash
# 1. Preprocess assembly (Windows + SolidWorks required)
python sw_preprocess.py

# 2. Run inference
python infer.py

# 3. View report → detections_report.xlsx
```

---

## Pipeline Overview
┌─ SolidWorks .sldasm ─────────────────────────┐
│                                               │
├─→ sw_preprocess.py (Windows only)            │
│   ├─ Suppress known components              │
│   ├─ Export assembly as STL                 │
│   └─ Log specs to Excel                     │
│                                               │
├─→ infer.py                                   │
│   ├─ Voxelize STL (1 mm/voxel)             │
│   ├─ Sliding-window inference               │
│   └─ 3D NMS post-processing                │
│                                               │
├─→ VoxelOneStage3D (3D CNN)                   │
│   ├─ Input: 500³ binary occupancy grid      │
│   ├─ Output: 50³ detection grid (stride 10) │
│   └─ Pred: objectness, center, size, class │
│                                               │
├─→ export_excel.py                           │
│   └─ Merge known + detected → Report        │
│                                               │
└─→ detections_report.xlsx ◀──────────────────┘

---

## Model Architecture

### VoxelOneStage3D — Anchorless 3D Detector

| Aspect | Details |
|--------|---------|
| **Input** | `(B, 1, 500, 500, 500)` binary occupancy |
| **Backbone** | Conv3d + GroupNorm + SiLU (5 blocks) |
| **Detection Head** | 1×1×1 Conv3d → `(B, 7+K, 50, 50, 50)` |
| **Grid Stride** | 10 voxels |
| **Grid Size** | 50 × 50 × 50 detection cells |
| **Classes** | K (configurable, default 10) |

### Per-Cell Prediction
Each grid cell predicts:
- **Objectness**: σ(logit) ∈ [0, 1]
- **Center offset**: Δ(x, y, z) ∈ [-5, +5] voxels
- **Size**: s(x, y, z) continuous, base 30 voxels
- **Classification**: K class logits

### Loss Function
Total Loss = α·L_objectness + β·L_center + γ·L_size + δ·L_class
L_objectness  : BCE with hard negative mining (neg/pos = 5:1)
L_center      : SmoothL1 (active cells only)
L_size        : SmoothL1 in log space (active cells only)
L_class       : CrossEntropy (positive cells only)

---

## Project Structure
voxel-nn/
├── single-stage/                  ← Main detector pipeline
│   ├── production/
│   │   └── model.py               # VoxelOneStage3D architecture
│   ├── training/
│   │   ├── assigner.py            # Dense grid target builder
│   │   ├── losses.py              # 4-component combined loss
│   │   └── trainer.py             # Training loop (AMP, gradient clipping)
│   ├── train.py                   # Training entry point
│   ├── validate.py                # Validation metrics + 3D visualization
│   ├── infer.py                   # Sliding-window inference (STL/NPZ/PNG)
│   ├── convert_data.py            # PNG slices + JSON → NPZ dataset
│   ├── sw_preprocess.py           # SolidWorks COM automation (Windows)
│   ├── export_excel.py            # Format detections → Excel report
│   ├── example.py                 # Minimal NPZ inference example
│   └── checkpoints/               # Model weights (ignored by .gitignore)
├── data/
│   └── svx/                       # Raw voxel density PNG slices (input)
├── evaluation/
│   └── BBtoSlices.py              # Visualize bounding boxes on 2D slices
├── requirements.txt               # Python dependencies
└── README.md                      # This file

---

## Installation & Setup

### Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA** (optional, for GPU inference/training)
- **SolidWorks** (Windows only, required for `sw_preprocess.py`)

### Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `pywin32` is included for Windows/SolidWorks integration.

---

## Detailed Usage

### 1️. Data Preparation (Training)

**Training dataset format** — one directory per sample:
data/dataset/
└── sample_001/
├── input.npz           # Binary volume (500×500×500)
└── annotation.txt      # One box per line

**Annotation format** (`annotation.txt`):
class_id  center_x  center_y  center_z  size_x  size_y  size_z
0         250       250       250        50      50      50
1         100       100       100        30      30      30

**Convert raw SolidWorks export to dataset format:**

```bash
cd single-stage
python convert_data.py
```

---

### 2️. Training

```bash
cd single-stage
python train.py
```

**Key Parameters** (edit in `train.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_classes` | 10 | Number of component types |
| `epochs` | 1000 | Total training epochs |
| `batch_size` | 8 | Batch size per GPU |
| `lr` | `1e-4` | Learning rate (AdamW) |
| `weight_decay` | `1e-4` | L2 regularization |
| `num_workers` | 2 | DataLoader parallel workers |

**Output:**
- `checkpoints/best.pt` — best model by validation IoU
- `logs/log.txt` — training log

---

### 3️. Inference

```bash
cd single-stage
python infer.py
```

**Edit CONFIG section at top of `infer.py`:**

| Setting | Default | Description |
|---------|---------|-------------|
| `INPUT_MODE` | `"stl"` | Format: `"stl"`, `"npz"`, or `"png"` |
| `STL_PATH` | `in/assembly.stl` | Input file path |
| `CKPT_PATH` | `checkpoints/best.pt` | Model checkpoint |
| `WINDOW_SIZE` | 500 | Sliding window size (voxels) |
| `OVERLAP` | `(450, 450, 450)` | Per-axis overlap |
| `THRESHOLD` | 0.2 | Objectness score threshold |
| `NMS_IOU_THR` | 0.3 | 3D NMS IoU threshold |

**Outputs:**
out/
├── infer_output.png              # 3D scatter plot
├── detections_raw.xlsx           # Raw predictions
└── detections_report.xlsx        # Formatted report

---

### 4️. SolidWorks Preprocessing (Windows Only)

```bash
cd single-stage
python sw_preprocess.py
```

**Pipeline:**
1. Open assembly (`.sldasm`)
2. Apply Display State (e.g., `"HV only"`)
3. Suppress components in `SUPPRESS_LIBRARY`
4. Export as STL + log specs to Excel
5. (Optional) Auto-run inference (`AUTO_RUN_INFER = True`)

---

### 5️. Validation

```bash
cd single-stage
python validate.py
```

**Output:** `val_vis/` — GT boxes (green) vs. predictions (red)

---

## . Excel Report Format

**detections_report.xlsx** contains two row types:

| Type | Color | Source |
|------|-------|--------|
| **Known Component** | Grey | `SUPPRESS_LIBRARY` |
| **NN Detection** | Green | Neural Network |

**Columns:** Component, Class_ID, Quick_Case, Size_of_bolt, Numbers_of_tightenings, Nm

**To add/update specs:**
Edit `CLASS_LIBRARY` in `export_excel.py`:

```python
CLASS_LIBRARY = {
    0: {"name": "M8_fastener", "bolt_size": "M8", "torque": 25},
    1: {"name": "M12_fastener", "bolt_size": "M12", "torque": 75},
}
```

---

## Input Formats

### STL Files
```python
INPUT_MODE = "stl"
STL_PATH = "in/assembly.stl"
# Automatically voxelized at 1 voxel = 1 mm
```

### Pre-voxelized NPZ
```python
INPUT_MODE = "npz"
NPZ_PATH = "data/sample_001/input.npz"
```

### PNG Slices
```python
INPUT_MODE = "png"
PNG_DIR = "data/svx/"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | `pip install torch torchvision` |
| `sw_preprocess.py` crashes | Ensure SolidWorks is open & `pywin32` installed |
| Out of memory on GPU | Reduce `batch_size` or `WINDOW_SIZE` |
| Poor accuracy | Check data format, ensure annotations are correct |
| Missing detections | Lower `THRESHOLD` (e.g., 0.1 instead of 0.2) |

---

## References

- 3D ShapeNets (Wu et al., 2015)
- VoxNet (Maturana & Scherer, 2015)
- YOLO (Redmon et al.) — anchor-free detection
- PointNet (Qi et al., 2017)
- Minkowski Engine (Choy et al., 2019)

---

## Publication

**White Paper:** "Dense 3D Detection from CAD-Derived Volumetric Data"
- **Authors:** Ing. Serhii Andriievskyi, Ing. Matouš Cejnek, Ph.D.
- **Event:** Studentská tvůrčí činnost 2026
- **Institution:** Czech Technical University in Prague

---

## Author & Contact

**Serhii Andriievskyi**
- Ph.D. Candidate, AI for Engineering Systems (CTU Prague)
- andriser@cvut.cz

**Supervisor:**  
Ing. Matouš Cejnek, Ph.D. — Czech Technical University in Prague

---

## License

MIT License

---

**Last Updated:** May 2026  
**Status:** Active research project
