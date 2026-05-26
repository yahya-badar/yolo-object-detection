# 🎯 Custom YOLO Object Detection System

> A from-scratch PyTorch implementation of a YOLO-inspired real-time object detection system, built as a university AI course project.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📸 Sample Detections

| Airport Scene | Living Room | Farm Animals |
|---|---|---|
| Persons, Suitcase detected | Couch, Chair, Potted Plant | Cows detected with bounding boxes |

> Model trained on COCO8 (5K image subset). Red = Predicted boxes, Green = Ground Truth boxes.

---

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Results](#training-results)
- [Dataset](#dataset)
- [Team](#team)

---

## 🧠 About The Project

This project implements a **custom YOLO-style object detection system** from scratch using PyTorch — no Ultralytics library, no pretrained backbone shortcuts. Every component was written manually:

- Custom **Backbone** using `Conv` → `C2f` → `SPPF` blocks
- Custom **Neck** (Feature Pyramid Network style) with upsampling and C2f aggregation
- Custom **Detection Head** with DFL (Distribution Focal Loss) for precise bounding boxes
- Custom **Loss Function**: CIoU box loss + BCE classification loss + DFL
- Custom **Training Loop** with AdamW optimizer and Automatic Mixed Precision (AMP)
- Custom **NMS** (Non-Maximum Suppression) post-processing

**Key Technical Choices:**
| Component | Choice | Reason |
|---|---|---|
| Activation | SiLU | Smoother gradients than ReLU |
| Feature Blocks | C2f (Cross-Stage Partial) | Better gradient flow, fewer params |
| Box Regression | CIoU Loss | More precise than standard IoU |
| Precision | torch.amp (Mixed Precision) | Faster training, prevents gradient underflow |
| Optimizer | AdamW | Better weight decay handling |

---

## 🏗️ Architecture

```
Input Image (640×640)
        │
   ┌────▼─────┐
   │ BACKBONE │  Conv → C2f → Conv → C2f → Conv → C2f → Conv → C2f → SPPF
   └────┬─────┘  (Feature extraction at scales: /8, /16, /32)
        │  out1, out2, out3
   ┌────▼─────┐
   │   NECK   │  Upsample + Concatenate + C2f  (Feature Pyramid aggregation)
   └────┬─────┘  (Multi-scale feature fusion)
        │  p3, p4, p5
   ┌────▼─────┐
   │   HEAD   │  Box branch + Class branch + DFL
   └────┬─────┘  (Strides: 8, 16, 32 → 8400 total anchors)
        │
   Detections [x1, y1, x2, y2, confidence, class_id]
```

### Module Summary

| Module | Description |
|---|---|
| `Conv` | Conv2d + BatchNorm2d + SiLU activation |
| `Bottleneck` | Two Conv blocks with optional residual skip connection |
| `C2f` | Cross-Stage Partial block: splits channels, runs through N Bottlenecks, concatenates |
| `SPPF` | Spatial Pyramid Pooling Fast: multi-scale max pooling for global context |
| `Backbone` | Stacked Conv + C2f blocks with progressively doubled channels |
| `Neck` | FPN-style: upsamples deep features and fuses with shallow features |
| `Head` | Separate box + class prediction branches per scale, DFL for box refinement |
| `DFL` | Distribution Focal Loss layer for sub-pixel bounding box precision |
| `Assigner` | Task-Aligned Assigner: matches predictions to ground truth boxes |
| `ComputeLoss` | Combines CIoU + BCE + DFL losses with configurable weights |

---

## 📁 Project Structure

```
yolo-object-detection/
│
├── 📓 yolo_final.ipynb          # Main training notebook (full pipeline)
├── 📓 yolo_inference.ipynb      # Inference & visualization notebook
├── 📓 yolo.ipynb                # Experimental/development notebook
│
├── 🐍 plot_training_history.py  # Script to plot loss curves from losshistory.txt
│
├── 📊 results/
│   └── losshistory.txt          # Training loss log (epoch, batch, total, box, cls, dfl)
│
├── 📄 docs/
│   ├── Project_Report.pdf       # Full project report
│   └── Project_Proposal.pdf     # Original project proposal
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files excluded from Git
└── README.md                    # This file
│
│  ── NOT UPLOADED (add locally after cloning) ──
├── best/best.pt                 # Trained model weights  [~50MB, not on GitHub]
├── coco8/                       # Dataset images + labels [too large for GitHub]
└── losshistory.txt              # Raw loss log (copy to results/)
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/yolo-object-detection.git
cd yolo-object-detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
This project uses **COCO8** — a curated 5K-image subset of the COCO dataset.

```bash
# Option 1: Download via Ultralytics (easiest)
pip install ultralytics
python -c "from ultralytics.data.utils import download; download('https://ultralytics.com/assets/coco8.zip')"

# Option 2: Manual download from https://cocodataset.org
# Place images in:  ./coco8/images/train2017/
# Place labels in:  ./coco8/labels/train2017/
```

Expected directory structure:
```
coco8/
├── images/
│   ├── train2017/   ← training images (.jpg)
│   └── val/         ← validation images (.jpg)
└── labels/
    ├── train2017/   ← labels (.txt, YOLO format)
    └── val/
```

### 5. (Optional) Download pretrained weights
Pre-trained weights are not hosted on GitHub due to file size. You can:
- Train from scratch (see Training section below)
- Contact the authors for the `best.pt` checkpoint

Place the weights file at: `./best/best.pt`

---

## 🚀 Usage

### Run Inference (on pre-trained weights)

Open **`yolo_inference.ipynb`** in Jupyter and run all cells.

The notebook will:
1. Load the model from `./best/best.pt`
2. Run inference on images in `./coco8/images/val/`
3. Display bounding boxes with class names and confidence scores
4. Compare predictions (Red) against Ground Truth (Green)

### Train From Scratch

Open **`yolo_final.ipynb`** in Jupyter and run all cells sequentially.

Key training parameters (configurable in Block 13):
```python
params = {
    'min_lr': 0.0001,      # Minimum learning rate
    'max_lr': 0.01,        # Maximum learning rate
    'momentum': 0.937,     # SGD momentum
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'box': 6.5,            # Box loss gain
    'cls': 1.5,            # Classification loss gain
    'dfl': 2.5,            # DFL loss gain
}
```

### Plot Training Loss Curves

```bash
python plot_training_history.py
```

Make sure `losshistory.txt` is in the same directory. This generates a log-scale plot of Total Loss, Box Loss, Class Loss, and DFL Loss across training steps.

---

## 📊 Training Results

### Loss Curves (Log Scale)

The model was trained for ~10 epochs on the 5K COCO image subset with batch size ~16.

| Loss Type | Start (Epoch 0) | End (Epoch 10) | Trend |
|---|---|---|---|
| Total Loss | ~7210 | ~5.0 | ✅ Rapid convergence |
| Box Loss | ~3.82 | ~1.45 | ✅ Stable decrease |
| Class Loss | ~7202 | ~2.1 | ✅ Large improvement |
| DFL Loss | ~4.21 | ~1.55 | ✅ Stable decrease |

> **Note:** The initial spike in Total Loss (visible on the training graph) is due to the large Class Loss at the start of training, which is expected behavior. The model stabilizes after approximately 200–300 batches.

### Detection Examples

The trained model successfully detects across all 80 COCO classes. Sample confident detections:
- `umbrella` — 0.93 confidence
- `couch` — 0.88 confidence
- `refrigerator` — 0.83 confidence
- `cow` — 0.78 confidence
- `wine glass` — 0.81 confidence

### Known Limitations

- Model is trained on a small subset (5K images vs full COCO's ~118K) — confidence scores are lower than production YOLO models
- Real-time inference requires a GPU. CPU inference is significantly slower
- Some crowded scenes show overlapping/missed boxes — expected for a compact model at this training scale

---

## 📦 Dataset

**COCO8** — a curated subset of [Microsoft COCO](https://cocodataset.org/)

| Property | Value |
|---|---|
| Total Images | ~5,000 |
| Classes | 80 (person, car, dog, cat, ...) |
| Input Resolution | 640 × 640 |
| Format | YOLO format (normalized xywh) |
| Augmentations | Mosaic, HSV color-space (planned), MixUp |

The full class list (80 classes) includes: person, car, bicycle, motorcycle, bus, truck, traffic light, cat, dog, horse, cow, elephant, zebra, giraffe, umbrella, handbag, suitcase, bottle, wine glass, cup, fork, spoon, bowl, pizza, hot dog, cake, chair, couch, potted plant, bed, dining table, tv, laptop, mouse, keyboard, cell phone, refrigerator, book, clock, vase, and more.

---

## 👥 Team

| Name | Student ID | University |
|---|---|---|
| Ibrahim Khan | UW-24-CS-BS-021 | University of Wah |
| Abdul Ahad Abbasi | UW-24-CS-BS-008 | University of Wah |
| M. Yahya Badar | UW-24-CS-BS-020 | University of Wah |

**Course:** Artificial Intelligence
**Instructor:** Sir Hassan Shafqat
**Department:** Computer Science

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

1. Redmon, J., et al. *"You Only Look Once: Unified, Real-Time Object Detection."* CVPR 2016.
2. [PyTorch Documentation](https://pytorch.org/docs/stable/) — `torch.nn`, `torch.amp`
3. [Ultralytics YOLOv8 Architecture](https://github.com/ultralytics/ultralytics) — Architecture reference
4. [COCO Dataset](https://cocodataset.org/) — Training and validation data

---

*Built from scratch with PyTorch — no Ultralytics shortcuts used in the core implementation.*
