# Table of Contents
- [Training the YOLOv8 model (how these weights were created)](#training-the-yolov8-model-how-these-weights-were-created)
  - [1) Install & imports](#1-install--imports)
  - [2) Dataset preparation — UECFOOD256 ➜ `new_data/`](#2-dataset-preparation--uecfood256--new_data)
  - [3) Build `data.yaml` from `category.txt`](#3-build-datayaml-from-categorytxt)
  - [4) Train (resume‑style or fresh)](#4-train-resume-style-or-fresh)
  - [5) Evaluate on the test split](#5-evaluate-on-the-test-split)
  - [6) Download trained artifacts from Kaggle](#6-download-trained-artifacts-from-kaggle)
- [Food Recognition & Nutrition Analyzer — Documentation](#food-recognition--nutrition-analyzer--documentation)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Configuration](#configuration)
  - [Input File Formats](#input-file-formats)
    - [`nutrition_256.csv` (required)](#nutrition_256csv-required)
    - [`density_overrides_template.csv` (optional)](#density_overrides_templatecsv-optional)
  - [Pipeline Details](#pipeline-details)
    - [1) Detection (YOLOv8)](#1-detection-yolov8)
    - [2) Masks](#2-masks)
    - [3) Depth → Height → Volume → Weight](#3-depth--height--volume--weight)
    - [4) Nutrition](#4-nutrition)
    - [5) Outputs](#5-outputs)
  - [Running the Script](#running-the-script)
    - [Single Image](#single-image)
    - [Batch Folder](#batch-folder)
  - [Calibration & Tuning](#calibration--tuning)
  - [Troubleshooting](#troubleshooting)
  - [Notes on Models & Downloads](#notes-on-models--downloads)
  - [Quick Reference of Key Functions](#quick-reference-of-key-functions)
  - [Example Output](#example-output)

# Training the YOLOv8 model (how these weights were created)

This section summarizes the steps used in your notebook **food-detection-with-yolov8-version-2.ipynb** to train the detector that the nutrition pipeline consumes.

## 1) Install & imports
```python
%pip install ultralytics  # or: pip install ultralytics
from ultralytics import YOLO
```

## 2) Dataset preparation — UECFOOD256 ➜ `new_data/`
The notebook **builds a full YOLO‑ready dataset** from UECFOOD256 and writes it under `new_data/` with the expected Ultralytics layout:
```
new_data/
  train/
    images/*.jpg
    labels/*.txt
  val/
    images/*.jpg
    labels/*.txt
  test/
    images/*.jpg
    labels/*.txt
```
Key steps performed in the notebook:

1. **Read class names** from `category.txt` (ID → name) and generate the ordered `names` list (IDs 1..256 → indices 0..255).
2. **Parse UECFOOD256 annotations** and convert each bounding box to YOLO format:
   - `cls_id = original_id - 1`
   - Normalize to image size: `x_c = (x1+x2)/2W`, `y_c = (y1+y2)/2H`, `w = (x2-x1)/W`, `h = (y2-y1)/H`
   - Skip invalid/empty boxes and clamp to `[0,1]`.
3. **Split the dataset** (e.g., 80/10/10) with class‑balanced shuffling.
4. **Write files** into `new_data/{train,val,test}/images` and `labels` (one `.txt` per image).
5. **(Optional)**: de‑duplicate images, fix missing files, and drop boxes with tiny area.

Example conversion snippet used in the notebook (conceptual):
```python
from pathlib import Path
import cv2, shutil

ROOT = Path("/kaggle/working")
OUT  = ROOT/"new_data"
for split in ["train","val","test"]:
    (OUT/split/"images").mkdir(parents=True, exist_ok=True)
    (OUT/split/"labels").mkdir(parents=True, exist_ok=True)

# for each image
img = cv2.imread(str(img_path))
H, W = img.shape[:2]
yolo_lines = []
for (x1,y1,x2,y2,cls_id1) in boxes_from_uec:
    cls = cls_id1 - 1
    x_c = ((x1+x2)/2) / W
    y_c = ((y1+y2)/2) / H
    w   = (x2-x1) / W
    h   = (y2-y1) / H
    if w <= 0 or h <= 0: 
        continue
    yolo_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

# write image and label
shutil.copy2(img_path, OUT/split/"images"/img_path.name)
(OUT/split/"labels"/img_path.with_suffix('.txt').name).write_text("
".join(yolo_lines))
```

> **Why this matters**: Ultralytics expects one `.txt` per image with **normalized** boxes and class indices starting at 0. Creating `new_data/` in this format lets training and validation run without custom dataloaders.

## 3) Build `data.yaml` from `category.txt`
The notebook parses `category.txt` and writes a YOLO config pointing to the `new_data` splits and the ordered class names.

```python
from pathlib import Path
import yaml

# names built from category.txt earlier
names = [...]

data = {
    "train": "/kaggle/working/new_data/train/images",
    "val":   "/kaggle/working/new_data/val/images",
    "test":  "/kaggle/working/new_data/test/images",
    "nc": len(names),
    "names": names,
}
Path("/kaggle/working/data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
```

## 4) Train (resume‑style or fresh)
The notebook loads the last checkpoint and continues training. You can also start from a base model (e.g., `yolov8m.pt`).

```python
# Resume style
model = YOLO("/kaggle/input/last-checkpoint/last.pt")  # or base: YOLO('yolov8m.pt')

model.train(
    data="/kaggle/working/data.yaml",
    epochs=25,
    imgsz=640,
    seed=42,
    patience=20,
    device="0,1",
    save=True,
    batch=30,
    save_period=10,
    project="runs",
    name="food256_yv8m_ddp_640",
    workers=4,
)
```

**Your run summary**
- You initially trained **`yolov8m.pt` for 50 epochs**.
- Then you **resumed training for +25 epochs** to improve the model.
- During the resumed run, the **best weights were found at epoch 1** of the +25.
- Training **terminated at epoch 21** of the +25 due to early stopping (**`patience=20`**: no improvement for the last 20 epochs).

**Outputs**: `runs/detect/food256_yv8m_ddp_640/weights/best.pt` — consumed by the nutrition pipeline below.

## 5) Evaluate on the test split
```python
model = YOLO("/kaggle/working/runs/food256_yv8m_ddp_640/weights/best.pt")
metrics = model.val(conf=0.25, split='test')
print(metrics)
```

## 6) Download trained artifacts from Kaggle
All files and folders produced by **train_yolov8m.ipynb** (including runs, weights, and data yaml) are published here:

**Kaggle Dataset:** https://kaggle.com/datasets/b70f785b5f4e21301154def41a4a5205681567d9d502eedf5320049f5f16ad8e

> Tip: After downloading, place `weights/best.pt` under `runs/.../weights/` or update the `WEIGHTS` path in `nutrition_pipeline.py`.

---



# Food Recognition & Nutrition Analyzer — Documentation

## Overview
This script runs an end‑to‑end pipeline to estimate **nutritional facts from a single image**:

1. **Detect foods** with YOLOv8.
2. **Build pixel masks** for each detection (use segmentation masks if present; otherwise derive using GrabCut from boxes).
3. Estimate **relative depth** with MiDaS and convert to **height → volume → weight** using a plate-based pixel scale and class densities.
4. Convert per‑item weight to **calories, protein, carbs, fat** using a nutrition table.
5. Save **per‑item and total CSVs** and an **annotated image** with boxes + mask contours.

---

## Requirements
- Python 3.9–3.11
- Packages:
  ```bash
  pip install ultralytics torch torchvision opencv-python pandas numpy
  ```
- Trained YOLO weights (`best.pt`) covering your 256 classes.
- CSV inputs:
  - **nutrition_256.csv** – per‑100 g macros for every label.
  - **density_overrides_template.csv** (optional) – label→density (g/cm³) overrides.

---

## Configuration
Edit the **CONFIG** block at the top of the script:

```python
BASE = r"E:\Food Recognition & Nutrition Analyzer"
WEIGHTS = f"{BASE}/runs/food256_yv8m_ddp_640/weights/best.pt"
NUTRITION_CSV = f"{BASE}/nutrition_256.csv"
DENSITY_CSV   = f"{BASE}/density_overrides_template.csv"
TEST_IMAGE = r".../images/test/96.jpg"
TEST_DIR   = r"E:\some\folder\with\images"
CONF = 0.25                 # detection threshold
DEVICE = 0                  # YOLO device id or "cpu"
IMG_SIZE = 640
DEFAULT_PLATE_DIAM_CM = 10  # used to estimate cm/px from a plate
MODEL_NAME = "DPT_Hybrid"   # MiDaS depth model (or "DPT_Large" / "MiDaS_small")
```

**Tip:** If no plate is visible, the script falls back to a heuristic pixel scale. For reproducibility, you can lock a fixed scale based on your camera.

---

## Input File Formats

### nutrition_256.csv (required)
Lower‑cased food names must match your YOLO class names. Units are **per 100 g**.
```csv
food,calories,protein,carbs,fat
rice,130,2.7,28.9,0.3
toast,265,9,49,3.2
...
```

### density_overrides_template.csv (optional)
If a label is missing, density defaults to **1.0 g/cm³**.
```csv
label,density
rice,0.85
toast,0.27
chicken,1.05
salad,0.25
...
```

---

## Pipeline Details

### 1) Detection (YOLOv8)
- Runs `det.predict(...)` with your weights.
- Applies **cross‑class Non‑Max Suppression (NMS)** so a single object doesn’t get multiple labels (e.g., “teriyaki grilled fish” vs “boiled fish”).

### 2) Masks
- If your YOLO model outputs **segmentation masks**, they are used directly.
- Otherwise, the script calls **GrabCut** on each box to produce a binary mask; if GrabCut fails, it falls back to a slightly shrunken rectangle.
- To avoid double‑counting pixels when foods overlap, masks are assigned **largest area → smallest area** (base foods like rice claim pixels first). A safeguard keeps at least 25% of a detection’s original mask if overlap removal removes too much.

### 3) Depth → Height → Volume → Weight
- MiDaS provides a **relative depth map** (not in cm). For each mask:
  - Compute a **ring** around the object to get a background/plate reference depth.
  - **Auto‑select height sign** (closer vs farther than ring) to ensure positive bumps.
  - Normalize such that the **95th percentile** of the mask’s height ≈ **2.3 cm** (tunable; clamps to 0–5 cm to suppress spikes).
  - Convert pixel area to cm² using the **plate‑based pixel scale**.
  - Weight (g) = volume (cm³) × **density** (g/cm³).

### 4) Nutrition
- For each label, the script looks up per‑100 g macros from `nutrition_256.csv` and scales by `weight_g/100` to produce per‑item calories, protein, carbs, and fat.

### 5) Outputs
- `out/<image_stem>/<image_stem>_annot.jpg` — annotated image with boxes, confidences, and mask contours.
- `out/<image_stem>/results_breakdown.csv` — per‑item rows: label, estimated grams, calories, protein, carbs, fat.
- `out/<image_stem>/results_summary.csv` — totals across the image.

> The code creates a **per‑image folder** automatically inside `BASE/out/`, named by the image stem (e.g., `out/96/`). Both the annotated image and CSV reports are saved there.

---

## Running the Script

### Single Image
```python
if os.path.isfile(TEST_IMAGE):
    df, total = run_one_image(TEST_IMAGE)
    print(df)
    print(total)
```

### Batch Folder
Set `TEST_DIR` to a folder path; the script will process all images and save:
- `batch_results_breakdown.csv` (all items across images)
- `batch_results_summary.csv` (grand totals)

---

## Calibration & Tuning

1. **Densities matter** (strongest lever on grams):
   - Add realistic densities per class, e.g., `rice=0.85`, `toast=0.27`, `noodles=0.55`, `chicken=1.05`, `salad=0.25`.
2. **Expected height** (inside `estimate_portion`):
   - `expected_peak_cm = 2.0–2.5` for flatter dishes; raise toward `3.0–3.5` if under‑estimating.
3. **Plate scale** (`DEFAULT_PLATE_DIAM_CM`):
   - Typical dinner plates are 24–27 cm. If values look high/low, set this accurately or fix a constant cm/px for your setup.
4. **Masks**:
   - Increase morphological cleanup/erosion if masks bleed into background, or use a YOLO **segmentation** model to skip GrabCut.

Prints like these help debugging:
```
[DEBUG] px_to_cm ≈ 0.05800 cm/px  (image HxW=800x1067)
[DIAG] rice | area_px=32145 obj_med=0.24513 ring_med=0.26045 h95=0.00453 px_to_cm=0.05800 vol_cm3≈122.7 density=0.85 -> weight_g≈104.3
```

---

## Troubleshooting

- **Weights far too large** (e.g., kilograms):
  - Plate scale wrong → check `[DEBUG] px_to_cm`.
  - Height normalization too high → lower `expected_peak_cm`.
  - Mask bleed → tighten GrabCut or switch to segmentation masks.

- **Zero weights**:
  - Empty mask → lower `MIN_AREA` or rely on rectangle fallback.
  - Very flat scene → the function already uses an epsilon for `h95`; ensure ring has enough pixels (dilation iterations).

- **Duplicate labels on one food**:
  - Adjust cross‑class NMS IoU threshold (default 0.60) to 0.55–0.70.

---

## Notes on Models & Downloads
- `DPT_Large` downloads ~1.3 GB; use `DPT_Hybrid` for a balance of accuracy and size, or `MiDaS_small` for speed.
- Torch Hub caches weights; you can pre‑download in restricted environments.

---

## Quick Reference of Key Functions
- `run_one_image(path)`: full pipeline + save annotated image.
- `save_annotated_image(...)`: draws boxes, confidences, and mask contours.
- `estimate_depth(image_bgr)`: MiDaS inference → image‑sized depth.
- `estimate_portion(mask, depth_map, label, img_shape, px_to_cm)`: height normalization & density to grams.
- `_estimate_plate_px_to_cm(img)`: cm/px via plate detection or heuristic.
- `load_nutrition_data(csv)`, `load_density_map(csv)`, `calculate_nutrition(...)`.

---

## Example Output
```
🖼️  Saved annotated image -> E:\Food Recognition & Nutrition Analyzer\out\96\96_annot.jpg
[DEBUG] px_to_cm ≈ 0.02793 cm/px  (image HxW=600x800)
[DIAG]                 rice | area_px= 83195 obj_med=2165.60840 ring_med=1518.26099 h95=958.10992 px_to_cm=0.02793 vol_cm3≈91.3 density=0.85 -> weight_g≈77.6
[DIAG]          boiled fish | area_px= 41978 obj_med=2456.50513 ring_med=2123.51904 h95=470.90112 px_to_cm=0.02793 vol_cm3≈48.6 density=1.02 -> weight_g≈49.6

Per-Food Breakdown:
           food  weight_g  calories  protein    carbs     fat
0         rice      77.6    100.88   2.0952  21.8832  0.2328
1  boiled fish      49.6     64.48  10.9120   0.0000  1.9840

Total Nutrition:
 {'calories': 165.36, 'protein': 13.0072, 'carbs': 21.8832, 'fat': 2.2168}

✅ Results saved as 'out/96/results_breakdown.csv' and 'out/96/results_summary.csv'

Per-Food breakdown:
           food  weight_g  calories  protein    carbs     fat
0         rice      77.6    100.88   2.0952  21.8832  0.2328
1  boiled fish      49.6     64.48  10.9120   0.0000  1.9840

Totals:
 {'calories': 165.36, 'protein': 13.0072, 'carbs': 21.8832, 'fat': 2.2168}
```
