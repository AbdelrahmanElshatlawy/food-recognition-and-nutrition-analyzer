# ==================================================
# Stage 2 (YOLO boxes) -> Masks -> Stage 3 Nutrition
# ==================================================

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import hashlib

# ===================== CONFIG =====================
BASE = r"E:\Local Repositories\food-recognition-and-nutrition-analyzer"
WEIGHTS = f"{BASE}/food256_yv8m_ddp_640/weights/best.pt"                       # <- your trained detector
NUTRITION_CSV = f"{BASE}/datasets/nutrition_256.csv"                                # <- food,calories,protein,carbs,fat
DENSITY_CSV   = f"{BASE}/datasets/density_overrides_template.csv"                   # <- label,density (optional overrides)
TEST_IMAGE = r"E:\Food Detection Project\Datasets\uec256_yolo\images\test\28.jpg"   # or use TEST_DIR below
CONF = 0.25                                                                         # detection conf threshold
DEVICE = 0                                                                          # YOLO device (0 for first GPU, or "cpu")
IMG_SIZE = 640
DEFAULT_PLATE_DIAM_CM = 10                                                          # assumed plate diameter
# NOTE: we no longer use a fixed DEPTH_TO_CM multiplier (we normalize per-image instead)
# ==================================================

# ------------- Nutrition lookup -------------
def load_nutrition_data(csv_path):
    df = pd.read_csv(csv_path)
    table = {}
    for _, row in df.iterrows():
        table[str(row["food"]).lower()] = {
            "calories": float(row["calories"]),
            "protein":  float(row["protein"]),
            "carbs":    float(row["carbs"]),
            "fat":      float(row["fat"]),
        }
    return table

# ------------- Density lookup (optional) -------------
def load_density_map(csv_path):
    """
    Expect a CSV with columns: label,density
    Returns: dict {lowercased label: float density}
    """
    if not os.path.isfile(csv_path):
        print(f"[WARN] Density CSV not found: {csv_path} -> using empty map (fallback=1.0).")
        return {}
    df = pd.read_csv(csv_path)
    d = {}
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip().lower()
        val = row.get("density", None)
        if label and pd.notna(val):
            try:
                d[label] = float(val)
            except Exception:
                pass
    return d

# Global dicts (lazy-loaded in run_one_image)
food_nutrition = None
DENSITY_MAP = None

def calculate_nutrition(food_label, weight_g):
    if not isinstance(food_label, str):  # safety
        return {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    key = food_label.lower()
    if food_nutrition is None or key not in food_nutrition:
        return {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    per100 = food_nutrition[key]
    f = max(float(weight_g), 0.0) / 100.0
    return {
        "calories": per100["calories"] * f,
        "protein":  per100["protein"]  * f,
        "carbs":    per100["carbs"]    * f,
        "fat":      per100["fat"]      * f,
    }

# ------------- MiDaS depth (robust) -------------
MODEL_NAME = "DPT_Hybrid"   # or "DPT_Large" (bigger) / "MiDaS_small" (fastest)
midas = torch.hub.load("intel-isl/MiDaS", MODEL_NAME, trust_repo=True)

# Avoid name clash with YOLO's DEVICE (int). Use a separate torch_device.
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(torch_device).eval()

# Load transforms module and pick the correct callable
_t = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
if MODEL_NAME in ("DPT_Large", "DPT_Hybrid"):
    _depth_transform = _t.dpt_transform      # expects NumPy RGB, returns dict {'image': tensor/ndarray}
else:  # MODEL_NAME == "MiDaS_small"
    _depth_transform = _t.small_transform

def estimate_depth(image_bgr):
    """
    Returns HxW depth map as float32 numpy array (same size as input).
    Works with MiDaS transforms that take HxWxC uint8 RGB (NumPy) and return a dict with 'image'.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform (expects NumPy; returns dict with 'image')
    sample = _depth_transform(img_rgb)
    x = sample["image"] if isinstance(sample, dict) else sample  # [3,H,W] tensor/ndarray

    # Ensure torch tensor on device, shape [1,3,H,W]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(torch_device)

    with torch.no_grad():
        pred = midas(x)  # [1, H', W'] or similar
        depth = F.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).squeeze(0)

    return depth.detach().cpu().numpy()

# ------------- Plate scale (px -> cm) -------------
def _estimate_plate_px_to_cm(img_bgr, default_plate_diam_cm=DEFAULT_PLATE_DIAM_CM):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    minR = int(0.15 * min(H, W) / 2)
    maxR = int(0.60 * min(H, W) / 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=min(H, W)//4, param1=100, param2=30,
                               minRadius=minR, maxRadius=maxR)

    if circles is not None:
        cx, cy = W/2, H/2
        circles = np.round(circles[0, :]).astype(int)
        # prefer largest and most centered circle
        circles = sorted(circles, key=lambda c: (-(c[2]), (c[0]-cx)**2 + (c[1]-cy)**2))
        r_px = circles[0][2]
        diam_px = 2 * r_px
        if diam_px > 0:
            return default_plate_diam_cm / float(diam_px)

    # Fallback: assume plate ~25% of image area
    plate_area_px = H * W * 0.25
    diam_px = 2.0 * np.sqrt(plate_area_px / np.pi)
    return default_plate_diam_cm / float(diam_px)

# ------------- Portion estimation (normalized & robust) -------------
def estimate_portion(mask, depth_map, food_label, img_shape, px_to_cm):
    """
    Convert relative MiDaS depth to height via per-image normalization:
    - Reference depth from a ring around the object
    - Auto-pick height direction (closer vs farther than ring)
    - Normalize so the 95th percentile inside the mask ~ 3 cm
    - Clamp to [0, 6] cm
    - Sum volume over mask and multiply by density
    """
    if mask is None or not np.any(mask):
        return 0.0

    m = mask.astype(bool)
    H, W = img_shape[:2]

    # Ring around object to estimate reference (plate/table) depth
    m_u8 = m.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(m_u8, kernel, iterations=6).astype(bool)
    ring = dil & ~m
    if np.count_nonzero(ring) < 50:
        ring = ~m  # fallback if ring too small

    # Medians to determine sign (closer vs farther)
    obj_med  = float(np.median(depth_map[m]))
    ring_med = float(np.median(depth_map[ring]))

    # Choose the direction that makes the object a positive "bump"
    if obj_med < ring_med:
        # object is closer (smaller depth) than ring
        height_rel = np.maximum(ring_med - depth_map, 0.0)
    else:
        # object is farther (larger depth) than ring
        height_rel = np.maximum(depth_map - ring_med, 0.0)

    # Robust normalization inside the mask
    h_vals = height_rel[m]
    if h_vals.size == 0:
        return 0.0
    h95 = float(np.percentile(h_vals, 95))

    # If scene is very flat, use a small epsilon so it doesn't collapse to 0
    eps = 1e-4
    if h95 <= eps:
        h95 = eps

    expected_peak_cm = 2.3  # tune 2–4 cm typical for food bump
    MAX_HEIGHT_CM = 5.0
    height_cm = (height_rel / h95) * expected_peak_cm

    # Clamp to plausible range to avoid spikes
    height_cm = np.clip(height_cm, 0.0, MAX_HEIGHT_CM)

    # Pixel area in cm²
    pixel_area_cm2 = float(px_to_cm) ** 2

    # Volume over the mask
    volume_cm3 = float(height_cm[m].sum()) * pixel_area_cm2

    # Density: fallback 1.0 if not provided
    dens = DENSITY_MAP.get(str(food_label).lower(), 1.0) if DENSITY_MAP else 1.0
    weight_g = volume_cm3 * dens

    # Diagnostics
    area_px = int(np.count_nonzero(m))
    print(f"[DIAG] {food_label:>20s} | area_px={area_px:6d} "
          f"obj_med={obj_med:.5f} ring_med={ring_med:.5f} h95={h95:.5f} "
          f"px_to_cm={px_to_cm:.5f} vol_cm3≈{volume_cm3:.1f} "
          f"density={dens:.2f} -> weight_g≈{weight_g:.1f}")

    return round(float(weight_g), 1)

def _label_color(label: str):
    """Deterministic BGR color from label string."""
    h = int(hashlib.md5(label.encode('utf-8')).hexdigest(), 16)
    return (h % 256, (h // 256) % 256, (h // 65536) % 256)  # B, G, R

def save_annotated_image(image_path, res, detections, out_dir):
    """
    Draw YOLO boxes + labels (with confidence) and mask contours.
    Saves an annotated image next to out_dir with suffix _annot.jpg.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]

    vis = img.copy()
    overlay = img.copy()

    # --- draw YOLO boxes + labels ---
    names = det.model.names
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        cls  = res.boxes.cls.detach().cpu().numpy().astype(int)
        conf = res.boxes.conf.detach().cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            label = names[cls[i]]
            score = float(conf[i])
            color = _label_color(label)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            txt = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # --- draw mask contours (from your final 'detections' masks) ---
    for detitem in detections:
        label = detitem["food"]
        mask = detitem["mask"].astype(np.uint8)
        color = _label_color(label)
        # contour outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        # optional semi-transparent fill
        overlay[mask.astype(bool)] = (0.6*np.array(color) + 0.4*overlay[mask.astype(bool)]).astype(np.uint8)

    # blend filled overlay with base (only if you want the soft fill effect)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # --- save ---
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_annot.jpg")
    cv2.imwrite(out_path, vis)
    print(f"🖼️  Saved annotated image -> {out_path}")
    return out_path

def process_results(image_path, detections, out_dir=None):
    """
    detections: [{"food": <str>, "mask": np.ndarray(bool, HxW)}, ...]
    Writes: results_breakdown.csv, results_summary.csv
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Depth once per image
    depth_map = estimate_depth(img)

    # Pixel scale from plate (or fallback heuristic)
    px_to_cm = _estimate_plate_px_to_cm(img, default_plate_diam_cm=DEFAULT_PLATE_DIAM_CM)
    print(f"[DEBUG] px_to_cm ≈ {px_to_cm:.5f} cm/px  (image HxW={img.shape[0]}x{img.shape[1]})")

    portions = []
    total = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}

    for det in detections:
        food_label = det["food"]
        mask = det["mask"].astype(bool)

        weight = estimate_portion(mask, depth_map, food_label, img.shape, px_to_cm)
        nutrit = calculate_nutrition(food_label, weight)
        portions.append({"food": food_label, "weight_g": weight, **nutrit})

        for k in total:
            total[k] += float(nutrit[k])

    # --- NEW: per-image folder ---
    if out_dir is None:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(BASE, "out", stem)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(portions)
    df.to_csv(os.path.join(out_dir, "results_breakdown.csv"), index=False)
    pd.DataFrame([total]).to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)

    print("\nPer-Food Breakdown:\n", df)
    print("\nTotal Nutrition:\n", total)
    print("\n✅ Results saved as 'results_breakdown.csv' and 'results_summary.csv'")
    return df, total

# ------------- YOLOv8 detection -> masks (GrabCut) -------------
det = YOLO(WEIGHTS)

def box_to_mask_grabcut(img_bgr, xyxy, iters=5, min_area=100):
    """
    Return a boolean mask for the object inside the box.
    If GrabCut fails (too small / empty), fall back to the rectangle itself.
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((H, W), dtype=bool)

    rect = (x1, y1, x2 - x1, y2 - y1)

    # init mask with "probably background"
    mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        # light cleanup
        kernel = np.ones((3,3), np.uint8)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
        area = np.count_nonzero(fg)
        if area >= min_area:
            return fg
    except Exception:
        pass

    # Fallback: the rectangle itself (shrunk a bit so we’re inside the food)
    fallback = np.zeros((H, W), dtype=bool)
    pad = max(1, int(0.03 * max(rect[2], rect[3])))  # shrink 3%
    xx1, yy1 = x1 + pad, y1 + pad
    xx2, yy2 = x2 - pad, y2 - pad
    if xx2 > xx1 and yy2 > yy1:
        fallback[yy1:yy2, xx1:xx2] = True
    return fallback

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def cross_class_nms(xyxy, confs, iou_thr=0.60):
    """
    Greedy NMS across ALL classes. Returns indices of kept boxes.
    """
    order = np.argsort(-confs)  # high -> low
    keep = []
    for i in order:
        suppress = False
        for j in keep:
            if _iou_xyxy(xyxy[i], xyxy[j]) >= iou_thr:
                suppress = True
                break
        if not suppress:
            keep.append(i)
    return np.array(keep, dtype=int)

def run_one_image(image_path, conf=CONF, device=DEVICE):
    global food_nutrition, DENSITY_MAP

    # --- lazy-load lookup tables ---
    if food_nutrition is None:
        if not os.path.isfile(NUTRITION_CSV):
            raise FileNotFoundError(f"Nutrition CSV missing: {NUTRITION_CSV}")
        food_nutrition = load_nutrition_data(NUTRITION_CSV)

    if DENSITY_MAP is None:
        DENSITY_MAP = load_density_map(DENSITY_CSV)
        if not DENSITY_MAP:
            print("[WARN] DENSITY_MAP is empty; using density=1.0 for all labels.")

    # --- read image ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]

    # --- detection ---
    res = det.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        device=device,
        conf=conf,
        verbose=False
    )[0]

    names = det.model.names
    detections = []

    if res.boxes is None or len(res.boxes) == 0:
        print("No detections found.")
        save_annotated_image(image_path, res, detections, os.path.join(BASE, "out"))
        return process_results(image_path, detections)

    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
    confs = res.boxes.conf.detach().cpu().numpy()

    # -------- NEW: cross-class NMS (keep one label per object) --------
    keep_idx = cross_class_nms(xyxy, confs, iou_thr=0.60)
    xyxy, cls, confs = xyxy[keep_idx], cls[keep_idx], confs[keep_idx]
    # ------------------------------------------------------------------

    # prefer segmentation masks if available
    has_seg = getattr(res, "masks", None) is not None and res.masks is not None
    if has_seg:
        mdata = res.masks.data.detach().cpu().numpy()
        mdata = mdata[keep_idx]  # keep same indices as NMS

        def mask_from_det(i):
            m = (mdata[i] > 0.5).astype(np.uint8)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            return m.astype(bool)
    else:
        def mask_from_det(i):
            return box_to_mask_grabcut(img, xyxy[i])

    # --- build all masks first (after NMS) ---
    built = []
    MIN_AREA = 100
    for i in range(len(cls)):
        m = mask_from_det(i)
        area = int(np.count_nonzero(m))
        if area >= MIN_AREA:
            built.append({
                "i": i,
                "label": names[cls[i]],
                "conf": float(confs[i]),
                "mask": m,
                "area": area,
                "box": xyxy[i].astype(int),
            })

    if not built:
        print("No usable masks.")
        save_annotated_image(image_path, res, detections, os.path.join(BASE, "out"))
        return process_results(image_path, detections)

    # --- assign pixels big -> small (base foods first) ---
    built.sort(key=lambda d: d["area"], reverse=True)
    used = np.zeros((H, W), dtype=bool)
    for d in built:
        m0 = d["mask"]
        m = m0 & (~used)
        kept = int(np.count_nonzero(m))
        if kept < max(MIN_AREA, int(0.25 * d["area"])):
            m = m0
            kept = d["area"]
        if kept >= MIN_AREA:
            detections.append({"food": d["label"], "mask": m, "conf": d["conf"]})
            used |= m

    # --- per-image output folder ---
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(BASE, "out", stem)
    os.makedirs(out_dir, exist_ok=True)

    # save annotated image into this folder
    save_annotated_image(
        image_path=image_path,
        res=res,
        detections=detections,
        out_dir=out_dir
    )

    # write CSVs into the same folder
    df, total = process_results(image_path, detections, out_dir=out_dir)
    return df, total


# ------------- Run -------------
if os.path.isfile(TEST_IMAGE):
    df, total = run_one_image(TEST_IMAGE)
    print("\nPer-Food breakdown:\n", df)
    print("\nTotals:\n", total)
