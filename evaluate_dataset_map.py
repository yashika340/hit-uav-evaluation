import os
import torch
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from groundingdino.util.inference import load_model, load_image, predict


# ==========================================
# CONFIG
# ==========================================

config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

images_dir = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages"
annotations_dir = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/Annotations"

classes = ["person", "car", "bicycle", "othervehicle"]

text_prompt = "person . car . bicycle . other vehicle"

device = "cpu"
box_threshold = 0.3
text_threshold = 0.25
iou_threshold = 0.5


# ==========================================
# IOU
# ==========================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter/union if union > 0 else 0


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
model = load_model(config_path, checkpoint_path)
model = model.to(device)


# ==========================================
# STORAGE STRUCTURES
# ==========================================

all_predictions = defaultdict(list)
all_ground_truths = defaultdict(list)

image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]


# ==========================================
# DATASET LOOP
# ==========================================

for image_file in image_files:

    image_id = image_file.replace(".jpg", "")
    image_path = os.path.join(images_dir, image_file)
    xml_path = os.path.join(annotations_dir, image_id + ".xml")

    if not os.path.exists(xml_path):
        continue

    # ---- IMAGE ----
    image_np, image_tensor = load_image(image_path)
    H, W = image_np.shape[:2]

    # ---- PREDICT ----
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    for box, score, label in zip(boxes, logits, phrases):

        box = box * torch.tensor([W, H, W, H])
        cx, cy, bw, bh = box.tolist()

        xmin = cx - bw/2
        ymin = cy - bh/2
        xmax = cx + bw/2
        ymax = cy + bh/2

        label = label.lower().strip().replace(" ", "")

        if label in classes:
            all_predictions[label].append({
                "image_id": image_id,
                "score": float(score),
                "box": [xmin, ymin, xmax, ymax]
            })

    # ---- GROUND TRUTH ----
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.find("name").text.lower()

        if name == "dontcare":
            continue

        name = name.replace(" ", "")

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        all_ground_truths[name].append({
            "image_id": image_id,
            "box": [xmin, ymin, xmax, ymax],
            "used": False
        })


# ==========================================
# AP CALCULATION
# ==========================================

def compute_ap(predictions, ground_truths, iou_thresh=0.5):

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    TP = np.zeros(len(predictions))
    FP = np.zeros(len(predictions))

    total_gt = len(ground_truths)

    for i, pred in enumerate(predictions):

        gt_for_image = [g for g in ground_truths if g["image_id"] == pred["image_id"]]

        best_iou = 0
        best_gt = None

        for gt in gt_for_image:
            iou = compute_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_iou >= iou_thresh and best_gt and not best_gt["used"]:
            TP[i] = 1
            best_gt["used"] = True
        else:
            FP[i] = 1

    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)

    recalls = cum_TP / total_gt if total_gt > 0 else np.zeros(len(TP))
    precisions = cum_TP / (cum_TP + cum_FP + 1e-6)

    # 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return ap


# ==========================================
# COMPUTE AP PER CLASS
# ==========================================

aps = []

print("\n===== PER CLASS AP =====")

for cls in classes:

    if cls not in all_ground_truths:
        continue

    ap = compute_ap(all_predictions[cls], all_ground_truths[cls], iou_threshold)
    aps.append(ap)

    print(f"{cls}: AP@0.5 = {ap:.4f}")

mAP = np.mean(aps) if len(aps) > 0 else 0

print("\n===== FINAL RESULTS =====")
print("mAP@0.5:", mAP)