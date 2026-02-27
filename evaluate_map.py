import os
import torch
import xml.etree.ElementTree as ET
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict
from collections import defaultdict


# ===============================
# CONFIG
# ===============================

config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

images_dir = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages"
annotations_dir = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/Annotations"

classes = ["person", "car", "bicycle", "othervehicle"]

text_prompt = "person . car . bicycle . other vehicle"

device = "cpu"

box_threshold = 0.3
text_threshold = 0.25
iou_thresh = 0.5


# ===============================
# IOU FUNCTION
# ===============================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter/union if union>0 else 0


# ===============================
# LOAD MODEL
# ===============================

print("Loading model...")
model = load_model(config_path, checkpoint_path)
model = model.to(device)


# ===============================
# STORAGE
# ===============================

all_gt = defaultdict(list)
all_pred = defaultdict(list)
gt_count = defaultdict(int)


# ===============================
# DATASET LOOP
# ===============================

image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

for idx, image_file in enumerate(image_files):

    image_path = os.path.join(images_dir, image_file)
    xml_path = os.path.join(annotations_dir, image_file.replace(".jpg", ".xml"))

    if not os.path.exists(xml_path):
        continue

    # Load image
    image_np, image_tensor = load_image(image_path)
    H, W = image_np.shape[:2]

    # Predict
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    # Convert predictions
    for box, score, label in zip(boxes, logits, phrases):

        box = box * torch.tensor([W, H, W, H])
        cx, cy, bw, bh = box.tolist()

        xmin = cx - bw/2
        ymin = cy - bh/2
        xmax = cx + bw/2
        ymax = cy + bh/2

        label = label.lower().replace(" ", "")

        if label in classes:
            all_pred[label].append({
                "image": image_file,
                "box": [xmin, ymin, xmax, ymax],
                "score": float(score)
            })

    # Load GT
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.find("name").text.lower()

        if name == "dontcare":
            continue

        name = name.replace(" ", "")

        if name not in classes:
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        all_gt[name].append({
            "image": image_file,
            "box": [xmin, ymin, xmax, ymax],
            "matched": False
        })

        gt_count[name] += 1

    if idx % 50 == 0:
        print(f"Processed {idx}/{len(image_files)}")


# ===============================
# COMPUTE AP PER CLASS
# ===============================

def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(len(precisions)-1, 0, -1):
        precisions[i-1] = np.maximum(precisions[i-1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices+1] - recalls[indices]) * precisions[indices+1])
    return ap


aps = []

for cls in classes:

    predictions = sorted(all_pred[cls], key=lambda x: x["score"], reverse=True)

    TP = np.zeros(len(predictions))
    FP = np.zeros(len(predictions))

    gt_cls = all_gt[cls]

    for i, pred in enumerate(predictions):

        best_iou = 0
        best_gt = None

        for gt in gt_cls:
            if gt["image"] != pred["image"]:
                continue

            iou = compute_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_iou >= iou_thresh and best_gt and not best_gt["matched"]:
            TP[i] = 1
            best_gt["matched"] = True
        else:
            FP[i] = 1

    TP_cum = np.cumsum(TP)
    FP_cum = np.cumsum(FP)

    recalls = TP_cum / gt_count[cls] if gt_count[cls] > 0 else np.zeros(len(TP))
    precisions = TP_cum / (TP_cum + FP_cum + 1e-6)

    ap = compute_ap(recalls, precisions)
    aps.append(ap)

    print(f"AP for {cls}: {ap:.4f}")

mAP = np.mean(aps)
print("\n====== FINAL mAP@0.5 ======")
print("mAP:", mAP)