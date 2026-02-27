import os
import torch
import xml.etree.ElementTree as ET
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict


# ========================================
# CONFIG
# ========================================

config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

images_dir = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages"
annotations_dir = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/Annotations"

text_prompt = "person . car . bicycle . other vehicle"

device = "cpu"

box_threshold = 0.3
text_threshold = 0.25
iou_thresh = 0.5


# ========================================
# IOU FUNCTION
# ========================================

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


# ========================================
# LOAD MODEL
# ========================================

print("Loading GroundingDINO...")
model = load_model(config_path, checkpoint_path)
model = model.to(device)


# ========================================
# DATASET LOOP
# ========================================

total_TP = 0
total_FP = 0
total_FN = 0

image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

print("Total images:", len(image_files))


for idx, image_file in enumerate(image_files):

    image_path = os.path.join(images_dir, image_file)
    xml_path = os.path.join(annotations_dir, image_file.replace(".jpg", ".xml"))

    if not os.path.exists(xml_path):
        continue

    # ------------------
    # LOAD IMAGE
    # ------------------
    image_np, image_tensor = load_image(image_path)
    H, W = image_np.shape[:2]

    # ------------------
    # PREDICT
    # ------------------
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    pred_boxes = []
    pred_labels = []

    for box, phrase in zip(boxes, phrases):
        box = box * torch.tensor([W, H, W, H])
        cx, cy, bw, bh = box.tolist()

        xmin = cx - bw/2
        ymin = cy - bh/2
        xmax = cx + bw/2
        ymax = cy + bh/2

        pred_boxes.append([xmin, ymin, xmax, ymax])
        pred_labels.append(phrase.lower().strip())

    # ------------------
    # LOAD GT
    # ------------------
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_boxes = []
    gt_labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text

        if name == "DontCare":
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        gt_boxes.append([xmin, ymin, xmax, ymax])
        gt_labels.append(name.lower())

    # ------------------
    # MATCHING
    # ------------------
    matched = set()
    TP = 0
    FP = 0

    for pbox, plabel in zip(pred_boxes, pred_labels):

        best_iou = 0
        best_gt = -1

        for j, (gtbox, gtlabel) in enumerate(zip(gt_boxes, gt_labels)):

            if j in matched:
                continue

            if plabel != gtlabel:
                continue

            iou = compute_iou(pbox, gtbox)

            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_thresh:
            TP += 1
            matched.add(best_gt)
        else:
            FP += 1

    FN = len(gt_boxes) - len(matched)

    total_TP += TP
    total_FP += FP
    total_FN += FN

    if idx % 50 == 0:
        print(f"Processed {idx}/{len(image_files)}")


# ========================================
# FINAL METRICS
# ========================================

precision = total_TP / (total_TP + total_FP) if (total_TP+total_FP) > 0 else 0
recall = total_TP / (total_TP + total_FN) if (total_TP+total_FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0

print("\n====== FINAL RESULTS ======")
print("Total TP:", total_TP)
print("Total FP:", total_FP)
print("Total FN:", total_FN)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)