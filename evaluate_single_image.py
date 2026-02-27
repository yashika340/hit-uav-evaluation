import json
import xml.etree.ElementTree as ET
import numpy as np

# -----------------------------
# IOU
# -----------------------------
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


# -----------------------------
# LOAD GT
# -----------------------------
xml_path = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/Annotations/0_60_30_0_01609.xml"

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


# -----------------------------
# LOAD PREDICTIONS
# -----------------------------
with open("predictions_single_image.json") as f:
    preds = json.load(f)

pred_boxes = []
pred_labels = []

for p in preds:
    pred_boxes.append(p["box"])
    pred_labels.append(p["label"].lower())


# -----------------------------
# MATCHING
# -----------------------------
iou_thresh = 0.5

matched = set()
TP = 0
FP = 0

for i, (pbox, plabel) in enumerate(zip(pred_boxes, pred_labels)):

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

precision = TP / (TP+FP) if TP+FP>0 else 0
recall = TP / (TP+FN) if TP+FN>0 else 0

print("\nGT objects:", len(gt_boxes))
print("Predicted:", len(pred_boxes))
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("Precision:", precision)
print("Recall:", recall)