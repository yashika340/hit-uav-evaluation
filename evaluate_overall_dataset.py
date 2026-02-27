import os
import glob
import torch
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict

# -------------------------
# CONFIG
# -------------------------

IMAGE_DIR = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages"
ANNOTATION_DIR = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/Annotations"
CONFIG_PATH = "/home/csio/Desktop/hit-uav2/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

CHECKPOINT_PATH = "/home/csio/Desktop/hit-uav2/GroundingDINO/weights/groundingdino_swint_ogc.pth"

TEXT_PROMPT = "person . car . bicycle . other vehicle"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

CLASSES = ["Person", "Car", "Bicycle", "OtherVehicle"]

# -------------------------
# IoU
# -------------------------

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1Area + box2Area - interArea

    if union == 0:
        return 0
    return interArea / union

# -------------------------
# Load GT from XML
# -------------------------

def load_ground_truth(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    gt = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name == "DontCare":
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        gt.append({
            "class": name,
            "bbox": [xmin, ymin, xmax, ymax],
            "matched": False
        })

    return gt

# -------------------------
# Compute AP
# -------------------------

def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

# -------------------------
# MAIN
# -------------------------

model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
model = model.to("cpu")
model.eval()

all_predictions = defaultdict(list)
all_ground_truths = defaultdict(int)

image_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

for img_path in tqdm(image_files):

    xml_path = os.path.join(
        ANNOTATION_DIR,
        os.path.basename(img_path).replace(".jpg", ".xml")
    )

    gt_objects = load_ground_truth(xml_path)

    for obj in gt_objects:
        all_ground_truths[obj["class"]] += 1

    image_source, image = load_image(img_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device="cpu" 
    )

    H, W = image_source.shape[:2]

    for box, logit, phrase in zip(boxes, logits, phrases):
        phrase = phrase.lower()

        if "person" in phrase:
            cls = "Person"
        elif "car" in phrase:
            cls = "Car"
        elif "bicycle" in phrase:
            cls = "Bicycle"
        elif "vehicle" in phrase:
            cls = "OtherVehicle"
        else:
            continue

        box = box * torch.tensor([W, H, W, H])
        box = box.cpu().numpy()
        x_center, y_center, w, h = box

        xmin = x_center - w / 2
        ymin = y_center - h / 2
        xmax = x_center + w / 2
        ymax = y_center + h / 2

        all_predictions[cls].append({
            "bbox": [xmin, ymin, xmax, ymax],
            "score": logit.item(),
            "image": img_path
        })

# -------------------------
# Evaluation
# -------------------------

print("\n===== Overall Dataset Results =====\n")

aps = []

for cls in CLASSES:

    predictions = sorted(all_predictions[cls], key=lambda x: x["score"], reverse=True)
    total_gt = all_ground_truths[cls]

    tp = []
    fp = []

    matched = defaultdict(set)

    for pred in predictions:

        xml_path = os.path.join(
            ANNOTATION_DIR,
            os.path.basename(pred["image"]).replace(".jpg", ".xml")
        )

        gt_objects = load_ground_truth(xml_path)
        gt_cls = [g for g in gt_objects if g["class"] == cls]

        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_cls):
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= IOU_THRESHOLD and best_gt_idx not in matched[pred["image"]]:
            tp.append(1)
            fp.append(0)
            matched[pred["image"]].add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / total_gt if total_gt > 0 else np.zeros_like(tp)
    precisions = tp / (tp + fp + 1e-6)

    ap = compute_ap(recalls, precisions)
    aps.append(ap)

    final_precision = precisions[-1] if len(precisions) > 0 else 0
    final_recall = recalls[-1] if len(recalls) > 0 else 0

    print(f"Class: {cls}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall:    {final_recall:.4f}")
    print(f"AP:        {ap:.4f}")
    print()

mAP = np.mean(aps)
print("===================================")
print(f"Overall mAP: {mAP:.4f}")