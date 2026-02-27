import xml.etree.ElementTree as ET
import numpy as np

# -----------------------------
# IoU Calculation
# -----------------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


# -----------------------------
# Read Ground Truth from XML
# -----------------------------
def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name != "person":
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])

    return boxes


# -----------------------------
# Match Predictions with GT
# -----------------------------
def evaluate(pred_boxes, gt_boxes, iou_threshold=0.5):

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        found_match = False
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            iou = compute_iou(pred, gt)

            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                found_match = True
                break

        if not found_match:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return tp, fp, fn, precision, recall


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # Example paths (CHANGE THESE)
    xml_path = "path_to_annotation.xml"

    # Manually paste predicted boxes here
    # Format: [[xmin, ymin, xmax, ymax], ...]
    pred_boxes = [
        # Example:
        # [50, 60, 120, 180],
    ]

    gt_boxes = read_xml(xml_path)

    tp, fp, fn, precision, recall = evaluate(pred_boxes, gt_boxes)

    print("Ground Truth Boxes:", len(gt_boxes))
    print("Predicted Boxes:", len(pred_boxes))
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("Precision:", precision)
    print("Recall:", recall)