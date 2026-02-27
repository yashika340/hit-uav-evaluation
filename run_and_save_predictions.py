import torch
import json
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict

# -----------------------------
# PATHS (EDIT IF NEEDED)
# -----------------------------
config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

image_path = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages/0_60_30_0_01609.jpg"
output_json = "predictions_single_image.json"

text_prompt = "person . car . bicycle . other vehicle"

device = "cpu"   # change to "cuda" if GPU available


# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading model...")
model = load_model(config_path, checkpoint_path)
model = model.to(device)


# -----------------------------
# LOAD IMAGE
# -----------------------------
print("Loading image...")
image_np, image_tensor = load_image(image_path)

# image_np is numpy array
H, W = image_np.shape[:2]


# -----------------------------
# RUN PREDICTION
# -----------------------------
print("Running prediction...")
boxes, logits, phrases = predict(
    model=model,
    image=image_tensor,
    caption=text_prompt,
    box_threshold=0.3,
    text_threshold=0.25,
    device=device
)

predictions = []

for box, phrase, score in zip(boxes, phrases, logits):

    # GroundingDINO gives boxes as [cx, cy, w, h] normalized (0-1)
    box = box * torch.tensor([W, H, W, H])

    cx, cy, bw, bh = box.tolist()

    # Convert center format to xmin, ymin, xmax, ymax
    xmin = cx - bw / 2
    ymin = cy - bh / 2
    xmax = cx + bw / 2
    ymax = cy + bh / 2

    predictions.append({
        "label": phrase.lower().strip(),
        "score": float(score),
        "box": [xmin, ymin, xmax, ymax]
    })


# -----------------------------
# SAVE JSON
# -----------------------------
with open(output_json, "w") as f:
    json.dump(predictions, f, indent=4)

print("\nSaved predictions to:", output_json)
print("Total predictions:", len(predictions))