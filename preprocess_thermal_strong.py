import cv2
import numpy as np

input_path = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages/0_60_30_0_01609.jpg"
output_path = "/home/csio/Desktop/hit-uav2/test_processed_strong.jpg"

# Read grayscale
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

print("Original:", img.shape, img.dtype)

# ---------- Step 1: Normalize ----------
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# ---------- Step 2: Strong CLAHE ----------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_norm)

# ---------- Step 3: Gamma Correction ----------
gamma = 1.5   # try 1.2–2.0
img_gamma = np.power(img_clahe / 255.0, gamma)
img_gamma = (img_gamma * 255).astype(np.uint8)

# ---------- Step 4: Upscale (important for small persons) ----------
img_up = cv2.resize(img_gamma, (1280, 1024), interpolation=cv2.INTER_CUBIC)

# ---------- Step 5: Convert to RGB ----------
img_rgb = cv2.cvtColor(img_up, cv2.COLOR_GRAY2RGB)

cv2.imwrite(output_path, img_rgb)

print("Saved:", output_path)
print("Processed:", img_rgb.shape)