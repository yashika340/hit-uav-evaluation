import cv2
import os

# -------- INPUT --------
input_path = "/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages/0_60_30_0_01609.jpg"
output_path = "/home/csio/Desktop/hit-uav2/test_processed.jpg"

# -------- READ IMAGE (grayscale) --------
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found!")

print("Original shape:", img.shape)
print("Original dtype:", img.dtype)

# -------- APPLY CLAHE (best for thermal) --------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img)

# -------- CONVERT TO RGB --------
img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

# -------- SAVE --------
cv2.imwrite(output_path, img_rgb)

print("Processed image saved at:", output_path)
print("Processed shape:", img_rgb.shape)