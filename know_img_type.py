from PIL import Image
import numpy as np

img = Image.open("/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages/0_60_30_0_01609.jpg")
print(np.array(img).shape)
print(np.array(img).dtype)