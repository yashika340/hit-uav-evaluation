import os
import pandas as pd

image_folder='/home/csio/Desktop/hit-uav2/HIT-UAV-Infrared-Thermal-Dataset-main/normal_xml/JPEGImages'
data=[]
for file in os.listdir(image_folder):
    if(file.endswith('.jpg')):
        parts=file.split('_')
        time_value = int(parts[0])
        time_label = "day" if time_value == 0 else "night"

        data.append({
            "image_path": file,
            "time": time_value,
            "time_label": time_label
        })

df = pd.DataFrame(data)
df.to_csv('metadata.csv', index=False)
print("Metadata created successfully!")