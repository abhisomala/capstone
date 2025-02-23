import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bar

# Define dataset path
dataset_path = "C:/Users/somal/OneDrive/Desktop/Capstone_Model/algae_bloom_dataset"

# Define color thresholds for algae bloom detection (Adjust these values)
lower_green = np.array([30, 40, 40])  # Lower bound of algae-like color
upper_green = np.array([90, 255, 255])  # Upper bound of algae-like color
threshold_ratio = 0.05  # Adjust: % of pixels needed to classify as bloom

# Function to detect algae bloom
def detect_algae_bloom(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None  # Skip if the image can't be read

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV
    mask = cv2.inRange(image, lower_green, upper_green)  # Filter green/yellow

    # Calculate % of bloom pixels
    algae_ratio = np.sum(mask > 0) / mask.size  
    return 1 if algae_ratio > threshold_ratio else 0  # Assign label

# Process all images and store labels
labels = []
for folder in ["train", "test", "val"]: 
    folder_path = os.path.join(dataset_path, folder)
    
    if not os.path.exists(folder_path):
        continue  # Skip missing folders
    
    print(f"Processing {folder} images...")
    
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            label = detect_algae_bloom(img_path)
            
            if label is not None:
                labels.append([img_path, label])

# Save labels to CSV
df = pd.DataFrame(labels, columns=["image_path", "label"])
csv_path = os.path.join("C:/Users/somal/Desktop/capstone", "labels.csv")
df.to_csv(csv_path, index=False)

print(f"Labeling complete! Saved labels to {csv_path}")
