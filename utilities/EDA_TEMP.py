import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import cv2
from tqdm import tqdm
import os
import sys

# Output setup
output_dir = "../Optical/VHRships_Imagenet/Optical_EDA_outputs/"
# output_dir = "SAR/SAR_EDA_outputs"
os.makedirs(output_dir, exist_ok=True)
sys.stdout = open(os.path.join(output_dir, "OptShip_terminal_output.txt"), "w")
# sys.stdout = open(os.path.join(output_dir, "SARShips_terminal_output.txt"), "w")

# Load dataset JSON
# json_path = "Skysatt_bench_test.json"
json_path = "../Optical/VHRships_Imagenet/json/coco_train.json"
# json_path = "Optical/docker2go/json/coco_train.json"
# json_path = "SAR/docker2go/json/coco_train.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract image metadata
image_info = pd.DataFrame(data["images"])
annotations = pd.DataFrame(data["annotations"])
categories = {cat["id"]: cat["name"] for cat in data["categories"]}

# 1. Width/Height Distribution
plt.figure(figsize=(10, 5))
sns.histplot(image_info["width"], bins=30, kde=True, color="blue", label="Width", alpha=0.6)
sns.histplot(image_info["height"], bins=30, kde=True, color="red", label="Height", alpha=0.6)
plt.legend()
plt.xlabel("Image Dimensions")
plt.ylabel("Frequency")
plt.title("Distribution of Image Widths and Heights")
plt.savefig(os.path.join(output_dir, "width_height_distribution.png"))
plt.close()

# 2. Aspect Ratio Distribution
image_info["aspect_ratio"] = image_info["width"] / image_info["height"]
plt.figure(figsize=(10, 5))
sns.histplot(image_info["aspect_ratio"], bins=30, kde=True, color="green")
plt.xlabel("Aspect Ratio (Width / Height)")
plt.ylabel("Frequency")
plt.title("Aspect Ratio Distribution")
plt.savefig(os.path.join(output_dir, "aspect_ratio_distribution.png"))
plt.close()

# 3. Resolution Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=image_info[["width", "height"]], palette="coolwarm")
plt.title("Boxplot of Image Resolutions")
plt.ylabel("Pixels")
plt.xticks([0, 1], ['Width', 'Height'])
plt.savefig(os.path.join(output_dir, "resolution_boxplot.png"))
plt.close()

# 4. Resolution Outliers
width_outliers = image_info["width"].quantile([0.05, 0.95])
height_outliers = image_info["height"].quantile([0.05, 0.95])
print(f"Width outlier threshold (5-95%): {width_outliers.tolist()}")
print(f"Height outlier threshold (5-95%): {height_outliers.tolist()}")

# 5. Class Distribution analysis
category_counts = annotations["category_id"].map(categories).value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Category")
plt.ylabel("Instance Count")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# 6. Bounding Box Area
annotations["bbox_area"] = annotations["bbox"].apply(lambda x: x[2] * x[3])
plt.figure(figsize=(10, 5))
sns.histplot(annotations["bbox_area"], bins=300, kde=True, color="purple")
plt.xlabel("Bounding Box Area")
plt.ylabel("Frequency")
plt.title("Bounding Box Area Distribution")
plt.savefig(os.path.join(output_dir, "bbox_area_distribution.png"))
plt.close()

# 7. Brightness Analysis
brightness_values = []
image_folder = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/VHRShips_ShipRSImageNEt/"
# image_folder = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/Optical/"
# image_folder = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/00_Benchmark_validation/Boat_images_labels_1km_cropped/images/"
for img in tqdm(image_info["file_name"], desc="Calculating brightness"):
    img_path = os.path.join(image_folder, img)
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))

if brightness_values:
    plt.figure(figsize=(10, 5))
    sns.histplot(brightness_values, bins=30, kde=True, color="orange")
    plt.xlabel("Mean Brightness")
    plt.ylabel("Frequency")
    plt.title("Image Brightness Distribution")
    plt.savefig(os.path.join(output_dir, "brightness_distribution.png"))
    plt.close()
else:
    print(" No valid images found for brightness analysis.")

# 8. Duplicates
file_counts = Counter(image_info["file_name"])
duplicates = [fname for fname, count in file_counts.items() if count > 1]
print(f"Total Duplicate Images: {len(duplicates)}")
if duplicates:
    print("Example duplicates:", duplicates[:5])

# 9. Summary Statistics
print("\nImage Info Summary:")
print(image_info.describe())

# Close log
sys.stdout.close()
