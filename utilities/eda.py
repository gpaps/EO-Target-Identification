# Consolidated EDA for Benchmark Dataset (Train Only)
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from collections import defaultdict, Counter
from tqdm import tqdm

# === CONFIG ===
# json_path = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/Final_Fully_Cleaned_Ships[VHRShips_ShipsImageNet].json"
# image_folder = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/VHRShips_ShipRSImageNEt/"
# output_dir = "../final_EDA_Benchmark_"
# output_dir = "../final_EDA_Benchmark_"
#Optical
# image_folder = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/JPG/"
# json_path = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/opt_ships.json"
# output_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/"
# SAR
image_folder = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships_dataset/"
json_path = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships.json"
output_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/eda_"

os.makedirs(output_dir, exist_ok=True)

# === Load COCO JSON ===
with open(json_path, 'r') as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = {cat["id"]: cat["name"] for cat in data["categories"]}

image_df = pd.DataFrame(images)
ann_df = pd.DataFrame(annotations)

# === 1. Image Size & Aspect Ratio ===
plt.figure(figsize=(10, 5))
sns.histplot(image_df["width"], color="blue", label="Width", kde=True)
sns.histplot(image_df["height"], color="red", label="Height", kde=True)
plt.title("Image Width/Height Distribution")
plt.legend()
plt.savefig(f"{output_dir}/image_dimensions.png")
plt.close()

image_df = image_df[image_df["height"] > 0]
image_df["aspect_ratio"] = image_df["width"] / image_df["height"]
plt.figure(figsize=(10, 5))
sns.histplot(image_df["aspect_ratio"], bins=30, kde=True)
plt.title("Aspect Ratio Distribution")
plt.savefig(f"{output_dir}/aspect_ratio.png")
plt.close()

# === 2. BBox Area ===
ann_df["bbox_area"] = ann_df["bbox"].apply(lambda b: b[2] * b[3])
sns.histplot(ann_df["bbox_area"], bins=150, kde=True, color="purple")
plt.title("BBox Area Distribution")
plt.savefig(f"{output_dir}/bbox_area.png")
plt.close()

# === 3. Normalized BBox Area ===
img_dims = {img["id"]: (img["width"], img["height"]) for img in images}
norm_areas = []
for ann in annotations:
    w, h = img_dims[ann["image_id"]]
    img_area = w * h
    bbox = ann["bbox"]
    area = bbox[2] * bbox[3]
    norm_areas.append(area / img_area if img_area > 0 else 0)

sns.histplot(norm_areas, bins=100, kde=True, color="orange")
plt.title("Normalized BBox Area Distribution")
plt.savefig(f"{output_dir}/norm_bbox_area.png")
plt.close()

# === 4. Class Distribution ===
ann_df["class_name"] = ann_df["category_id"].map(categories)
cls_counts = ann_df["class_name"].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=cls_counts.index, y=cls_counts.values)
plt.xticks(rotation=45)
plt.title("Class Distribution (Instances)")
plt.tight_layout()
plt.savefig(f"{output_dir}/class_distribution.png")
plt.close()

# === 5. Brightness ===
brightness_vals = []
for img in tqdm(images, desc="Brightness Analysis"):
    path = os.path.join(image_folder, img["file_name"])
    if os.path.exists(path):
        raw = cv2.imread(path)
        if raw is not None:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            brightness_vals.append(np.mean(gray))

if brightness_vals:
    sns.histplot(brightness_vals, bins=30, kde=True, color="gray")
    plt.title("Image Brightness Distribution")
    plt.savefig(f"{output_dir}/brightness_distribution.png")
    plt.close()

# === 6. Resize Suggestion ===
object_sizes = np.sqrt(ann_df["bbox_area"].values)
avg_object_side = np.mean(object_sizes)
base_size = int(np.clip(avg_object_side * 20, 512, 1280))

# ---------------------------------------------------------
# Tiny / Small / Medium / Large object size analysis (ADD)
# ---------------------------------------------------------
print("\n=== OBJECT SIZE ANALYSIS (in pixels) ===")

# Basic stats
print("Side-length stats (px):")
print("  min   :", object_sizes.min())
print("  p5    :", np.percentile(object_sizes, 5))
print("  p25   :", np.percentile(object_sizes, 25))
print("  median:", np.percentile(object_sizes, 50))
print("  p75   :", np.percentile(object_sizes, 75))
print("  p95   :", np.percentile(object_sizes, 95))
print("  max   :", object_sizes.max())

# Thresholds for bins — adjust to your EO use case
tiny_thr   = 12      # < 12 px (very tiny)
small_thr  = 32      # 12–32 px
medium_thr = 96      # 32–96 px (typical ships)
# large ≥ 96 px

# Boolean masks
tiny_mask   = object_sizes < tiny_thr
small_mask  = (object_sizes >= tiny_thr) & (object_sizes < small_thr)
medium_mask = (object_sizes >= small_thr) & (object_sizes < medium_thr)
large_mask  = object_sizes >= medium_thr

total = len(object_sizes)

print("\nObject size bins:")
print(f"  tiny (<{tiny_thr}px):          {tiny_mask.sum()}  ({tiny_mask.sum()/total:.2%})")
print(f"  small ({tiny_thr}–{small_thr}px):   {small_mask.sum()} ({small_mask.sum()/total:.2%})")
print(f"  medium ({small_thr}–{medium_thr}px): {medium_mask.sum()} ({medium_mask.sum()/total:.2%})")
print(f"  large (≥{medium_thr}px):        {large_mask.sum()} ({large_mask.sum()/total:.2%})")

# Optional: print the smallest 5 boxes for manual inspection
print("\n5 smallest objects (width, height):")
smallest_idx = np.argsort(object_sizes)[:5]
for idx in smallest_idx:
    print(" ", ann_df.iloc[idx]["bbox"])
# ---------------------------------------------------------

# Add percentile-based side suggestions
min_side = np.percentile(np.minimum(image_df['width'], image_df['height']), [10, 50, 90])
max_side = np.percentile(np.maximum(image_df['width'], image_df['height']), [10, 50, 90])

resize_suggestion = f"""
Suggested Resize Parameters:
- Based on object size:
  MIN_SIZE_TRAIN = range({int(base_size*0.8)}, {int(base_size)})
  MAX_SIZE_TRAIN = {int(base_size*1.5)}
  MIN_SIZE_TEST  = {int(base_size)}

- Based on resolution percentiles:
  MIN_SIZE_TRAIN = range({int(min_side[0])}, {int(min_side[2])})
  MAX_SIZE_TRAIN = {int(max_side[2])}
"""

with open(f"{output_dir}/resize_suggestion.txt", "w") as f:
    f.write(resize_suggestion)

# === 7. Print Summary ===
summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("Total Images: %d\n" % len(images))
    f.write("Total Annotations: %d\n" % len(annotations))
    f.write("Unique Classes: %d\n\n" % len(categories))
    f.write(resize_suggestion)
    f.write("\nAspect Ratio (mean): %.2f\n" % image_df["aspect_ratio"].mean())
    f.write("Normalized BBox Area (mean): %.5f\n" % np.mean(norm_areas))
    f.write("BBox Area (mean): %.1f\n" % ann_df["bbox_area"].mean())
    if brightness_vals:
        f.write("Brightness (mean): %.1f\n" % np.mean(brightness_vals))
