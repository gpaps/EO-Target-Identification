
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Setup
output_dir = "../Optical/Optical_EDA_outputs/Optical_EDA_outputs_discrepancy/norm_bbox_category_views/"
os.makedirs(output_dir, exist_ok=True)

train_json = "Optical/json/VHRShips_Imagenet_Consolidated.json"
test_json = "Skysatt_bench_test.json"

with open(train_json) as f:
    train_data = json.load(f)
with open(test_json) as f:
    test_data = json.load(f)

categories = {cat["id"]: cat["name"] for cat in train_data["categories"]}

def extract_norm_bbox_areas(coco):
    image_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    per_cat_norm = defaultdict(list)
    for ann in coco["annotations"]:
        bbox = ann["bbox"]
        area = bbox[2] * bbox[3]
        w, h = image_dims[ann["image_id"]]
        image_area = w * h
        norm = area / image_area if image_area > 0 else 0
        per_cat_norm[ann["category_id"]].append(norm)
    return per_cat_norm

train_norm = extract_norm_bbox_areas(train_data)
test_norm = extract_norm_bbox_areas(test_data)

# Plot
for cat_id, cat_name in categories.items():
    t_vals = train_norm.get(cat_id, [])
    v_vals = test_norm.get(cat_id, [])
    if not t_vals and not v_vals:
        continue
    plt.figure(figsize=(12, 5))
    if t_vals:
        sns.histplot(t_vals, bins=150, kde=True, stat="density", color="blue", label="Train", alpha=0.6)
    if v_vals:
        sns.histplot(v_vals, bins=150, kde=True, stat="density", color="green", label="Test", alpha=0.6)
    plt.title(f"Normalized BBox Area: {cat_name}")
    plt.xlabel("BBox Area / Image Area")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"norm_bbox_area_{cat_name.replace(' ', '_')}.png"))
    plt.close()
