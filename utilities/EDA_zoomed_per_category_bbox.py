
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict

# Setup
output_dir = "../Optical/Optical_EDA_outputs/Optical_EDA_outputs_discrepancy/zoomed_category_views/"
os.makedirs(output_dir, exist_ok=True)

train_json_path = "../Optical/json/VHRShips_Imagenet_Consolidated.json"
test_json_path = "../Optical/Skysatt_bench_test.json"

# Load
with open(train_json_path) as f:
    train_data = json.load(f)
with open(test_json_path) as f:
    test_data = json.load(f)

# Category names
categories = {cat["id"]: cat["name"] for cat in train_data["categories"]}

def extract_bbox_areas(data):
    per_category = defaultdict(list)
    image_dims = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
    for ann in data["annotations"]:
        bbox = ann["bbox"]
        w, h = bbox[2], bbox[3]
        area = w * h
        per_category[ann["category_id"]].append(area)
    return per_category

train_bbox = extract_bbox_areas(train_data)
test_bbox = extract_bbox_areas(test_data)

# Plot function with zoom
def plot_zoomed_category(train_vals, test_vals, cat_name, max_area=20000):
    train_zoom = [v for v in train_vals if v <= max_area]
    test_zoom = [v for v in test_vals if v <= max_area]
    if not train_zoom and not test_zoom:
        return
    plt.figure(figsize=(12, 5))
    if train_zoom:
        sns.histplot(train_zoom, bins=60, kde=True, color="blue", label="Train", stat="density", alpha=0.6)
    if test_zoom:
        sns.histplot(test_zoom, bins=60, kde=True, color="green", label="Test", stat="density", alpha=0.6)
    plt.title(f"Zoomed BBox Area (<{max_area} px²): {cat_name}")
    plt.xlabel("BBox Area")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"zoomed_bbox_area_{cat_name.replace(' ', '_')}.png"))
    plt.close()

# Create all category plots
for cat_id, cat_name in categories.items():
    train_vals = train_bbox.get(cat_id, [])
    test_vals = test_bbox.get(cat_id, [])
    plot_zoomed_category(train_vals, test_vals, cat_name, max_area=20000)
    plot_zoomed_category(train_vals, test_vals, cat_name, max_area=5000)
