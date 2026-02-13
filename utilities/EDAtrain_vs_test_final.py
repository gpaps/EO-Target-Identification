import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2
from tqdm import tqdm
from collections import defaultdict

# Output directory
output_dir = "../Optical/Optical_EDA_outputs/Optical_EDA_outputs_discrepancy/"
os.makedirs(output_dir, exist_ok=True)

# Input files
train_json_path = "/Optical/json/VHRShips_Imagenet_Consolidated.json"
# train_json_path = "Optical/json/VHRShips_Imagenet_Consolidated.json"
test_json_path = "/Skysatt_bench_test.json"

# Image folders for brightness
train_image_folder = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/Optical/"
test_image_folder = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/00_Benchmark_validation/Boat_images_labels_1km_cropped/images/"

# Load data
with open(train_json_path) as f:
    train_data = json.load(f)
with open(test_json_path) as f:
    test_data = json.load(f)

# Category mapping
categories = {cat["id"]: cat["name"] for cat in train_data["categories"]}


def extract_stats(coco_dict, image_folder=None):
    img_info = {img["id"]: (img["width"], img["height"], img["file_name"]) for img in coco_dict["images"]}
    widths, heights, aspects, bbox_areas, norm_bbox_areas, brightness = [], [], [], [], [], []
    per_cat_bbox = defaultdict(list)

    for ann in coco_dict["annotations"]:
        w, h, fname = img_info[ann["image_id"]]
        bbox = ann["bbox"]
        area = bbox[2] * bbox[3]
        widths.append(w)
        heights.append(h)
        aspects.append(w / h)
        bbox_areas.append(area)
        norm_bbox_areas.append(area / (w * h))
        per_cat_bbox[ann["category_id"]].append(area)

    if image_folder:
        for _, (_, _, fname) in img_info.items():
            path = os.path.join(image_folder, fname)
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    brightness.append(np.mean(gray))

    return {
        "widths": widths,
        "heights": heights,
        "aspects": aspects,
        "bbox_areas": bbox_areas,
        "norm_bbox_areas": norm_bbox_areas,
        "brightness": brightness,
        "per_category_bbox": per_cat_bbox
    }


train_stats = extract_stats(train_data, train_image_folder)
test_stats = extract_stats(test_data, test_image_folder)


# ---- Normalized BBox Metrics and Auto Suggestions ----
def compute_normalized_bbox_metrics(stats, label):
    areas = np.array(stats["bbox_areas"])
    widths = np.array(stats["widths"])
    heights = np.array(stats["heights"])
    image_areas = widths * heights
    normalized_areas = areas / image_areas

    print(f"==== {label} Dataset BBox Stats ====")
    print(f"Mean Normalized BBox Area     : {np.mean(normalized_areas):.5f}")
    print(f"Median Normalized BBox Area   : {np.median(normalized_areas):.5f}")
    print(f"Min / Max Normalized BBox Area: {np.min(normalized_areas):.5f} / {np.max(normalized_areas):.5f}")
    print(f"Estimated Object Side (px)    : {np.mean(np.sqrt(areas)):.1f}")
    print(f"Objects per image (density)   : {len(areas) / len(widths):.2f}")
    print()

    return {
        "mean": np.mean(normalized_areas),
        "median": np.median(normalized_areas),
        "object_side": np.mean(np.sqrt(areas)),
        "density": len(areas) / len(widths)
    }


train_bbox_metrics = compute_normalized_bbox_metrics(train_stats, "Train")
test_bbox_metrics = compute_normalized_bbox_metrics(test_stats, "Test")


# ---- Auto Suggest Resize Policy ----
def suggest_resize_from_object_size(object_side):
    # Clamp the values to a sane lower and upper range
    base_image_size = int(np.clip(object_side * 20, 512, 1280))  # aim for object ~5% of image
    print("Auto-Suggested Resize Params (Clamped):")
    print(f"  MIN_SIZE_TRAIN = range({int(base_image_size*0.8)}, {int(base_image_size*1.0)})")
    print(f"  MAX_SIZE_TRAIN = {int(base_image_size*1.5)}")
    print(f"  MIN_SIZE_TEST  = {int(base_image_size)}")
    print("  Use with: T.ResizeShortestEdge or T.ResizeScale")
    print()


suggest_resize_from_object_size(test_bbox_metrics["object_side"])


def side_by_side_plot(data1, data2, title, xlabel, label1, label2, filename, bins=150):
    plt.figure(figsize=(12, 5))
    sns.histplot(data1, bins=bins, kde=True, color="blue", label=label1, stat="density", alpha=0.6)
    sns.histplot(data2, bins=bins, kde=True, color="green", label=label2, stat="density", alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# Core comparisons
side_by_side_plot(train_stats["widths"], test_stats["widths"], "Image Width Distribution", "Width (px)", "Train",
                  "Test", "width_comparison.png")
side_by_side_plot(train_stats["heights"], test_stats["heights"], "Image Height Distribution", "Height (px)", "Train",
                  "Test", "height_comparison.png")
side_by_side_plot(train_stats["aspects"], test_stats["aspects"], "Aspect Ratio Distribution", "Width / Height", "Train",
                  "Test", "aspect_ratio_comparison.png")
side_by_side_plot(train_stats["bbox_areas"], test_stats["bbox_areas"], "Bounding Box Area", "BBox Area (px^2)", "Train",
                  "Test", "bbox_area_comparison.png")
side_by_side_plot(train_stats["norm_bbox_areas"], test_stats["norm_bbox_areas"], "Normalized BBox Area",
                  "BBox Area / Image Area", "Train", "Test", "norm_bbox_area_comparison.png")
side_by_side_plot(train_stats["brightness"], test_stats["brightness"], "Image Brightness Comparison", "Mean Brightness",
                  "Train", "Test", "brightness_comparison.png")

# Per-category bbox area plots
for cat_id, cat_name in categories.items():
    train_cat = train_stats["per_category_bbox"].get(cat_id, [])
    test_cat = test_stats["per_category_bbox"].get(cat_id, [])
    if not train_cat and not test_cat:
        continue
    side_by_side_plot(
        train_cat, test_cat,
        f"BBox Area: {cat_name}",
        "BBox Area (px^2)",
        "Train", "Test",
        f"bbox_area_category_{cat_name.replace(' ', '_')}.png"
    )
