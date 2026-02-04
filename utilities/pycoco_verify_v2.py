from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load COCO JSON
# json_path = "Skysatt_bench_test.json"
# json_path = "Optical/docker2go/json/coco_train.json"
# json_path = "../Skysatt_bench_test.json"
# json_path = "../Optical/VHRships_Imagenet/json/coco_train.json"
# json_path = "/home/gpaps/PycharmProject/Esa_Ships/Optical/json/VHRShips_Imagenet_Consolidated.json"
# json_path = "/home/gpaps/PycharmProject/Esa_Ships/SAR/json/SAR_Ships_[HRSID-SSDD]_cleaned.json"
# json_path = "Optical/docker2go/VHR-Imagene_json_with_greekBG/coco_train.json"
# json_path = "SAR/docker2go/json/coco_train.json"
# json_path = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Benchmark_Dataset/Ships/annotations/master.json"
json_path = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/coco_dataset_cleaned.json"
coco = COCO(json_path)

# Print basic dataset info
print("\n✅ COCO Loaded Successfully")
print(f"Total Images: {len(coco.imgs)}")
print(f"Total Annotations: {len(coco.anns)}")

# Data structures to store statistics
category_image_count = defaultdict(set)  # Images per category
category_instance_count = defaultdict(int)  # Instances per category
bb_widths, bb_heights, bb_areas, bb_aspect_ratios = [], [], [], []
images_with_multiple_instances = defaultdict(int)

# Process annotations
for ann in coco.anns.values():
    cat_id = ann["category_id"]
    img_id = ann["image_id"]
    bbox = ann["bbox"]  # [x, y, width, height]

    # Collect category-wise data
    category_image_count[cat_id].add(img_id)
    category_instance_count[cat_id] += 1

    # Store bounding box properties
    width, height = bbox[2], bbox[3]
    bb_widths.append(width)
    bb_heights.append(height)
    bb_areas.append(width * height)
    bb_aspect_ratios.append(width / height if height > 0 else 0)

    # Count images with multiple instances
    images_with_multiple_instances[img_id] += 1

# Print per-category statistics
print("\n **Category Statistics**")
for cat_id, cat_info in coco.cats.items():
    cat_name = cat_info["name"]
    num_images = len(category_image_count[cat_id])
    num_instances = category_instance_count[cat_id]

    print(f" Category: {cat_name}")
    print(f" - Total Images: {num_images}")
    print(f" - Total Instances: {num_instances}")
    if num_images > 0:
        print(f"   - Avg Instances per Image: {num_instances / num_images:.2f}")
    else:
        print(f"   - Avg Instances per Image: N/A (No images found)")


# Print dataset insights
multi_instance_images = sum(1 for count in images_with_multiple_instances.values() if count > 1)

print("\n Dataset Insights")
if len(coco.imgs) > 0:
    print(f"Total images with multiple objects: {multi_instance_images} ({multi_instance_images / len(coco.imgs) * 100:.2f}%)")
else:
    print("Total images with multiple objects: 0 (No images found)")

if bb_widths and bb_heights and bb_areas and bb_aspect_ratios:
    print("Bounding Box Size Summary:")
    print(f"   - Min Width: {min(bb_widths):.1f}, Max Width: {max(bb_widths):.1f}, Avg: {np.mean(bb_widths):.1f}")
    print(f"   - Min Height: {min(bb_heights):.1f}, Max Height: {max(bb_heights):.1f}, Avg: {np.mean(bb_heights):.1f}")
    print(f"   - Min Area: {min(bb_areas):.1f}, Max Area: {max(bb_areas):.1f}, Avg: {np.mean(bb_areas):.1f}")
    print("Aspect Ratio Summary (Width/Height):")
    print(f"   - Min: {min(bb_aspect_ratios):.2f},"
          f" Max: {max(bb_aspect_ratios):.2f},"
          f" Avg: {np.mean(bb_aspect_ratios):.2f}")
else:
    print("Bounding Box Size Summary: No bounding boxes available.")

# --- VISUALIZATIONS ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Bounding Box & Aspect Ratio Distributions", fontsize=16)

# Histogram of bounding box widths
axes[0, 0].hist(bb_widths, bins=30, color='blue', alpha=0.7)
axes[0, 0].set_title("Bounding Box Width Distribution")
axes[0, 0].set_xlabel("Width (pixels)")
axes[0, 0].set_ylabel("Frequency")

# Histogram of bounding box heights
axes[0, 1].hist(bb_heights, bins=30, color='red', alpha=0.7)
axes[0, 1].set_title("Bounding Box Height Distribution")
axes[0, 1].set_xlabel("Height (pixels)")
axes[0, 1].set_ylabel("Frequency")

# Histogram of aspect ratios
axes[1, 0].hist(bb_aspect_ratios, bins=30, color='green', alpha=0.7)
axes[1, 0].set_title("Bounding Box Aspect Ratio Distribution")
axes[1, 0].set_xlabel("Aspect Ratio (Width/Height)")
axes[1, 0].set_ylabel("Frequency")

# Scatter plot of width vs. height
axes[1, 1].scatter(bb_widths, bb_heights, alpha=0.5, color='purple')
axes[1, 1].set_title("Bounding Box Width vs. Height")
axes[1, 1].set_xlabel("Width (pixels)")
axes[1, 1].set_ylabel("Height (pixels)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("\n Analysis Complete!")