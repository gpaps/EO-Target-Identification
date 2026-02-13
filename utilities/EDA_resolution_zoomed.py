import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Output
output_dir = "../Optical/Optical_EDA_outputs/Optical_EDA_outputs_discrepancy/"
os.makedirs(output_dir, exist_ok=True)

# JSON files
train_json_path = "../Optical/json/VHRShips_Imagenet_Consolidated.json"
test_json_path = "../Optical/Skysatt_bench_test.json"

# Load data
with open(train_json_path) as f:
    train_data = json.load(f)
with open(test_json_path) as f:
    test_data = json.load(f)


def extract_resolution_stats(coco):
    resolutions, widths, heights = [], [], []
    for img in coco["images"]:
        w, h = img["width"], img["height"]
        widths.append(w)
        heights.append(h)
        resolutions.append(w * h)
    return np.array(resolutions), np.array(widths), np.array(heights)


# Extract stats
train_res, train_w, train_h = extract_resolution_stats(train_data)
test_res, test_w, test_h = extract_resolution_stats(test_data)

# Plot zoomed-in resolution (<2MP)
plt.figure(figsize=(12, 5))
sns.histplot(train_res[train_res < 2_000_000], bins=150, kde=True, color="blue", label="Train", stat="density",
             alpha=0.6)
sns.histplot(test_res[test_res < 2_000_000], bins=150, kde=True, color="green", label="Test", stat="density", alpha=0.6)
plt.title("Resolution Distribution (Filtered: < 2MP)")
plt.xlabel("Resolution (px²)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "resolution_zoomed_under_2mp.png"))
plt.close()


# Print resolution statistics
def summarize(name, widths, heights, resolutions):
    print(f"=== {name} ===")
    print(f" Width  - min: {widths.min()}, mean: {widths.mean():.1f}, max: {widths.max()}")
    print(f" Height - min: {heights.min()}, mean: {heights.mean():.1f}, max: {heights.max()}")
    print(f" Res    - min: {resolutions.min()}, mean: {resolutions.mean():.1f}, max: {resolutions.max()}")
    print()


summarize("Train", train_w, train_h, train_res)
summarize("Test", test_w, test_h, test_res)


# Auto-suggest ResizeShortestEdge
def suggest_resize_range(widths, heights):
    min_side = np.percentile(np.minimum(widths, heights), [10, 50, 90])
    max_side = np.percentile(np.maximum(widths, heights), [10, 50, 90])
    print("Auto-suggested ResizeShortestEdge parameters based on percentiles:")
    print(f"  MIN_SIZE_TRAIN: range({int(min_side[0])}, {int(min_side[2])})")
    print(f"  MAX_SIZE_TRAIN: {int(max_side[2])}")
    print()
    print("This helps match the test image scale more closely and preserve bounding box ratios.")


suggest_resize_range(test_w, test_h)
