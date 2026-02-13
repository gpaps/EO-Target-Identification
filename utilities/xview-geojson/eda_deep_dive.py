from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==== CONFIGURATION ====
JSON_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/coco_dataset_cleaned.json"

# Thresholds for "Suspicious" Data
MIN_AREA_THRESH = 10  # Pixels^2 (Box area smaller than this is suspicious)
MAX_ASPECT_RATIO = 10  # (Width/Height > 10 is likely a line/glitch)


# =======================

def main():
    coco = COCO(JSON_PATH)

    # 1. Collect Detailed Data
    data = []
    suspicious_tiny = []
    suspicious_ratio = []

    cats = coco.loadCats(coco.getCatIds())
    cat_names = {c['id']: c['name'] for c in cats}

    print("\n🔍 SCANNING FOR ANOMALIES...")

    for ann_id, ann in coco.anns.items():
        w, h = ann['bbox'][2], ann['bbox'][3]
        area = w * h
        ratio = w / h if h > 0 else 0
        img_id = ann['image_id']
        cat_name = cat_names[ann['category_id']]

        # Log Logic
        data.append({
            "Category": cat_name,
            "Width": w,
            "Height": h,
            "Area": area,
            "AspectRatio": ratio
        })

        # Check Anomalies
        if area < MIN_AREA_THRESH:
            suspicious_tiny.append((img_id, ann_id, area, cat_name))

        if ratio > MAX_ASPECT_RATIO or ratio < (1 / MAX_ASPECT_RATIO):
            suspicious_ratio.append((img_id, ann_id, ratio, cat_name))

    df = pd.DataFrame(data)

    # 2. Print Anomaly Report
    print(f"\n⚠️ FOUND {len(suspicious_tiny)} TINY BOXES (Area < {MIN_AREA_THRESH} px²)")
    print("   (Showing first 5 examples)")
    for img, ann, val, cat in suspicious_tiny[:5]:
        img_info = coco.loadImgs(img)[0]
        print(f"   - Image: {img_info['file_name']} | ID: {ann} | Class: {cat} | Area: {val:.2f}")

    print(f"\n⚠️ FOUND {len(suspicious_ratio)} EXTREME ASPECT RATIOS (Ratio > {MAX_ASPECT_RATIO}:1)")
    print("   (Showing first 5 examples)")
    for img, ann, val, cat in suspicious_ratio[:5]:
        img_info = coco.loadImgs(img)[0]
        print(f"   - Image: {img_info['file_name']} | ID: {ann} | Class: {cat} | Ratio: {val:.2f}")

    # 3. Visualization
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deep Dive Dataset Analysis", fontsize=16)

    # Plot A: Box Area per Class (Log Scale)
    sns.boxplot(x="Category", y="Area", data=df, ax=axes[0, 0], showfliers=False)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Box Area Distribution per Class (Log Scale)")
    axes[0, 0].set_ylabel("Area (pixels²)")
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot B: Aspect Ratio per Class
    sns.boxplot(x="Category", y="AspectRatio", data=df, ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_title("Aspect Ratio Distribution per Class")
    axes[0, 1].axhline(1, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot C: Width vs Height Heatmap (Log Scale)
    # This helps see if we have clusters of specific sizes
    h = axes[1, 0].hist2d(df['Width'], df['Height'], bins=50, norm='log', cmap='magma')
    axes[1, 0].set_title("Object Dimensions Heatmap (Log Scale)")
    axes[1, 0].set_xlabel("Width")
    axes[1, 0].set_ylabel("Height")
    plt.colorbar(h[3], ax=axes[1, 0])

    # Plot D: Instance Count per Class
    sns.countplot(x="Category", data=df, ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Total Instances per Class")
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("eda_deep_dive2.png")
    print("\n✅ Plots saved to 'eda_deep_dive.png'. Check this image to see class imbalances.")


if __name__ == "__main__":
    main()