import json
import os
import random
import re
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# INPUT: Use your FINAL cleaned dataset here
INPUT_JSON = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/coco_dataset_cleaned.json"
OUTPUT_DIR = os.path.dirname(INPUT_JSON)

# RATIO: 90% Train / 10% Val
TRAIN_RATIO = 0.90
# Val is automatically 1.0 - TRAIN_RATIO

SEED = 42


# =================================================

def extract_base_filename(filename):
    # Regex to strip _numbers_numbers tiling suffix to prevent leakage
    base = re.sub(r'_\d+_\d+\.(jpg|png|tif|bmp)$', '', filename, flags=re.IGNORECASE)
    base = os.path.splitext(base)[0]
    return base


def save_coco(data, subset_name, output_dir):
    filename = os.path.join(output_dir, f"coco_{subset_name}.json")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"   Saved {subset_name.upper()}: {filename}")


def plot_distribution(distributions, class_names):
    labels = list(class_names.values())

    train_counts = [distributions['train'].get(i, 0) for i in class_names]
    val_counts = [distributions['val'].get(i, 0) for i in class_names]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, train_counts, width, label='Train (90%)', color='#4CAF50')
    ax.bar(x + width / 2, val_counts, width, label='Val (10%)', color='#2196F3')

    ax.set_ylabel('Number of Annotations')
    ax.set_title('Class Distribution (90/10 Stratified)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "split_distribution_90_10.png"))
    print("   📊 Saved distribution plot to split_distribution_90_10.png")


def main():
    random.seed(SEED)
    print(f"🚀 Loading {os.path.basename(INPUT_JSON)}...")
    with open(INPUT_JSON, 'r') as f:
        coco = json.load(f)

    # 1. Group Images by Parent Scene (Anti-Leakage)
    scene_groups = defaultdict(list)
    img_id_to_cats = defaultdict(set)

    for ann in coco['annotations']:
        img_id_to_cats[ann['image_id']].add(ann['category_id'])

    for img in coco['images']:
        base_name = extract_base_filename(img['file_name'])

        # Priority Tagging
        cats = img_id_to_cats[img['id']]
        priority_tag = "Common"

        # Priority: Submarine (13) > Fishing (15) > Military (12)
        if 13 in cats:
            priority_tag = "Submarine"
        elif 15 in cats:
            priority_tag = "Fishing"
        elif 12 in cats:
            priority_tag = "Military"

        scene_groups[base_name].append({
            "image": img,
            "cats": cats,
            "tag": priority_tag
        })

    print(f"   Found {len(scene_groups)} unique parent scenes from {len(coco['images'])} images.")

    # 2. Perform Grouped Stratified Split (Just 2 buckets now)
    scene_keys = list(scene_groups.keys())
    scene_tags = [scene_groups[k][0]['tag'] for k in scene_keys]

    train_scenes, val_scenes = train_test_split(
        scene_keys,
        train_size=TRAIN_RATIO,
        stratify=scene_tags,
        random_state=SEED
    )

    # 3. Build Final JSONs
    splits = {"train": train_scenes, "val": val_scenes}
    final_stats = defaultdict(lambda: defaultdict(int))
    cat_map = {c['id']: c['name'] for c in coco['categories']}

    for split_name, scenes in splits.items():
        split_data = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": [],
            "annotations": []
        }

        split_img_ids = set()
        for scene in scenes:
            for item in scene_groups[scene]:
                img = item['image']
                split_data['images'].append(img)
                split_img_ids.add(img['id'])

        for ann in coco['annotations']:
            if ann['image_id'] in split_img_ids:
                split_data['annotations'].append(ann)
                final_stats[split_name][ann['category_id']] += 1

        save_coco(split_data, split_name, OUTPUT_DIR)
        print(
            f"   {split_name.upper()}: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")

    # 4. Verification
    plot_distribution(final_stats, cat_map)
    print("\n✅ 90/10 Split Complete. Ready for training.")


if __name__ == "__main__":
    main()