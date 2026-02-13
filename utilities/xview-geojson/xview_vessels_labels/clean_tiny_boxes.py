from pycocotools.coco import COCO
import json
import os

# ==== CONFIGURATION ====
INPUT_JSON = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/clustered_coco_.json"
OUTPUT_JSON = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/coco_dataset_cleaned.json"

# Removal Thresholds
MIN_AREA = 10  # Pixels squared (Delete anything smaller than this)


# =======================

def main():
    print(f"🧹 Loading dataset: {os.path.basename(INPUT_JSON)}...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    original_count = len(data['annotations'])
    clean_annotations = []
    removed_count = 0
    removed_examples = []

    for ann in data['annotations']:
        width = ann['bbox'][2]
        height = ann['bbox'][3]
        area = width * height

        # Check for Tiny Area
        if area < MIN_AREA:
            removed_count += 1
            if len(removed_examples) < 5:
                removed_examples.append(f"ID {ann['id']} (Area: {area:.2f})")
            continue  # SKIP adding this annotation (Effectively deleting it)

        # (Optional) Check for Extreme Aspect Ratio
        # ratio = width / height if height > 0 else 0
        # if ratio > 20 or ratio < 0.05:
        #    continue

        # If it passes checks, keep it
        clean_annotations.append(ann)

    # Update dataset
    data['annotations'] = clean_annotations

    print(f"\n📊 Cleaning Summary:")
    print(f"   - Original Annotations: {original_count}")
    print(f"   - Tiny Boxes Removed:   {removed_count}")
    print(f"   - Final Annotations:    {len(clean_annotations)}")

    if removed_examples:
        print(f"\n   Examples of deleted boxes: {removed_examples}")

    # Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"\n✅ Cleaned dataset saved to:\n   {OUTPUT_JSON}")


if __name__ == "__main__":
    main()