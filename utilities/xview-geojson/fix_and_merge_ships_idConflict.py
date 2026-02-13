import json
import os

# ================= CONFIGURATION =================
# 1. The Instance file (Needs ID fixing: 11->1, 12->2...)
INSTANCES_JSON = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/CONSOLIDATED_SHIPS/instances_ships_consolidated.json"

# 2. The Master Benchmark file (Has correct IDs: 1, 2...)
MASTER_JSON = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/Annot_VHRShips_ShipRSImageNEt/VHRShips_ShipRSImageNet_master.json"

# 3. Output Merged File
OUTPUT_JSON = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/coco_dataset.json"

# ID Mapping (From Instances -> Master)
ID_FIX_MAP = {
    11: 1,  # Commercial
    12: 2,  # Military
    13: 3,  # Submarines
    14: 4,  # Recreational Boats
    15: 5  # Fishing Boats
}


# =================================================

def fix_instance_ids(json_path, output_path):
    print(f"🔧 Fixing IDs in {os.path.basename(json_path)}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Fix Category Definitions
    fixed_count = 0
    for cat in data['categories']:
        if cat['id'] in ID_FIX_MAP:
            cat['id'] = ID_FIX_MAP[cat['id']]
            fixed_count += 1

    # 2. Fix Annotations
    ann_fixed_count = 0
    for ann in data['annotations']:
        if ann['category_id'] in ID_FIX_MAP:
            ann['category_id'] = ID_FIX_MAP[ann['category_id']]
            ann_fixed_count += 1

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"   ✅ Remapped {fixed_count} categories and {ann_fixed_count} annotations.")
    print(f"   Saved to: {os.path.basename(output_path)}")
    return output_path


def merge_datasets(input_paths, output_path):
    print(f"\n Merging {len(input_paths)} datasets...")
    merged_data = {
        "info": {"description": "Merged Ship Dataset", "year": 2025},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    category_name_to_id = {}
    image_id_counter = 0
    annotation_id_counter = 0

    # PROCESS MASTER FILE FIRST to lock in the correct ID order (1, 2, 3, 4, 5)
    for json_path in input_paths:
        dataset_name = os.path.basename(json_path)
        print(f"   Processing: {dataset_name}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # 1. Merge Categories (Based on Name)
        old_cat_id_to_new = {}
        for cat in data["categories"]:
            name = cat["name"]
            # If category not seen yet, add it
            if name not in category_name_to_id:
                # Use existing ID if possible, or generate new
                new_id = cat['id']
                # Check conflict: if this ID is already taken by a diff name, we must change it.
                # But since we run Master first, it sets the standard.
                if new_id in category_name_to_id.values():
                    new_id = len(category_name_to_id) + 1

                category_name_to_id[name] = new_id
                merged_data["categories"].append({"id": new_id, "name": name})

            old_cat_id_to_new[cat["id"]] = category_name_to_id[name]

        # 2. Merge Images (Re-index IDs to avoid collisions)
        old_image_id_to_new = {}
        for img in data["images"]:
            new_image_id = image_id_counter
            old_image_id_to_new[img["id"]] = new_image_id

            # Create new image entry
            new_img = img.copy()
            new_img["id"] = new_image_id
            merged_data["images"].append(new_img)
            image_id_counter += 1

        # 3. Merge Annotations
        for ann in data["annotations"]:
            new_ann = ann.copy()
            new_ann["id"] = annotation_id_counter
            new_ann["image_id"] = old_image_id_to_new[ann["image_id"]]
            new_ann["category_id"] = old_cat_id_to_new[ann["category_id"]]

            merged_data["annotations"].append(new_ann)
            annotation_id_counter += 1

    # Save
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"\n SUCCESS! Final merged dataset saved to:\n   {output_path}")
    print(f"   - Total Images: {len(merged_data['images'])}")
    print(f"   - Total Annotations: {len(merged_data['annotations'])}")
    print(f"   - Categories: {[c['name'] for c in merged_data['categories']]}")


if __name__ == "__main__":
    # Step 1: Fix the Instances file
    fixed_instances_path = INSTANCES_JSON.replace(".json", "_fixed.json")
    fix_instance_ids(INSTANCES_JSON, fixed_instances_path)

    # Step 2: Merge (Master First + Fixed Instances)
    # Putting MASTER_JSON first ensures the final JSON uses its category IDs.
    datasets_to_merge = [MASTER_JSON, fixed_instances_path]

    merge_datasets(datasets_to_merge, OUTPUT_JSON)