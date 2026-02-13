import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
# UNCOMMENT the pair you want to process
# PAIR 1: JPGs
input_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Benchmark_Dataset/Ships/annotations/BMP_ship/"  # Update this path
output_json = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Benchmark_Dataset/Ships/annotations/ship_BMP.json"

# PAIR 2: BMPs (Run this script a second time with these paths)
# input_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Benchmark_Dataset/Ships/XML_BMP/"
# output_json = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Benchmark_Dataset/Ships/annotations/ship_BMP_fixed.json"

# Classes
target_classes = ["Commercial", "Military", "Submarines", "Recreational Boats", "Fishing Boats"]
category_mapping = {name: i + 1 for i, name in enumerate(target_classes)}
# -----------------------------------------------

# COCO Structure
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [{"id": v, "name": k, "supercategory": "none"} for k, v in category_mapping.items()]
}

annotation_id = 1
image_id = 1  # <--- CRITICAL: Must be initialized OUTSIDE the loop

print(f" Starting conversion for: {input_dir}")

# Filter only XML files
xml_files = [f for f in os.listdir(input_dir) if f.endswith(".xml")]

for xml_file in tqdm(xml_files):
    try:
        tree = ET.parse(os.path.join(input_dir, xml_file))
        root = tree.getroot()

        filename = root.find("filename").text

        # Safety check for size
        size_node = root.find("size")
        if size_node is None:
            # Fallback if size is missing
            width, height = 1024, 1024
        else:
            width = int(size_node.find("width").text)
            height = int(size_node.find("height").text)

        # Add image entry
        coco_output["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # Process objects
        for obj in root.findall("object"):
            label = obj.find("name").text

            # Normalize label names (optional, handles case sensitivity)
            # label = label.lower().capitalize()

            if label not in category_mapping:
                continue

            bbox = obj.find("bndbox")

            # PRECISION FIX: Use float - 1 for 0-indexed COCO
            xmin = float(bbox.find("xmin").text) - 1
            ymin = float(bbox.find("ymin").text) - 1
            xmax = float(bbox.find("xmax").text) - 1
            ymax = float(bbox.find("ymax").text) - 1

            width_box = max(0, xmax - xmin)
            height_box = max(0, ymax - ymin)

            # Skip invalid 0-area boxes
            if width_box < 1 or height_box < 1:
                continue

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_mapping[label],
                "bbox": [xmin, ymin, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })
            annotation_id += 1

        # CRITICAL: Increment Image ID for the next file
        image_id += 1

    except Exception as e:
        print(f"❌ Error processing {xml_file}: {e}")

# Save JSON
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w") as f:
    json.dump(coco_output, f, indent=4)

print(f"\n✅ Saved to {output_json}")
print(f" - Total Images: {len(coco_output['images'])}")
print(f" - Unique IDs: {len(set(img['id'] for img in coco_output['images']))} (Should match Total Images)")
print(f" - Total Annotations: {len(coco_output['annotations'])}")