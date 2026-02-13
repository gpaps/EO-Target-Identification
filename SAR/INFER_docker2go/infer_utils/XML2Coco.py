import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Configuration
# target_classes = ["Commercial", "Military", "Submarine", "Recreational Boat", "Fishing Boat"]
# target_classes = ["aircraft", "helicopter"]
# target_classes = ["ships"]
target_classes = ["oil tanks", "harbour", "bridge"]
category_mapping = {name: i + 1 for i, name in enumerate(target_classes)}
# input_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/aircrafts/labels/"

# input_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/labels/"
# input_dir = "annot/labels_my-project-name_2025-09-20-03-51-31_vocxml/"
# output_json = "/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/json/coco_train2.json"
# output_json = "/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/json/coco_train2.json"
# output_json = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/labels/planes_thess_coco.json"
# input_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/labels/"
# output_json = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/opt_ships.json"
input_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/infrastructure/labels/"
output_json = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/infrastructure/opt_infra.json"
# COCO
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [{"id": v, "name": k, "supercategory": "none"} for k, v in category_mapping.items()]
}

annotation_id = 1
image_id = 1

for xml_file in tqdm(os.listdir(input_dir)):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(input_dir, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

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
        if label not in category_mapping:
            continue

        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        width_box = xmax - xmin
        height_box = ymax - ymin

        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_mapping[label],
            "bbox": [xmin, ymin, width_box, height_box],
            "area": width_box * height_box,
            "iscrowd": 0
        })
        annotation_id += 1

    image_id += 1

# Save JSON
with open(output_json, "w") as f:
    json.dump(coco_output, f, indent=4)

print(f"✅ COCO JSON saved to {output_json}")
