import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- Hardcoded paths ---
# XML_DIR = "/home/gpaps/PycharmProject/Esa_Aircrafts/sar_bench/XMLv0/"  # Replace with your real path
XML_DIR = "XML/"  # Replace with your real path
OUTPUT_JSON = "sar_ship.json"
CATEGORY_NAME = "boats"

def convert_xmls_to_coco(xml_dir, output_json, category_name="airplane"):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": category_name,
            "supercategory": category_name
        }]
    }

    ann_id = 1
    for img_id, xml_file in enumerate(tqdm(os.listdir(xml_dir))):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name != category_name:
                continue

            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            w = xmax - xmin
            h = ymax - ymin

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"Done. {len(coco['annotations'])} annotations written to {output_json}")

# --- Run it directly ---
if __name__ == "__main__":
    convert_xmls_to_coco(XML_DIR, OUTPUT_JSON, CATEGORY_NAME)
