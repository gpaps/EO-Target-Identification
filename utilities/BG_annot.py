import os
import json
from PIL import Image

# ✅ Set your folder and output path here
BG_FOLDER = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/BGs_/"
OUTPUT_JSON = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/bg_coco.json"
ID_OFFSET = 10000  # To avoid collision (adjust if needed)

def generate_bg_coco_json(bg_folder, output_json, id_offset=10000):
    image_entries = []

    # image_files = sorted([f for f in os.listdir(bg_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    image_files = sorted([
        f for f in os.listdir(bg_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    ])

    for idx, filename in enumerate(image_files):
        new_filename = f"BGSAT_{idx:05d}.jpg"
        old_path = os.path.join(bg_folder, filename)
        new_path = os.path.join(bg_folder, new_filename)
        os.rename(old_path, new_path)

        with Image.open(new_path) as img:
            width, height = img.size

        image_entries.append({
            "id": id_offset + idx,
            "file_name": new_filename,
            "width": width,
            "height": height
        })

    coco_json = {
        "images": image_entries,
        "annotations": [],
        "categories": []
    }

    with open(output_json, "w") as f:
        json.dump(coco_json, f, indent=2)

    print(f"✅ COCO JSON with {len(image_entries)} BG images saved to: {output_json}")

# 🔁 Run it
generate_bg_coco_json(BG_FOLDER, OUTPUT_JSON, ID_OFFSET)
