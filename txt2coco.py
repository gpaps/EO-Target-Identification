import json
import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2

# ================= USER CONFIGURATION =================

# 1. INPUT FOLDERS (Updated to the correct path found in your logs)
SOURCE_FOLDERS = [
    "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/SHIPS_skysat_2025_04_10_piraeus",
    "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/SHIPS_skysat_2025_04_10_salimina",
    "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/SHIPS_skysat_2025_09_05_piraeus",
    "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/SHIPS_WV_2025_25_11_piraeus",
]

# 2. OUTPUT DESTINATION
OUTPUT_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/CONSOLIDATED_SHIPS"
OUTPUT_JSON_NAME = "instances_ships_consolidated.json"

# 3. CLASS MAPPING
# Maps the XML <name> (which is "11", "12" etc.) to COCO ID
CLASS_MAP = {
    "11": 11,  # Commercial
    "12": 12,  # Military
    "13": 13,  # Submarines
    "14": 14,  # Recreational
    "15": 15  # Fishing
}

# 4. FINAL CATEGORY NAMES
CATEGORIES = [
    {"id": 11, "name": "Commercial"},
    {"id": 12, "name": "Military"},
    {"id": 13, "name": "Submarines"},
    {"id": 14, "name": "Recreational"},
    {"id": 15, "name": "Fishing"}
]


# =================================================

def main():
    # Setup Output
    out_images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(out_images_dir, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON_NAME)

    images_list = []
    annotations_list = []

    img_id_counter = 1
    ann_id_counter = 1

    # Supported image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    print(f"🚀 Starting XML -> COCO Consolidation...")
    print(f"📂 Output: {out_images_dir}")

    for source_path in SOURCE_FOLDERS:
        # Strip trailing slashes to prevent path errors
        source_path = source_path.rstrip(os.sep)
        dataset_name = os.path.basename(source_path)

        print(f"\n------------------------------------------------")
        print(f"Processing Dataset: {dataset_name}")

        img_dir = os.path.join(source_path, "images")
        lbl_dir = os.path.join(source_path, "labels")

        # PATH CHECK
        if not os.path.exists(img_dir):
            print(f"❌ ERROR: Images folder NOT found at: {img_dir}")
            continue
        if not os.path.exists(lbl_dir):
            print(f"❌ ERROR: Labels folder NOT found at: {lbl_dir}")
            continue

        # Get all images
        files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]

        if not files:
            print(f"⚠️ No images found in {img_dir}")
            continue

        success_count = 0

        for filename in tqdm(files, desc=f"Importing {dataset_name}"):
            base_name = os.path.splitext(filename)[0]

            # Look for XML (Assume same basename + .xml)
            xml_name = base_name + ".xml"
            xml_path = os.path.join(lbl_dir, xml_name)

            if not os.path.exists(xml_path):
                # Print once per dataset to avoid spamming console
                # print(f"Skipping {filename} - No XML found")
                continue

            # 1. Parse XML
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                print(f"Error parsing {xml_path}: {e}")
                continue

            # 2. Rename & Copy Image
            new_filename = f"{dataset_name}_{filename}"
            dst_img_path = os.path.join(out_images_dir, new_filename)
            src_img_path = os.path.join(img_dir, filename)

            # Get Dims (Prefer XML, fallback to Image read)
            size_node = root.find("size")
            if size_node is not None:
                width = int(size_node.find("width").text)
                height = int(size_node.find("height").text)
                shutil.copy2(src_img_path, dst_img_path)
            else:
                img = cv2.imread(src_img_path)
                if img is None: continue
                height, width = img.shape[:2]
                shutil.copy2(src_img_path, dst_img_path)

            # 3. Add Image to JSON
            images_list.append({
                "id": img_id_counter,
                "file_name": new_filename,
                "width": width,
                "height": height
            })

            # 4. Extract Annotations
            has_valid_obj = False
            for obj in root.findall("object"):
                name = obj.find("name").text

                # Check mapping
                if name not in CLASS_MAP:
                    continue

                category_id = CLASS_MAP[name]

                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                w_box = xmax - xmin
                h_box = ymax - ymin

                if w_box <= 0 or h_box <= 0:
                    continue

                annotations_list.append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, w_box, h_box],
                    "area": w_box * h_box,
                    "iscrowd": 0
                })
                ann_id_counter += 1
                has_valid_obj = True

            if has_valid_obj:
                success_count += 1

            img_id_counter += 1

        print(f"✅ Imported {success_count} images from {dataset_name}")

    # 5. Save Final JSON
    coco_output = {
        "info": {"description": "Consolidated Ship Dataset (XML Source)", "year": 2025},
        "licenses": [],
        "images": images_list,
        "annotations": annotations_list,
        "categories": CATEGORIES
    }

    with open(json_path, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"\n DONE! Merged Dataset Created at: {OUTPUT_DIR}")
    print(f" - Total Images: {len(images_list)}")
    print(f" - Total Annotations: {len(annotations_list)}")


if __name__ == "__main__":
    main()