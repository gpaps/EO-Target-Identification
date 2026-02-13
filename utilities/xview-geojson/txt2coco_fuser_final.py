import json
import os
import cv2
import shutil
from tqdm import tqdm

# ================= USER CONFIGURATION =================

# 1. INPUT FOLDERS
# I added the new xView tiled folder to your list here:
SOURCE_FOLDERS = [
    # "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/SHIPS_skysat_2025_04_10_piraeus",
    # "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/SHIPS_skysat_2025_04_10_salimina",
    # "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/SHIPS_skysat_2025_09_05_piraeus",
    # "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/Optical/SHIPS_WV_2025_25_11_piraeus",
    "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/Xview/"  # <--- ADDED: Your new tiled xView ships
]

# 2. OUTPUT DESTINATION
OUTPUT_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/Xview/"
OUTPUT_JSON_NAME = "xview_tile_coco.json"

# 3. ID MAPPING (Local ID -> Global ID)
# Assumes xView/SkySat text files use 0=Commercial, 1=Military, etc.
ID_MAP = {
    0: 11,  # Commercial
    1: 12,  # Military
    2: 13,  # Submarines
    3: 14,  # Recreational
    4: 15  # Fishing
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

def yolo_to_coco(x_c, y_c, w, h, img_w, img_h):
    w_px = w * img_w
    h_px = h * img_h
    x_c_px = x_c * img_w
    y_c_px = y_c * img_h

    x_min = x_c_px - (w_px / 2)
    y_min = y_c_px - (h_px / 2)

    return [max(0, x_min), max(0, y_min), w_px, h_px]


def main():
    out_images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(out_images_dir, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON_NAME)

    images_list = []
    annotations_list = []

    img_id_counter = 1
    ann_id_counter = 1
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    print(f"🚀 Starting Consolidation (including xView Tiles)...")

    for source_path in SOURCE_FOLDERS:
        # Strip slash and get name
        source_path = source_path.rstrip(os.sep)
        dataset_name = os.path.basename(source_path)

        # Handle relative path for xview_tiled
        if dataset_name == "." or dataset_name == "":
            dataset_name = "xview_tiled"

        print(f"\nProcessing Dataset: {dataset_name}")

        img_dir = os.path.join(source_path, "images")
        lbl_dir = os.path.join(source_path, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"⚠️  Skipping {dataset_name}: Folder structure not found at {source_path}")
            continue

        files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]

        for filename in tqdm(files, desc=f"Importing {dataset_name}"):
            base_name = os.path.splitext(filename)[0]

            # Check for label (txt)
            txt_path = os.path.join(lbl_dir, base_name + ".txt")
            if not os.path.exists(txt_path):
                continue

            # Rename and Copy
            new_filename = f"{dataset_name}_{filename}"
            dst_img_path = os.path.join(out_images_dir, new_filename)
            src_img_path = os.path.join(img_dir, filename)

            try:
                img = cv2.imread(src_img_path)
                if img is None: continue
                height, width = img.shape[:2]
                shutil.copy2(src_img_path, dst_img_path)
            except:
                continue

            images_list.append({
                "id": img_id_counter,
                "file_name": new_filename,
                "width": width,
                "height": height
            })

            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue

                    local_id = int(parts[0])
                    if local_id in ID_MAP:
                        global_id = ID_MAP[local_id]
                        x_c, y_c, w, h = map(float, parts[1:])
                        bbox = yolo_to_coco(x_c, y_c, w, h, width, height)

                        annotations_list.append({
                            "id": ann_id_counter,
                            "image_id": img_id_counter,
                            "category_id": global_id,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        })
                        ann_id_counter += 1

            img_id_counter += 1

    # Save
    coco_output = {
        "info": {"description": "Merged Dataset (SkySat + xView Tiled)", "year": 2025},
        "licenses": [],
        "images": images_list,
        "annotations": annotations_list,
        "categories": CATEGORIES
    }

    with open(json_path, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"\n✅ DONE! Merged dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()