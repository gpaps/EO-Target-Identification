import os
import json
import glob
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
TILE_SIZE = 768  # 448  # 256 #386  # 512
STRIDE = 576  # 224  # 64 #128  # 256
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships_dataset/"
# ANNOT_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships.json"
# OUT_IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/800x800_Images_crop_400p"
# OUT_ANN_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/800x88sar_ships.json"

IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/JPG/"
ANNOT_PATH = "/home/gpaps/PycharmProject/Esa_Ships/opt_ships.json"
OUT_IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/temp768/"
OUT_ANN_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/temp576/opt_ships.json"

# --- Setup output dirs ---
os.makedirs(OUT_IMAGE_DIR, exist_ok=True)

# --- Collect all available images ---
img_candidates = []
for ext in ["*.png", "*.PNG", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
    img_candidates.extend(glob.glob(os.path.join(IMAGE_DIR, "**", ext), recursive=True))
img_map = {os.path.basename(p).lower(): p for p in img_candidates}

# --- Load original COCO ---
with open(ANNOT_PATH, "r") as f:
    coco = json.load(f)

new_images = []
new_anns = []
ann_id_counter = 1
img_id_counter = 1

# --- Loop through COCO images ---
for image in tqdm(coco["images"]):
    img_name = os.path.basename(image["file_name"]).lower()
    img_path = img_map.get(img_name)

    if not img_path or not os.path.exists(img_path):
        print(f" Skipping missing image: {img_name}")
        continue

    img = Image.open(img_path)
    w, h = img.size
    anns_for_img = [a for a in coco["annotations"] if a["image_id"] == image["id"]]

    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            x_end, y_end = min(x + TILE_SIZE, w), min(y + TILE_SIZE, h)

            if x_end - x < TILE_SIZE or y_end - y < TILE_SIZE:
                continue  # skip partial tiles

            tile = img.crop((x, y, x_end, y_end))
            # Patch: Pad tile if it's smaller than TILE_SIZE
            if tile.size != (TILE_SIZE, TILE_SIZE):
                tile = tile.resize((TILE_SIZE, TILE_SIZE), resample=Image.BILINEAR)

            tile_filename = f"{os.path.splitext(img_name)[0]}_{x}_{y}.png"
            tile_path = os.path.join(OUT_IMAGE_DIR, tile_filename)
            tile.save(tile_path)

            # tile_filename = f"{os.path.splitext(img_name)[0]}_{x}_{y}.png"
            # tile_path = os.path.join(OUT_IMAGE_DIR, tile_filename)
            # tile.save(tile_path)

            new_images.append({
                "id": img_id_counter,
                "file_name": tile_filename,
                "width": TILE_SIZE,
                "height": TILE_SIZE
            })

            for ann in anns_for_img:
                bx, by, bw, bh = ann["bbox"]
                if bx + bw < x or bx > x_end or by + bh < y or by > y_end:
                    continue  # does not intersect

                new_bbox = [
                    max(0, bx - x),
                    max(0, by - y),
                    min(bw, x_end - bx),
                    min(bh, y_end - by)
                ]

                # Patch: extra sanity checks for bbox
                if (
                    new_bbox[2] <= 0 or new_bbox[3] <= 0 or
                    new_bbox[2] > TILE_SIZE or new_bbox[3] > TILE_SIZE
                ):
                    continue

                new_anns.append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": ann["category_id"],
                    "bbox": new_bbox,
                    "area": new_bbox[2] * new_bbox[3],
                    "iscrowd": 0
                })
                ann_id_counter += 1

            img_id_counter += 1

# --- Save tiled JSON ---
tiled_coco = {
    "images": new_images,
    "annotations": new_anns,
    "categories": coco["categories"]
}
os.makedirs(os.path.dirname(OUT_ANN_PATH), exist_ok=True)
with open(OUT_ANN_PATH, "w") as f:
    json.dump(tiled_coco, f, indent=2)
