import os
import json
import glob
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
TILE_SIZE = 256  # 448  # 256 #386  # 512
STRIDE = 256   # 224  # 64  #128  # 256

# SAR example (commented out)
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships_dataset/"
# ANNOT_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships.json"
# OUT_IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/800x800_Images_crop_400p"
# OUT_ANN_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/800x88sar_ships.json"

# OPTICAL example
IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/images/PNG/"
ANNOT_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/labels/lean/optical_airv2.json"
OUT_IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/images/temp256/"
OUT_ANN_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/images/temp256/opt_Air.json"

# If you want to drop tiny fragments of objects, you can use this:
MIN_VISIBLE_FRAC = 0.0  # e.g. 0.3 keeps only tiles with >=30% of the object inside

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

# --- Pre-index annotations by image_id for speed ---
anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

# --- Loop through COCO images ---
for image in tqdm(coco["images"], desc="Tiling images"):
    img_name = os.path.basename(image["file_name"]).lower()
    img_path = img_map.get(img_name)

    if not img_path or not os.path.exists(img_path):
        print(f" Skipping missing image: {img_name}")
        continue

    img = Image.open(img_path)
    w, h = img.size
    anns_for_img = anns_by_image.get(image["id"], [])

    # Slide window over image
    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            x_end = min(x + TILE_SIZE, w)
            y_end = min(y + TILE_SIZE, h)

            # Skip partial tiles (only full tiles)
            if x_end - x < TILE_SIZE or y_end - y < TILE_SIZE:
                continue

            # Crop tile
            tile = img.crop((x, y, x_end, y_end))  # box=(left, upper, right, lower)
            # At this point tile.size should always be (TILE_SIZE, TILE_SIZE),
            # but keep this for robustness if you change the skip logic above.
            if tile.size != (TILE_SIZE, TILE_SIZE):
                tile = tile.resize((TILE_SIZE, TILE_SIZE), resample=Image.BILINEAR)
                scale_x = TILE_SIZE / float(x_end - x)
                scale_y = TILE_SIZE / float(y_end - y)
            else:
                scale_x = 1.0
                scale_y = 1.0

            tile_filename = f"{os.path.splitext(img_name)[0]}_{x}_{y}.png"
            tile_path = os.path.join(OUT_IMAGE_DIR, tile_filename)
            tile.save(tile_path)

            new_images.append({
                "id": img_id_counter,
                "file_name": tile_filename,
                "width": TILE_SIZE,
                "height": TILE_SIZE
            })

            # Process annotations for this tile
            for ann in anns_for_img:
                bx, by, bw, bh = ann["bbox"]  # COCO: [x, y, width, height]

                # Convert to [x1, y1, x2, y2] in original image coords
                bx1 = bx
                by1 = by
                bx2 = bx + bw
                by2 = by + bh

                # Tile box in original coords
                tx1 = x
                ty1 = y
                tx2 = x_end
                ty2 = y_end

                # Compute intersection (in original global coords)
                inter_x1 = max(bx1, tx1)
                inter_y1 = max(by1, ty1)
                inter_x2 = min(bx2, tx2)
                inter_y2 = min(by2, ty2)

                # If no overlap, skip
                if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                    continue

                # Optional: skip if visible fraction too small
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                orig_area = bw * bh
                if orig_area <= 0:
                    continue
                if inter_area / orig_area < MIN_VISIBLE_FRAC:
                    continue

                # Convert intersection to tile-local coords (before any resizing)
                local_x1 = inter_x1 - tx1
                local_y1 = inter_y1 - ty1
                local_w = inter_x2 - inter_x1
                local_h = inter_y2 - inter_y1

                # Apply scaling if we resized the tile
                local_x1 *= scale_x
                local_y1 *= scale_y
                local_w *= scale_x
                local_h *= scale_y

                # Final sanity checks: clip to tile bounds
                if local_w <= 0 or local_h <= 0:
                    continue
                if local_x1 < 0 or local_x1 + local_w > TILE_SIZE:
                    # clamp but keep consistent
                    local_x1 = max(0, min(local_x1, TILE_SIZE - 1))
                    local_w = max(1, min(local_w, TILE_SIZE - local_x1))
                if local_y1 < 0 or local_y1 + local_h > TILE_SIZE:
                    local_y1 = max(0, min(local_y1, TILE_SIZE - 1))
                    local_h = max(1, min(local_h, TILE_SIZE - local_y1))

                new_bbox = [
                    float(local_x1),
                    float(local_y1),
                    float(local_w),
                    float(local_h),
                ]

                new_anns.append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": ann["category_id"],
                    "bbox": new_bbox,
                    "area": new_bbox[2] * new_bbox[3],
                    "iscrowd": ann.get("iscrowd", 0),
                })
                ann_id_counter += 1

            img_id_counter += 1

# --- Save tiled JSON ---
tiled_coco = {
    "images": new_images,
    "annotations": new_anns,
    "categories": coco["categories"],
}
os.makedirs(os.path.dirname(OUT_ANN_PATH), exist_ok=True)
with open(OUT_ANN_PATH, "w") as f:
    json.dump(tiled_coco, f, indent=2)

print(f"Done. Wrote {len(new_images)} tiles and {len(new_anns)} annotations to:")
print(f"  images -> {OUT_IMAGE_DIR}")
print(f"  COCO   -> {OUT_ANN_PATH}")
