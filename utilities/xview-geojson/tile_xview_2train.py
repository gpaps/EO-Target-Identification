import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= CONFIGURATION =================
# 1. INPUTS (Where your xView data is NOW)
INPUT_IMG_DIR = '/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Multiclass_dataset/xView[Annot-Yes][Extract_Geojson]/JPG/'
INPUT_LABEL_DIR = './xview_vessels_labels'  # Output from your geojson2bb script

# 2. OUTPUTS (Where the tiles will go)
OUTPUT_IMG_DIR = '/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/Xview/images'
OUTPUT_LABEL_DIR = '/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/Xview/labels'

# 3. TILING SETTINGS (Mimicking your Skysat Data)
TILE_SIZE = 1024  # Target size (matches your Skysat 1024x1024)
STRIDE = 800  # 1024 - 800 = 224px overlap (Prevents cutting ships at edges)
IOU_THRESH = 0.4  # If a ship is cut, keep it if >40% is visible
KEEP_EMPTY_TILES = False  # Set False to ignore tiles with no ships (Recommended)


# =================================================

def compute_iou(boxA, boxB):
    # Determine the (x, y) - coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    # boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) # Not needed for this specific logic

    # Compute intersection over boxA area (How much of the object is inside the tile?)
    iou = interArea / float(boxAArea + 1e-6)
    return iou


def yolo_to_pixel(x_c, y_c, w, h, img_w, img_h):
    x_min = int((x_c - w / 2) * img_w)
    y_min = int((y_c - h / 2) * img_h)
    x_max = int((x_c + w / 2) * img_w)
    y_max = int((y_c + h / 2) * img_h)
    return [x_min, y_min, x_max, y_max]


def pixel_to_yolo(xmin, ymin, xmax, ymax, tile_w, tile_h):
    x_c = ((xmin + xmax) / 2) / tile_w
    y_c = ((ymin + ymax) / 2) / tile_h
    w = (xmax - xmin) / tile_w
    h = (ymax - ymin) / tile_h
    return x_c, y_c, w, h


def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    # Get list of labeled files only
    txt_files = [f for f in os.listdir(INPUT_LABEL_DIR) if f.endswith(".txt")]

    print(f"🚀 Starting Tiling Process on {len(txt_files)} large scenes...")

    total_tiles_saved = 0

    for txt_file in tqdm(txt_files):
        base_name = os.path.splitext(txt_file)[0]

        # Find corresponding image
        img_path = os.path.join(INPUT_IMG_DIR, base_name + ".tif")
        if not os.path.exists(img_path):
            img_path = os.path.join(INPUT_IMG_DIR, base_name + ".jpg")
            if not os.path.exists(img_path):
                continue

        # Load Image
        # Using cv2 for speed, careful with huge TIFs
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        # Load Labels
        boxes = []
        with open(os.path.join(INPUT_LABEL_DIR, txt_file), 'r') as f:
            for line in f:
                c, x, y, w, h = map(float, line.strip().split())
                pixel_box = yolo_to_pixel(x, y, w, h, img_w, img_h)
                boxes.append({'class': int(c), 'box': pixel_box})

        # Slide Window
        for y in range(0, img_h, STRIDE):
            for x in range(0, img_w, STRIDE):
                # Define Tile Geometry
                x_end = min(x + TILE_SIZE, img_w)
                y_end = min(y + TILE_SIZE, img_h)

                # Correct start if we hit the edge (to ensure strict 1024 size)
                x_start = x
                y_start = y
                if x_end - x_start < TILE_SIZE:
                    x_start = max(0, img_w - TILE_SIZE)
                if y_end - y_start < TILE_SIZE:
                    y_start = max(0, img_h - TILE_SIZE)

                # Crop Image
                tile_img = img[y_start:y_end, x_start:x_end]
                tile_h, tile_w = tile_img.shape[:2]

                # Process Labels for this Tile
                tile_labels = []
                tile_box_coords = [x_start, y_start, x_start + tile_w, y_start + tile_h]

                for obj in boxes:
                    cls_id = obj['class']
                    ox_min, oy_min, ox_max, oy_max = obj['box']

                    # Check Intersection
                    # Shift coordinates relative to tile
                    tx_min = max(0, ox_min - x_start)
                    ty_min = max(0, oy_min - y_start)
                    tx_max = min(tile_w, ox_max - x_start)
                    ty_max = min(tile_h, oy_max - y_start)

                    # Valid Box?
                    if tx_max > tx_min and ty_max > ty_min:
                        # Check how much of the original ship is in this tile
                        overlap_ratio = compute_iou([ox_min, oy_min, ox_max, oy_max],
                                                    [x_start, y_start, x_start + tile_w, y_start + tile_h])

                        if overlap_ratio > IOU_THRESH:
                            # Convert back to YOLO format for the TILE
                            ny_xc, ny_yc, ny_w, ny_h = pixel_to_yolo(tx_min, ty_min, tx_max, ty_max, tile_w, tile_h)
                            tile_labels.append(f"{cls_id} {ny_xc:.6f} {ny_yc:.6f} {ny_w:.6f} {ny_h:.6f}")

                # Save if useful
                if tile_labels or KEEP_EMPTY_TILES:
                    tile_name = f"{base_name}_{x_start}_{y_start}"

                    # Save Label
                    with open(os.path.join(OUTPUT_LABEL_DIR, tile_name + ".txt"), 'w') as f:
                        f.write("\n".join(tile_labels))

                    # Save Image
                    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, tile_name + ".jpg"), tile_img)
                    total_tiles_saved += 1

    print(f"\n✅ Tiling Complete!")
    print(f"   Created {total_tiles_saved} tiles matching your dataset specs (1024x1024).")
    print(f"   Output: {OUTPUT_IMG_DIR}")


if __name__ == "__main__":
    main()