#!/usr/bin/env python3
"""
copy_images_from_coco_recursive.py

Given a COCO JSON and a root folder with tiles, this script:

- Reads all `file_name` entries from the JSON's "images" section
- Recursively searches for those files under `images_root`
  (e.g. .../_outputs_v2/*/tiles_*/<file_name>.png)
- Copies OR moves the found images into a flat `output_dir`

No new JSON is created. We assume the master JSON already encodes
exactly the images you want (e.g. only annotated tiles).

Example (matches your layout):

python copy_images_from_coco_recursive.py \
  --coco_json "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_json/sar_ships.json" \
  --images_root "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2" \
  --output_dir "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_json/sar_ships_dataset"

Add `--move` if you want to move instead of copy.
"""

import os
import json
import argparse
import shutil


def load_coco(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_image_index(images_root: str):
    """
    Walk `images_root` recursively and build:
        basename -> full_path

    We match using just the basename because COCO `file_name` is
    like "ICEYE_X6_..._x4096_y10240_w2048_h1399.png".
    """
    name_to_path = {}
    duplicates = []

    print(f"[INFO] Scanning for images under: {images_root}")
    for root, dirs, files in os.walk(images_root):
        for fname in files:
            # You can restrict to .png here if you want:
            # if not fname.lower().endswith(".png"):
            #     continue
            full_path = os.path.join(root, fname)
            if fname in name_to_path:
                duplicates.append((fname, name_to_path[fname], full_path))
            else:
                name_to_path[fname] = full_path

    print(f"[INFO] Indexed {len(name_to_path)} unique filenames.")
    if duplicates:
        print(f"[WARN] Found {len(duplicates)} duplicate basenames. Using the first occurrence.")
        # If you want details, uncomment:
        # for fname, p1, p2 in duplicates:
        #     print(f"  {fname}\n    {p1}\n    {p2}")

    return name_to_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy/move all images listed in a COCO JSON by searching recursively under a root folder."
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="Path to master COCO JSON (already merged/filtered).",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Root folder where tiles live (e.g. .../_outputs_v2). "
             "Script will search recursively in all subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination folder where images will be copied/moved.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="If set, move files instead of copying them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    coco = load_coco(args.coco_json)
    images = coco.get("images", [])
    print(f"[INFO] JSON contains {len(images)} images.")

    # Build index from disk: basename -> full path
    name_to_path = build_image_index(args.images_root)

    os.makedirs(args.output_dir, exist_ok=True)

    missing = []
    copied = 0
    op = "move" if args.move else "copy"

    for img in images:
        fname = img["file_name"]          # e.g. "ICEYE_..._tile_xxxxx.png"
        basename = os.path.basename(fname)

        if basename not in name_to_path:
            missing.append(basename)
            print(f"[WARN] Not found on disk for file_name='{fname}' (basename='{basename}')")
            continue

        src = name_to_path[basename]
        dst = os.path.join(args.output_dir, basename)

        if args.move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

        copied += 1

    print(f"[OK] {copied} images {op}d to: {args.output_dir}")
    if missing:
        print(f"[WARN] {len(missing)} images listed in JSON were NOT found on disk.")
        # Uncomment if you want to see all of them:
        # for m in missing:
        #     print("  -", m)


if __name__ == "__main__":
    main()
