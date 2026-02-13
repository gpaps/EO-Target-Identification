#!/usr/bin/env python3
"""
tiles_2_coco.py
Build per-tile COCO JSON from a full-image VOC XML + tiler outputs.

Fixes:
- Filter manifest to this scene only (by filename stem)
- Re-derive x,y,w,h from tile filename (truth)
- Inclusive clipping (w-1/h-1) + clamp to actual tile size
- VOC->COCO uses +1 width/height
- No relative ../ in image file_name
"""

import os, csv, json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

# =========================
# ===== USER SETTINGS =====
# =========================
TILES_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/______/tiles_4096/"
ORIG_VOC_XML = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/labels/.xml"
MANIFEST_CSV = os.path.join(TILES_DIR, "manifest.csv")
OUTPUT_DIR = os.path.join(TILES_DIR, "coco")

# Keep partially visible boxes? 0.0 keeps everything that touches a tile.
MIN_VISIBLE_FRAC = 0.3

# Optional label tweaks
CATEGORY_REMAP = {}  # e.g., {"boat": "ship"}
IGNORE_LABELS = set()  # e.g., {"ignore_me"}


# =========================
# ====== CORE LOGIC =======
# =========================

def ensure_dir(p): os.makedirs(p, exist_ok=True)


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)
    filename = (root.findtext("filename") or "").strip()
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or ""
        if name in IGNORE_LABELS:
            continue
        bb = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin")))
        ymin = int(float(bb.findtext("ymin")))
        xmax = int(float(bb.findtext("xmax")))
        ymax = int(float(bb.findtext("ymax")))
        boxes.append((name, xmin, ymin, xmax, ymax))
    return dict(width=W, height=H, filename=filename, boxes=boxes)


def parse_xywh_from_name(tile_path):
    stem = Path(tile_path).stem  # ..._x{X}_y{Y}_w{W}_h{H}
    vals = {}
    for tok in stem.split("_"):
        if len(tok) > 1 and tok[0] in "xywh":
            try:
                vals[tok[0]] = int(tok[1:])
            except Exception:
                pass
    return (vals.get("x"), vals.get("y"), vals.get("w"), vals.get("h"))


def load_manifest_for_scene(csv_path, tiles_dir, scene_stem):
    """
    Keep only rows whose tile filename starts with scene_stem + '_'.
    Always (re)derive x,y,w,h from the tile filename.
    """
    rows = []

    def add_row(tile_path):
        base = os.path.basename(tile_path)
        if not base.startswith(scene_stem + "_"):
            return
        x, y, w, h = parse_xywh_from_name(tile_path)
        if None in (x, y, w, h):
            return
        rows.append({"file_name": tile_path, "x": x, "y": y, "w": w, "h": h})

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                tile_path = row.get("file_name") or os.path.join(tiles_dir, row.get("filename", ""))
                if not tile_path:
                    continue
                add_row(tile_path)
    else:
        for f in os.listdir(tiles_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                add_row(os.path.join(tiles_dir, f))

    # Stable order (helps debugging)
    rows = sorted(rows, key=lambda r: (r["y"], r["x"]))
    return rows


def coco_init():
    return {
        "info": {"description": "Tiles COCO export", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }


def coco_add_category(cat_name, cat2id, coco_dict):
    if cat_name in cat2id:
        return cat2id[cat_name]
    new_id = 1 + len(cat2id)
    cat2id[cat_name] = new_id
    coco_dict["categories"].append({"id": new_id, "name": cat_name, "supercategory": "object"})
    return new_id


def clip_box_to_tile(box, tile):
    """
    box=(name,xmin,ymin,xmax,ymax) in full-image coords
    tile=(x0,y0,w,h) in full-image coords
    Returns (name, ixmin, iymin, ixmax, iymax) in *full-image* coords (inclusive),
    or None if no overlap or < MIN_VISIBLE_FRAC.
    """
    name, xmin, ymin, xmax, ymax = box
    x0, y0, w, h = tile
    # inclusive tile edges:
    tx_max = x0 + w - 1
    ty_max = y0 + h - 1

    ixmin = max(xmin, x0)
    iymin = max(ymin, y0)
    ixmax = min(xmax, tx_max)
    iymax = min(ymax, ty_max)
    if ixmax < ixmin or iymax < iymin:
        return None

    # visible fraction (inclusive areas)
    orig_area = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    vis_area = max(0, ixmax - ixmin + 1) * max(0, iymax - iymin + 1)
    if orig_area == 0 or (vis_area / orig_area) < MIN_VISIBLE_FRAC:
        return None
    return (name, ixmin, iymin, ixmax, iymax)


def build_coco(tiles_rows, voc_data, out_dir):
    ensure_dir(out_dir)
    coco = coco_init()
    cat2id = {}
    image_id = 1
    ann_id = 1

    # categories from VOC (after optional remap)
    for (name, _, _, _, _) in voc_data["boxes"]:
        mapped = CATEGORY_REMAP.get(name, name)
        if mapped and mapped not in IGNORE_LABELS:
            coco_add_category(mapped, cat2id, coco)

    for row in tiles_rows:
        tile_path = row["file_name"]
        x0, y0, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])

        # actual tile size on disk
        try:
            with Image.open(tile_path) as im:
                tile_w, tile_h = im.size
        except Exception:
            tile_w, tile_h = w, h

        # register image (basename only, no ../)
        coco["images"].append({
            "id": image_id,
            "file_name": os.path.basename(tile_path),
            "width": tile_w,
            "height": tile_h
        })

        # boxes for this tile
        for box in voc_data["boxes"]:
            clipped = clip_box_to_tile(box, (x0, y0, w, h))
            if clipped is None:
                continue
            name, fx1, fy1, fx2, fy2 = clipped  # inclusive, full-image coords

            # shift to tile coords
            tx1 = fx1 - x0
            ty1 = fy1 - y0
            tx2 = fx2 - x0
            ty2 = fy2 - y0

            # clamp to true tile size (inclusive)
            tx1 = max(0, min(tile_w - 1, tx1))
            ty1 = max(0, min(tile_h - 1, ty1))
            tx2 = max(0, min(tile_w - 1, tx2))
            ty2 = max(0, min(tile_h - 1, ty2))
            if tx2 < tx1 or ty2 < ty1:
                continue

            # VOC inclusive -> COCO bbox (x,y,w,h) with +1
            bbox_w = tx2 - tx1 + 1
            bbox_h = ty2 - ty1 + 1
            area = float(bbox_w * bbox_h)

            mapped = CATEGORY_REMAP.get(name, name)
            if not mapped or mapped in IGNORE_LABELS:
                continue
            cat_id = coco_add_category(mapped, cat2id, coco)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [int(tx1), int(ty1), int(bbox_w), int(bbox_h)],
                "area": area,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

        image_id += 1

    out_json = os.path.join(out_dir, "coco_annotationsv4.json")
    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"[OK] COCO saved → {out_json}")
    print(
        f"  images: {len(coco['images'])}, annotations: {len(coco['annotations'])}, categories: {len(coco['categories'])}")
    return out_json


def main():
    ensure_dir(OUTPUT_DIR)
    voc = parse_voc_xml(ORIG_VOC_XML)
    scene_stem = Path(voc["filename"]).stem
    rows = load_manifest_for_scene(MANIFEST_CSV, TILES_DIR, scene_stem)
    if not rows:
        raise RuntimeError("No tiles found for this scene. Check TILES_DIR and scene stem.")
    # quick sanity: count tiles whose actual size != xywh parsed
    mism = 0
    for r in rows:
        try:
            with Image.open(r["file_name"]) as im:
                if im.size != (r["w"], r["h"]):
                    mism += 1
        except Exception:
            pass
    print(f"[INFO] tiles: {len(rows)} | edge-size mismatches (expected on borders): {mism}")
    build_coco(rows, voc, OUTPUT_DIR)


if __name__ == "__main__":
    main()
