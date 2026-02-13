#!/usr/bin/env python3
# voc_tiles_mapper.py
# Map original VOC bboxes (full image coords) to per-tile VOC XMLs using tiler manifest.

import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

# ======= USER PATHS =======
# Ships
TILES_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318/tiles_5120/"          # directory with tile PNG/JPGs
ORIG_XML = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/labels/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318.xml"

# Aircraft
# TILES_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/aircrafts/_outputs_v2/ICEYE_X42_GRD_SLEDP_6090213_20250906T093217/tiles_5120/"          # directory with tile PNG/JPGs
# ORIG_XML = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/aircrafts/labels/ICEYE_X42_GRD_SLEDP_6090213_20250906T093217.xml"
MANIFEST_CSV = os.path.join(TILES_DIR, "manifest.csv")
OUTPUT_VOC_DIR = os.path.join(TILES_DIR, "voc")   # per-tile VOC output
CATEGORY_NAME = "ship"  # set your class name(s) if needed
# CATEGORY_NAME = "aircraft"  # set your class name(s) if needed

# Keep a box only if at least this fraction of the original box remains visible in the tile
MIN_VISIBLE_FRAC = 0.3

os.makedirs(OUTPUT_VOC_DIR, exist_ok=True)

def parse_voc_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # read image size
    size_el = root.find("size")
    W = int(size_el.findtext("width"))
    H = int(size_el.findtext("height"))
    # read all objects
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin")))
        ymin = int(float(bb.findtext("ymin")))
        xmax = int(float(bb.findtext("xmax")))
        ymax = int(float(bb.findtext("ymax")))
        boxes.append((name, xmin, ymin, xmax, ymax))
    # original filename (tif)
    filename = root.findtext("filename")
    return dict(width=W, height=H, filename=filename, boxes=boxes)

def clip_and_shift(box, tile):
    # box: (xmin,ymin,xmax,ymax) in original coords
    # tile: (x0,y0,w,h) in original coords
    _, xmin, ymin, xmax, ymax = ("",) + box  # keep name separately outside
    x0, y0, w, h = tile
    ixmin = max(xmin, x0)
    iymin = max(ymin, y0)
    ixmax = min(xmax, x0 + w)
    iymax = min(ymax, y0 + h)
    if ixmax <= ixmin or iymax <= iymin:
        return None  # no overlap
    # visible area fraction (optional filter)
    orig_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    vis_area = max(0, ixmax - ixmin) * max(0, iymax - iymin)
    if orig_area <= 0 or (vis_area / orig_area) < MIN_VISIBLE_FRAC:
        return None
    # shift to tile coords
    txmin = ixmin - x0
    tymin = iymin - y0
    txmax = ixmax - x0
    tymax = iymax - y0
    return int(txmin), int(tymin), int(txmax), int(tymax)

def write_voc_xml(out_xml, tile_img_name, tile_w, tile_h, objects):
    # objects: list of (name, xmin, ymin, xmax, ymax) in tile coords
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = Path(tile_img_name).parent.name
    ET.SubElement(ann, "filename").text = Path(tile_img_name).name
    ET.SubElement(ann, "path").text = str(Path(tile_img_name).resolve())
    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "Unknown"

    sz = ET.SubElement(ann, "size")
    ET.SubElement(sz, "width").text = str(tile_w)
    ET.SubElement(sz, "height").text = str(tile_h)
    ET.SubElement(sz, "depth").text = "3"

    ET.SubElement(ann, "segmented").text = "0"

    for name, x1, y1, x2, y2 in objects:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(max(0, x1))
        ET.SubElement(bb, "ymin").text = str(max(0, y1))
        ET.SubElement(bb, "xmax").text = str(max(0, x2))
        ET.SubElement(bb, "ymax").text = str(max(0, y2))

    ET.ElementTree(ann).write(out_xml, encoding="utf-8", xml_declaration=True)

def parse_tile_filename(f):
    # expects pattern: base_x{X}_y{Y}_w{W}_h{H}.png
    # robust parse without regex assumptions
    stem = Path(f).stem
    parts = stem.split("_")
    # find tokens x?, y?, w?, h?
    vals = {}
    for p in parts:
        if len(p) > 1 and p[0] in "xywh" and p[1:].isdigit():
            vals[p[0]] = int(p[1:])
    if not all(k in vals for k in ("x","y","w","h")):
        return None
    return vals["x"], vals["y"], vals["w"], vals["h"]

def tile_image_size(tile_path):
    from PIL import Image
    with Image.open(tile_path) as im:
        return im.size  # (W, H)

def main():
    annot = parse_voc_boxes(ORIG_XML)
    base_tif = Path(annot["filename"]).stem  # e.g., ICEYE_X6_GRD_SLED_4410498_20241216T014014
    # read manifest (preferred), else parse from filenames
    rows = []
    if os.path.exists(MANIFEST_CSV):
        with open(MANIFEST_CSV, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    else:
        # fallback: infer x,y,w,h from file names inside TILES_DIR
        for f in os.listdir(TILES_DIR):
            if not f.lower().endswith((".png",".jpg",".jpeg")):
                continue
            if not f.startswith(base_tif + "_"):
                continue
            xywh = parse_tile_filename(f)
            if xywh is None:
                continue
            x, y, w, h = xywh
            rows.append({"file_name": os.path.join(TILES_DIR, f), "x": x, "y": y, "w": w, "h": h})

    # process each tile
    kept = 0
    total = 0
    for row in rows:
        tile_path = row["file_name"] if isinstance(row["file_name"], str) else row["file_name"]
        x0 = int(row["x"]); y0 = int(row["y"]); w = int(row["w"]); h = int(row["h"])
        # Collect objects for this tile
        objs = []
        for (name, xmin, ymin, xmax, ymax) in annot["boxes"]:
            clipped = clip_and_shift((name, xmin, ymin, xmax, ymax), (x0, y0, w, h))
            if clipped is None:
                continue
            tx1, ty1, tx2, ty2 = clipped
            objs.append((name, tx1, ty1, tx2, ty2))
        total += 1
        if not objs:
            continue  # skip empty tiles (no objects)
        tile_w, tile_h = tile_image_size(tile_path)
        out_xml = os.path.join(OUTPUT_VOC_DIR, Path(tile_path).stem + ".xml")
        write_voc_xml(out_xml, tile_path, tile_w, tile_h, objs)
        kept += 1

    print(f"[OK] Wrote {kept} VOC XMLs with objects (from {total} tiles).")
    print(f"[INFO] Output dir: {OUTPUT_VOC_DIR}")

if __name__ == "__main__":
    main()
