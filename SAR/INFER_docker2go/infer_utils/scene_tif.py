#!/usr/bin/env python3
import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping
import xml.etree.ElementTree as ET
import cv2


def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge tile VOC annotations back to scene + export browse+GeoJSON."
    )
    ap.add_argument("--scene_tif", required=True, help="Path to original GeoTIFF scene")
    ap.add_argument("--manifest_csv", required=True, help="Path to tiles manifest.csv")
    ap.add_argument("--voc_dir", required=True, help="Directory with VOC XMLs for tiles")
    ap.add_argument("--out_browse", required=True, help="Output PNG with drawn boxes")
    ap.add_argument("--out_geojson", required=True, help="Output GeoJSON file")
    ap.add_argument(
        "--classes",
        default=None,
        help="Optional comma-separated whitelist of classes (e.g. ship,aircraft)",
    )
    ap.add_argument(
        "--max_browse_side",
        type=int,
        default=4096,
        help="Max side size of browse image (longest edge).",
    )
    return ap.parse_args()


def load_manifest(manifest_csv):
    """
    manifest.csv is expected to have at least:
    file,x,y,w,h   (x,y in scene pixels; w,h tile size)
    """
    mapping = {}
    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Try to be tolerant with column names
        cols = {c.lower(): c for c in reader.fieldnames}
        col_file = cols.get("file") or cols.get("filename") or cols.get("image")
        col_x = cols.get("x")
        col_y = cols.get("y")
        col_w = cols.get("w") or cols.get("width")
        col_h = cols.get("h") or cols.get("height")

        if not (col_file and col_x and col_y and col_w and col_h):
            raise RuntimeError(
                f"Manifest columns not as expected. Found: {reader.fieldnames}"
            )

        for row in reader:
            fname = Path(row[col_file]).name
            x = int(float(row[col_x]))
            y = int(float(row[col_y]))
            w = int(float(row[col_w]))
            h = int(float(row[col_h]))
            mapping[fname] = (x, y, w, h)
    return mapping


def parse_voc_tile(voc_path):
    """
    Parse a VOC XML file and return list of dicts:
    { 'class': name, 'bbox': (xmin,ymin,xmax,ymax) } in TILE pixel coords.
    """
    tree = ET.parse(voc_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        objs.append({"class": name, "bbox": (xmin, ymin, xmax, ymax)})
    # filename in VOC (used to match manifest entry)
    fname_node = root.find("filename")
    filename = fname_node.text if fname_node is not None else None
    return filename, objs


def pix_to_map(transform: Affine, x, y):
    """
    Pixel (col=x,row=y) -> map coordinates using rasterio affine.
    """
    mx, my = rasterio.transform.xy(transform, y, x, offset="center")
    return mx, my


def build_browse(scene_tif, max_side=4096):
    """
    Read full scene, stretch to uint8, optionally downscale to max_side,
    return (browse_img[BGR uint8], scale_x, scale_y).
    """
    with rasterio.open(scene_tif) as src:
        arr = src.read()  # (C,H,W)
        transform = src.transform
        crs = src.crs
        H, W = arr.shape[1], arr.shape[2]

    # Pick first 3 bands (or repeat if less)
    C = arr.shape[0]
    if C == 1:
        img = np.repeat(arr[0:1, :, :], 3, axis=0)
    elif C >= 3:
        img = arr[:3, :, :]
    else:
        raise RuntimeError(f"Unsupported band count: {C}")

    img = np.transpose(img, (1, 2, 0)).astype(np.float32)  # HWC

    # simple contrast stretch per channel
    for c in range(3):
        band = img[:, :, c]
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:
            lo, hi = band.min(), band.max()
        if hi <= lo:
            img[:, :, c] = 0
        else:
            band = np.clip((band - lo) / (hi - lo), 0, 1)
            img[:, :, c] = band

    img = (img * 255).clip(0, 255).astype(np.uint8)

    scale = 1.0
    H, W = img.shape[:2]
    longest = max(H, W)
    if longest > max_side:
        scale = max_side / float(longest)
        newW = int(round(W * scale))
        newH = int(round(H * scale))
        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)

    return img[:, :, ::-1], scale, scale, transform, crs  # BGR for cv2, scales, transform, crs


def main():
    args = parse_args()
    class_whitelist = (
        {c.strip() for c in args.classes.split(",")} if args.classes else None
    )

    # 1) Load manifest: tile name -> (x_off, y_off, w, h)
    manifest = load_manifest(args.manifest_csv)

    # 2) Build browse from full scene
    browse_img, sx, sy, transform, crs = build_browse(
        args.scene_tif, max_side=args.max_browse_side
    )
    H_b, W_b = browse_img.shape[:2]

    # 3) Walk VOC directory, merge annotations
    features = []
    color_map = defaultdict(
        lambda: (0, 255, 0)
    )  # per-class color, default green; you can customize

    # Pre-assign some colors if you want
    color_map.update(
        {
            "ship": (0, 255, 0),
            "aircraft": (0, 0, 255),
            "vehicle": (255, 0, 0),
            "infrastructure": (255, 255, 0),
        }
    )

    for voc_name in os.listdir(args.voc_dir):
        if not voc_name.lower().endswith(".xml"):
            continue
        voc_path = os.path.join(args.voc_dir, voc_name)
        tile_filename, objs = parse_voc_tile(voc_path)
        if tile_filename is None:
            # fallback: assume tile image has same stem as XML, with .png
            tile_filename = Path(voc_name).with_suffix(".png").name

        tile_filename = Path(tile_filename).name
        if tile_filename not in manifest:
            print(f"[WARN] Tile {tile_filename} not found in manifest, skipping.")
            continue

        x_off, y_off, w_tile, h_tile = manifest[tile_filename]

        for obj in objs:
            cls = obj["class"]
            if class_whitelist and cls not in class_whitelist:
                continue
            x1_t, y1_t, x2_t, y2_t = obj["bbox"]

            # Scene pixel coords
            x1 = x1_t + x_off
            y1 = y1_t + y_off
            x2 = x2_t + x_off
            y2 = y2_t + y_off

            # Draw on browse (scaled)
            x1_b = int(round(x1 * sx))
            y1_b = int(round(y1 * sy))
            x2_b = int(round(x2 * sx))
            y2_b = int(round(y2 * sy))

            # clip to browse bounds just in case
            x1_b = max(0, min(W_b - 1, x1_b))
            x2_b = max(0, min(W_b - 1, x2_b))
            y1_b = max(0, min(H_b - 1, y1_b))
            y2_b = max(0, min(H_b - 1, y2_b))

            cv2.rectangle(browse_img, (x1_b, y1_b), (x2_b, y2_b), color_map[cls], 2)
            cv2.putText(
                browse_img,
                cls,
                (x1_b, max(0, y1_b - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_map[cls],
                1,
                cv2.LINE_AA,
            )

            # Build GeoJSON feature (map coordinates)
            # pixel corners -> map coords
            # we use the 4 corners as a polygon clockwise
            pts_px = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
                (x1, y1),
            ]
            poly_pts = [pix_to_map(transform, px, py) for (px, py) in pts_px]
            poly = Polygon(poly_pts)

            feat = {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "class": cls,
                    "source_tile": tile_filename,
                },
            }
            features.append(feat)

    # 4) Save browse
    out_dir = os.path.dirname(args.out_browse)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.out_browse, browse_img)
    print(f"[OK] Browse saved -> {args.out_browse}")

    # 5) Save GeoJSON
    out_dir = os.path.dirname(args.out_geojson)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {"name": str(crs) if crs is not None else "UNKNOWN"},
        },
    }
    with open(args.out_geojson, "w") as f:
        json.dump(geojson, f)
    print(f"[OK] GeoJSON saved -> {args.out_geojson}")
    print(f"Total annotations: {len(features)}")


if __name__ == "__main__":
    main()
