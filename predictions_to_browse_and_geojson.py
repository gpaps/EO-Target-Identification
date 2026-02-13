#!/usr/bin/env python3
import os
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
import cv2

from post.sanity_script import scene_tif


def load_manifest(manifest_csv):
    """
    manifest.csv must have at least:
    file_name,x,y,w,h   (x,y = offsets in scene pixels; w,h = tile size)
    Returns: dict[basename] -> (x_off, y_off, w, h)
    """
    mapping = {}
    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames}
        col_file = (
                cols.get("file_name")
                or cols.get("file")
                or cols.get("filename")
                or cols.get("image")
        )
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
            x = float(row[col_x])
            y = float(row[col_y])
            w = float(row[col_w])
            h = float(row[col_h])
            mapping[fname] = (x, y, w, h)
    return mapping


def pix_to_map(transform: Affine, x, y):
    """Pixel (col=x,row=y) -> map coords using rasterio affine."""
    mx, my = rasterio.transform.xy(transform, y, x, offset="center")
    return mx, my


def build_browse(scene_tif, max_side=4096):
    """
    Read a *downsampled* version of the scene directly from disk,
    stretch to uint8, and return a BGR browse image.

    Returns:
      browse_img[BGR uint8, contiguous],
      scale_x, scale_y (pixel -> browse),
      transform (original, not scaled),
      crs
    """
    with rasterio.open(scene_tif) as src:
        transform = src.transform
        crs = src.crs
        H = src.height
        W = src.width
        C = src.count

        # Compute downsampled size
        longest = max(H, W)
        if longest > max_side:
            scale = max_side / float(longest)
            newH = int(round(H * scale))
            newW = int(round(W * scale))
        else:
            scale = 1.0
            newH, newW = H, W

        # Choose bands: 1 band -> repeat, >=3 use first 3
        if C == 1:
            indexes = [1]
        elif C >= 3:
            indexes = [1, 2, 3]
        else:
            raise RuntimeError(f"Unsupported band count: {C}")

        # Read already-downsampled data
        arr = src.read(
            indexes=indexes,
            out_shape=(len(indexes), newH, newW),
            resampling=Resampling.bilinear,
        )

    # If single-band, repeat to 3 channels
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)

    # (C,H,W) -> (H,W,C)
    img = np.transpose(arr, (1, 2, 0)).astype(np.float32)

    # Percentile stretch per channel
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

    # RGB -> BGR for OpenCV, make contiguous
    img_bgr = img[:, :, ::-1]
    img_bgr = np.ascontiguousarray(img_bgr)

    # scale is how to go from original pixel coords -> browse coords
    # x_browse = x_orig * scale, y_browse = y_orig * scale
    return img_bgr, scale, scale, transform, crs


def load_predictions(pred_csv, score_thresh=0.0, class_whitelist=None):
    """
    Reads prediction_log.csv and returns list of:
    (tile_name, cls_name, score, x1, y1, x2, y2)
    """
    detections = []
    with open(pred_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames}

        col_image = (
                cols.get("image")
                or cols.get("file")
                or cols.get("filename")
                or cols.get("tile")
        )
        col_cls = cols.get("cls_name") or cols.get("class") or cols.get("category")
        col_score = cols.get("score") or cols.get("conf") or cols.get("confidence")
        col_x1 = (
                cols.get("x1")
                or cols.get("bbox_x1")
                or cols.get("xmin")
                or cols.get("left")
        )
        col_y1 = (
                cols.get("y1")
                or cols.get("bbox_y1")
                or cols.get("ymin")
                or cols.get("top")
        )
        col_x2 = (
                cols.get("x2")
                or cols.get("bbox_x2")
                or cols.get("xmax")
                or cols.get("right")
        )
        col_y2 = (
                cols.get("y2")
                or cols.get("bbox_y2")
                or cols.get("ymax")
                or cols.get("bottom")
        )

        required = [col_image, col_cls, col_score, col_x1, col_y1, col_x2, col_y2]
        if not all(required):
            raise RuntimeError(
                f"Prediction CSV columns not as expected. Found: {reader.fieldnames}"
            )

        for row in reader:
            tile_name = Path(row[col_image]).name
            cls_name = str(row[col_cls])
            score = float(row[col_score])
            if score < score_thresh:
                continue
            if class_whitelist and cls_name not in class_whitelist:
                continue
            x1 = float(row[col_x1])
            y1 = float(row[col_y1])
            x2 = float(row[col_x2])
            y2 = float(row[col_y2])

            detections.append((tile_name, cls_name, score, x1, y1, x2, y2))

    return detections


def predictions_to_browse_and_geojson(
        scene_tif,
        manifest_csv,
        pred_csv,
        out_browse,
        out_geojson,
        score_thresh=0.5,
        classes=None,
        max_browse_side=4096,
):
    """
    Read scene + manifest + prediction_log.csv,
    draw detections on a browse image, export GeoJSON in map coords.
    """
    class_whitelist = {c.strip() for c in classes.split(",")} if classes else None

    # 1) Manifest: tile basename -> (x_off, y_off, w, h)
    manifest = load_manifest(manifest_csv)

    # 2) Browse image from (downsampled) scene
    browse_img, sx, sy, transform, crs = build_browse(
        scene_tif, max_side=max_browse_side
    )
    browse_img = np.ascontiguousarray(browse_img)
    H_b, W_b = browse_img.shape[:2]

    # 3) Predictions
    detections = load_predictions(
        pred_csv,
        score_thresh=score_thresh,
        class_whitelist=class_whitelist,
    )

    # 4) Draw + build GeoJSON
    features = []
    color_map = defaultdict(lambda: (0, 255, 0))  # default green
    color_map.update({
        "Car": (255, 0, 0),  # blue
        "Truck": (0, 165, 255),  # orange
        "Bus": (0, 0, 255),  # red
    })

    # color_map.update(
    #     {
    #         "Commercial": (0, 255, 0),
    #         "Recreational Boats": (0, 0, 255),
    #         "Fishing": (255, 0, 0),
    #         "Military": (255, 255, 0),
    #     }
    # )

    for tile_name, cls_name, score, x1_t, y1_t, x2_t, y2_t in detections:
        if tile_name not in manifest:
            print(f"[WARN] Tile {tile_name} not found in manifest, skipping.")
            continue

        x_off, y_off, _, _ = manifest[tile_name]

        # Scene pixel coords
        x1 = x1_t + x_off
        y1 = y1_t + y_off
        x2 = x2_t + x_off
        y2 = y2_t + y_off

        # Browse coords (scaled)
        x1_b = int(round(x1 * sx))
        y1_b = int(round(y1 * sy))
        x2_b = int(round(x2 * sx))
        y2_b = int(round(y2 * sy))

        x1_b = max(0, min(W_b - 1, x1_b))
        x2_b = max(0, min(W_b - 1, x2_b))
        y1_b = max(0, min(H_b - 1, y1_b))
        y2_b = max(0, min(H_b - 1, y2_b))

        color = color_map[cls_name]
        cv2.rectangle(browse_img, (x1_b, y1_b), (x2_b, y2_b), color, 2)
        cv2.putText(
            browse_img,
            f"{cls_name[:10]}",
            (x1_b, max(0, y1_b - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

        # GeoJSON polygon in map coords (original transform)
        px_corners = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
            (x1, y1),
        ]
        coords = [list(pix_to_map(transform, px, py)) for (px, py) in px_corners]

        feat = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "class": cls_name,
                "score": score,
                "source_tile": tile_name,
            },
        }
        features.append(feat)

    # 5) Save browse
    out_dir = os.path.dirname(out_browse)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # cv2.imwrite(out_browse, browse_img) (BGR)
    cv2.imwrite(out_browse, cv2.cvtColor(browse_img, cv2.COLOR_BGR2RGB))

    # 6) Save GeoJSON
    out_dir = os.path.dirname(out_geojson)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {
                "name": str(crs) if crs is not None else "UNKNOWN",
            },
        },
    }
    with open(out_geojson, "w") as f:
        json.dump(geojson, f)

    print(f"[OK] Browse saved -> {out_browse}")
    print(f"[OK] GeoJSON saved -> {out_geojson}")
    print(f"Total detections exported: {len(features)}")


if __name__ == "__main__":
    predictions_to_browse_and_geojson(
        scene_tif="/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/pansharpened_thessaloniki.tif",
        # scene_tif="/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/pansharpened.tif",
        # scene_tif= "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/vehicles/SLEDP_6667842_461490/ICEYE_X49_GRD_SLEDP_6667842_20251021T125916.tif",
        manifest_csv= "/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/Thessaloniki/tiles_640x512/manifest.csv",
        # manifest_csv="/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/Peiraeus640x512/pansharpened/tiles_640/manifest.csv",

        pred_csv=   "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/3cl/xview_res50_640/Optical_Final_R50_ROI1024_Fraction25_Solar1.5_30k/output_thess640x512/Optical_Final_R50_ROI1024_Fraction25_Solar1.5_30k_s0.7_n0.3_imgs/prediction_log.csv",
        out_browse= "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/3cl/xview_res50_640/Optical_Final_R50_ROI1024_Fraction25_Solar1.5_30k/output_thess640x512/Optical_Final_R50_ROI1024_Fraction25_Solar1.5_30k_s0.7_n0.3_imgs/THess_3clVeh_predictions_browse.png",
        out_geojson="/media/gpaps/My Passport/CqqqqqqqqVRL-GeorgeP/Trained_models/VehiOpt/3cl/xview_res50_640/Optical_Final_R50_ROI1024_Fraction25_Solar1.5_30k/output_thess640x512/Optical_Final_R50_ROI1024_Fraction25_Solar1.5_30k_s0.7_n0.3_imgs/Thess_3clVeh_predictions.geojson",
        score_thresh=0.7,
        classes="Car, Truck, Bus", # "aircraft, helicopter",#"Commercial,Military,Submarines,Recreational Boats,Fishing Boats",
        # e.g. "Commercial,Recreational Boats"
        max_browse_side=6096,
    )
    # predictions_to_browse_and_geojson(
    #     scene_tif="/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/images/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218.tif",
    #     manifest_csv="/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218/tiles_3072/manifest.csv",
    #     pred_csv="/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december/finetune_base_v5_lr0.0002_b512_v21/output_2/finetune_base_v5_lr0.0002_b512_v21_s0.6_n0.3/prediction_log.csv",
    #     out_browse="/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december/finetune_base_v5_lr0.0002_b512_v21/output_2/finetune_base_v5_lr0.0002_b512_v21_s0.7_n0.3/predictionsx25_browse.png",
    #     out_geojson="/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december/finetune_base_v5_lr0.0002_b512_v21/output_2/finetune_base_v5_lr0.0002_b512_v21_s0.7_n0.3/x25_predictions.geojson",
    #     score_thresh=0.5,
    #     classes= "ship",   #"Commercial,Military,Submarines,Recreational Boats,Fishing Boats",
    #     # e.g. "Commercial,Recreational Boats"
    #     max_browse_side=2096,
    # )

