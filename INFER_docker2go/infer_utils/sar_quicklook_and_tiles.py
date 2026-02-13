#!/usr/bin/env python3
import os
import csv
import math
import argparse
from typing import Tuple, List, Optional, Any

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

# -----------------------------
# Utilities
# -----------------------------

def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    raise ValueError(f"Expected uint8, got {arr.dtype}. Use scale_to_uint8 first.")

def percentiles_from_thumbnail(src: Any, bands: List[int], thumb_max: int = 2048) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Compute low/high percentiles from a small downsampled read.
    Returns (lo, hi, is_single_band) where lo/hi are shape (C,).
    For SAR single-band, we compute on log1p of the band.
    """
    width, height = src.width, src.height
    scale = max(width, height) / float(thumb_max)
    if scale < 1.0:
        scale = 1.0
    out_h = max(1, int(round(height / scale)))
    out_w = max(1, int(round(width / scale)))

    # Read downsampled
    data = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.average)
    # C, H, W -> H, W, C
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)

    is_single = data.shape[2] == 1
    if is_single:
        # SAR style visualization: log compress
        data = np.log1p(np.maximum(0, data))
        lo = np.percentile(data, 2)
        hi = np.percentile(data, 98)
        lo = np.array([lo], dtype=np.float32)
        hi = np.array([hi], dtype=np.float32)
    else:
        lo = np.percentile(data, 2, axis=(0, 1))
        hi = np.percentile(data, 98, axis=(0, 1))
        lo = lo.astype(np.float32)
        hi = hi.astype(np.float32)

    # Avoid degenerate ranges
    hi = np.maximum(hi, lo + 1e-6)
    return lo, hi, is_single

def scale_to_uint8(data: np.ndarray, lo: np.ndarray, hi: np.ndarray, is_single_band: bool) -> np.ndarray:
    """
    Scale array to uint8 using per-band lo/hi.
    data: HxW (single band) or HxWxC (multi band) float32
    """
    if data.ndim == 2:
        data = data[:, :, None]
    data = data.astype(np.float32)

    if is_single_band and data.shape[2] == 1:
        data = np.log1p(np.maximum(0, data))

    # Broadcast lo/hi
    lo_b = lo.reshape((1, 1, -1))
    hi_b = hi.reshape((1, 1, -1))

    out = (data - lo_b) / (hi_b - lo_b)
    out = np.clip(out, 0, 1) * 255.0
    out = out.astype(np.uint8)

    if out.shape[2] == 1:
        return out[:, :, 0]
    return out

def write_png(path: str, arr_uint8: np.ndarray) -> None:
    if arr_uint8.ndim == 2:
        img = Image.fromarray(arr_uint8, mode="L")
    elif arr_uint8.ndim == 3 and arr_uint8.shape[2] == 3:
        img = Image.fromarray(arr_uint8, mode="RGB")
    else:
        raise ValueError(f"Unsupported shape for PNG: {arr_uint8.shape}")
    img.save(path)

def debug_info(src: Any) -> str:
    return (f"[INFO] Size: {src.width}x{src.height}, bands: {src.count}, "
            f"dtype: {src.dtypes}, nodata: {src.nodatavals}")

# -----------------------------
# Preview
# -----------------------------

def build_quicklook(in_tif: str, out_png: str, bands: Optional[List[int]] = None, thumb_max: int = 2048) -> None:
    with rasterio.open(in_tif) as src:
        print(debug_info(src))
        if bands is None:
            bands = list(range(1, min(3, src.count) + 1)) if src.count >= 3 else [1]

        lo, hi, is_single = percentiles_from_thumbnail(src, bands, thumb_max=thumb_max)

        # Read downsampled at desired size for output quicklook
        width, height = src.width, src.height
        scale = max(width, height) / float(thumb_max)
        scale = max(scale, 1.0)
        out_h = max(1, int(round(height / scale)))
        out_w = max(1, int(round(width / scale)))
        data = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.average)
        data = np.transpose(data, (1, 2, 0))

        if data.shape[2] == 1:
            data = data[:, :, 0]  # grayscale for saving later
        arr8 = scale_to_uint8(data, lo, hi, is_single_band=is_single)
        write_png(out_png, arr8)
        print(f"[OK] Quicklook saved: {out_png}")

# def crop_roi(in_tif: str, out_png: str, x: int, y: int, w: int, h: int, bands: Optional[List[int]] = None) -> None:
#     with rasterio.open(in_tif) as src:
#         print(debug_info(src))
#         if bands is None:
#             bands = list(range(1, min(3, src.count) + 1)) if src.count >= 3 else [1]
#
#         lo, hi, is_single = percentiles_from_thumbnail(src, bands, thumb_max=2048)
#
#         win = rasterio.windows.Window(x, y, w, h)
#         data = src.read(bands, window=win)
#         data = np.transpose(data, (1, 2, 0))
#
#         if data.shape[2] == 1:
#             data = data[:, :, 0]
#         arr8 = scale_to_uint8(data, lo, hi, is_single_band=is_single)
#         write_png(out_png, arr8)
#         print(f"[OK] ROI saved: {out_png} (x={x}, y={y}, w={w}, h={h})")

# -----------------------------
# Tiling
# -----------------------------
def crop_roi(in_tif: str, out_png: str, x: int, y: int, w: int, h: int, bands: Optional[List[int]] = None) -> None:
    with rasterio.open(in_tif) as src:
        print(debug_info(src))
        if bands is None:
            bands = list(range(1, min(3, src.count) + 1)) if src.count >= 3 else [1]

        # clamp to image bounds
        x = max(0, min(x, src.width - 1))
        y = max(0, min(y, src.height - 1))
        w = max(1, min(w, src.width - x))
        h = max(1, min(h, src.height - y))

        lo, hi, is_single = percentiles_from_thumbnail(src, bands, thumb_max=2048)
        win = rasterio.windows.Window(x, y, w, h)
        data = src.read(bands, window=win)
        data = np.transpose(data, (1, 2, 0))

        if data.shape[2] == 1:
            data = data[:, :, 0]
        arr8 = scale_to_uint8(data, lo, hi, is_single_band=is_single)
        write_png(out_png, arr8)
        print(f"[OK] ROI saved: {out_png} (x={x}, y={y}, w={w}, h={h})")

def iter_windows(width: int, height: int, tile: int, stride: Optional[int] = None):
    if stride is None:
        stride = tile
    xs = list(range(0, width, stride))
    ys = list(range(0, height, stride))
    for y in ys:
        for x in xs:
            w = min(tile, width - x)
            h = min(tile, height - y)
            if w <= 0 or h <= 0:
                continue
            yield rasterio.windows.Window(x, y, w, h)

def tile_image(
    in_tif: str,
    out_dir: str,
    tile: int = 512,
    stride: Optional[int] = None,
    bands: Optional[List[int]] = None,
    fmt: str = "png",
    quality: int = 95,
    skip_if_low_variance: bool = False,
    var_threshold: float = 3.0,
    csv_manifest: Optional[str] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(in_tif))[0]

    with rasterio.open(in_tif) as src:
        print(debug_info(src))
        if bands is None:
            bands = list(range(1, min(3, src.count) + 1)) if src.count >= 3 else [1]

        lo, hi, is_single = percentiles_from_thumbnail(src, bands, thumb_max=2048)

        wins = list(iter_windows(src.width, src.height, tile, stride))
        print(f"[INFO] Tiling with tile={tile}, stride={stride or tile} -> {len(wins)} windows")

        writer = None
        f_csv = None
        if csv_manifest:
            f_csv = open(csv_manifest, "w", newline="")
            writer = csv.writer(f_csv)
            writer.writerow(["file_name", "x", "y", "w", "h"])

        count = 0
        for win in wins:
            data = src.read(bands, window=win)
            data = np.transpose(data, (1, 2, 0))

            if data.shape[2] == 1:
                data = data[:, :, 0]

            arr8 = scale_to_uint8(data, lo, hi, is_single_band=is_single)

            if skip_if_low_variance:
                v = float(np.var(arr8))
                if v < var_threshold:
                    continue

            x, y, w, h = int(win.col_off), int(win.row_off), int(win.width), int(win.height)
            out_name = f"{base}_x{x}_y{y}_w{w}_h{h}.{fmt.lower()}"
            out_path = os.path.join(out_dir, out_name)

            if fmt.lower() in ("jpg", "jpeg"):
                Image.fromarray(arr8 if arr8.ndim == 2 else arr8[:, :, :3]).save(out_path, "JPEG", quality=quality, subsampling=0)
            elif fmt.lower() == "png":
                if arr8.ndim == 3 and arr8.shape[2] > 3:
                    arr8 = arr8[:, :, :3]
                if arr8.ndim == 3 and arr8.shape[2] == 1:
                    arr8 = arr8[:, :, 0]
                write_png(out_path, arr8)
            else:
                raise ValueError("fmt must be 'png' or 'jpg'")

            if writer:
                writer.writerow([out_path, x, y, w, h])

            count += 1
            if count % 200 == 0:
                print(f"[INFO] Saved {count} tiles...")

        if f_csv:
            f_csv.close()
        print(f"[OK] Tiles saved in {out_dir} (total: {count})")
        if csv_manifest:
            print(f"[OK] Manifest: {csv_manifest}")

# -----------------------------
# CLI
# -----------------------------

def parse_bands(bands_str: Optional[str]) -> Optional[List[int]]:
    if bands_str is None:
        return None
    toks = [b.strip() for b in bands_str.split(",") if b.strip()]
    return [int(t) for t in toks]

def main():
    ap = argparse.ArgumentParser(description="Large GeoTIFF preview & tiler (SAR-friendly, no metadata).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # preview
    ap_prev = sub.add_parser("preview", help="Create a quicklook PNG (and optionally crop an ROI).")
    ap_prev.add_argument("--in_tif", required=True)
    ap_prev.add_argument("--out_png", required=True)
    ap_prev.add_argument("--bands", default=None, help="1-based band indices, e.g. '1,2,3' or '1'")
    ap_prev.add_argument("--thumb_max", type=int, default=2048, help="Longest side in pixels for quicklook")
    ap_prev.add_argument("--roi", default=None, help="ROI crop as x,y,w,h (full-res). If set, only ROI is saved.")

    # tile
    ap_tile = sub.add_parser("tile", help="Cut tiles (PNG/JPG) and write optional CSV manifest.")
    ap_tile.add_argument("--in_tif", required=True)
    ap_tile.add_argument("--out_dir", required=True)
    ap_tile.add_argument("--bands", default=None, help="1-based band indices, e.g. '1,2,3' or '1'")
    ap_tile.add_argument("--tile", type=int, default=512)
    ap_tile.add_argument("--stride", type=int, default=None, help="Default=stride=tile (no overlap)")
    ap_tile.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg"])
    ap_tile.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap_tile.add_argument("--skip_if_low_variance", action="store_true", help="Skip tiles with low variance")
    ap_tile.add_argument("--var_threshold", type=float, default=3.0, help="Variance threshold for skipping")
    ap_tile.add_argument("--csv_manifest", default=None, help="Optional path to CSV manifest")

    args = ap.parse_args()

    if args.cmd == "preview":
        bands = parse_bands(args.bands)
        if args.roi:
            x, y, w, h = (int(v) for v in args.roi.split(","))
            crop_roi(args.in_tif, args.out_png, x, y, w, h, bands=bands)
        else:
            build_quicklook(args.in_tif, args.out_png, bands=bands, thumb_max=args.thumb_max)

    elif args.cmd == "tile":
        bands = parse_bands(args.bands)
        tile_image(
            args.in_tif,
            args.out_dir,
            tile=args.tile,
            stride=args.stride,
            bands=bands,
            fmt=args.fmt,
            quality=args.quality,
            skip_if_low_variance=args.skip_if_low_variance,
            var_threshold=args.var_threshold,
            csv_manifest=args.csv_manifest,
        )

if __name__ == "__main__":
    main()
