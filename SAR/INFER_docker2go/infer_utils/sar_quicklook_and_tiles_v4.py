#!/usr/bin/env python3
import os
import csv
import argparse
from typing import Tuple, List, Optional, Any

import numpy as np
from PIL import Image, ImageEnhance
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

try:
    from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------- Sensor-aware profile ----------------
def choose_profile(src: Any, tif_path: str):
    """Return dict with per-sensor knobs. Single-band => SAR; >=3 => RGB."""
    name = os.path.basename(tif_path).upper()
    if src.count == 1:
        return dict(
            profile="SAR",
            bands=[1],
            p_low=2.0, p_high=98.0,
            use_log_for_stats=True,
            apply_gamma=False, gamma=1.0,
            post_contrast=1.00
        )
    else:
        # RGB-like (SkySat often B,G,R,NIR -> map to [3,2,1])
        bands = [3,2,1] if src.count >= 3 else list(range(1, min(3, src.count)+1))
        return dict(
            profile="RGB",
            bands=bands,
            p_low=0.5, p_high=99.5,
            use_log_for_stats=False,
            apply_gamma=True, gamma=1.6,
            post_contrast=1.05
        )

# ---------------- Core utils ----------------
def percentiles_from_thumbnail(src: Any, bands: List[int], thumb_max: int,
                               p_low: float, p_high: float,
                               use_log_for_stats: bool) -> Tuple[np.ndarray, np.ndarray, bool]:
    width, height = src.width, src.height
    scale = max(width, height) / float(thumb_max)
    if scale < 1.0: scale = 1.0
    out_h = max(1, int(round(height / scale)))
    out_w = max(1, int(round(width / scale)))

    data = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.average)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)
    masks = src.read_masks(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.nearest)
    masks = np.transpose(masks, (1, 2, 0)) > 0
    data[~masks] = np.nan

    is_single = data.shape[2] == 1
    if is_single and use_log_for_stats:
        data = np.log1p(np.maximum(0, data))

    if is_single:
        lo = np.nanpercentile(data[..., 0], p_low)
        hi = np.nanpercentile(data[..., 0], p_high)
        lo = np.array([lo], dtype=np.float32)
        hi = np.array([hi], dtype=np.float32)
    else:
        # JOINT lo/hi → avoids tints
        lo = np.nanpercentile(data, p_low)
        hi = np.nanpercentile(data, p_high)
        lo = np.array([lo], dtype=np.float32).repeat(data.shape[2])
        hi = np.array([hi], dtype=np.float32).repeat(data.shape[2])

    hi = np.maximum(hi, lo + 1e-6)
    return lo, hi, is_single

def scale_to_uint8(data: np.ndarray, lo: np.ndarray, hi: np.ndarray, is_single_band: bool) -> np.ndarray:
    if data.ndim == 2:
        data = data[:, :, None]
    data = data.astype(np.float32)
    if is_single_band and data.shape[2] == 1:
        data = np.log1p(np.maximum(0, data))
    lo_b = lo.reshape((1, 1, -1))
    hi_b = hi.reshape((1, 1, -1))
    out = (data - lo_b) / (hi_b - lo_b)
    out = np.clip(out, 0, 1) * 255.0
    out = out.astype(np.uint8)
    if out.shape[2] == 1:
        return out[:, :, 0]
    return out

def apply_smoothing_uint8(img: np.ndarray, smooth: str) -> np.ndarray:
    if smooth == "none":
        return img
    def _box(img: np.ndarray, k: int) -> np.ndarray:
        pad = k // 2
        if img.ndim == 2:
            imgf = img.astype(np.float32)
            padded = np.pad(imgf, pad, mode="edge")
            out = np.zeros_like(imgf)
            for dy in range(k):
                for dx in range(k):
                    out += padded[dy:dy+img.shape[0], dx:dx+img.shape[1]]
            out /= (k * k)
            return np.clip(out, 0, 255).astype(np.uint8)
        else:
            chs = [_box(img[..., c], k) for c in range(img.shape[2])]
            return np.stack(chs, axis=2)

    if smooth.startswith("mean"):
        k = int(smooth.replace("mean", ""))
        if HAS_SCIPY:
            size = (k, k, 1) if img.ndim == 3 else (k, k)
            return uniform_filter(img, size=size).astype(np.uint8)
        return _box(img, k)
    if smooth.startswith("median"):
        k = int(smooth.replace("median", ""))
        if HAS_SCIPY:
            size = (k, k, 1) if img.ndim == 3 else (k, k)
            return median_filter(img, size=size).astype(np.uint8)
        return _box(img, k)
    if smooth.startswith("gauss"):
        sigma = float(smooth.replace("gauss", ""))
        if HAS_SCIPY:
            sigma_tuple = (sigma, sigma, 0) if img.ndim == 3 else (sigma, sigma)
            return gaussian_filter(img, sigma=sigma_tuple).astype(np.uint8)
        k = max(3, int(round(6 * sigma)) | 1)
        return _box(img, k)
    return img

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

def clamp_roi_to_bounds(x: int, y: int, w: int, h: int, width: int, height: int):
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h

def robust_read(src: Any, bands: List[int], window: Window, on_error: str = "skip", fill_value: float = 0.0):
    try:
        return src.read(bands, window=window)
    except RasterioIOError as e:
        print(f"[WARN] Read failed at window {window}: {e}")
        if on_error == "fill":
            h, w = int(window.height), int(window.width)
            c = len(bands)
            arr = np.full((c, h, w), fill_value, dtype=np.float32)
            return arr
        elif on_error == "skip":
            return None
        else:
            raise

# ---------------- Public API ----------------
def build_quicklook(in_tif: str, out_png: str, bands: Optional[List[int]] = None,
                    thumb_max: int = 2048, smooth: str = "none") -> None:
    with rasterio.open(in_tif) as src:
        print(debug_info(src))
        prof = choose_profile(src, in_tif)
        if bands is None:
            bands = prof["bands"]
        lo, hi, is_single = percentiles_from_thumbnail(
            src, bands, thumb_max=thumb_max,
            p_low=prof["p_low"], p_high=prof["p_high"],
            use_log_for_stats=prof["use_log_for_stats"]
        )
        width, height = src.width, src.height
        scale = max(width, height) / float(thumb_max)
        scale = max(scale, 1.0)
        out_h = max(1, int(round(height / scale)))
        out_w = max(1, int(round(width / scale)))
        data = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.average)
        data = np.transpose(data, (1, 2, 0))
        if data.shape[2] == 1:
            data = data[:, :, 0]
        arr8 = scale_to_uint8(data, lo, hi, is_single_band=is_single)

        # RGB: gamma + tiny contrast (like your v2)
        if prof["apply_gamma"] and arr8.ndim == 3 and arr8.shape[2] == 3:
            GAMMA = prof["gamma"]
            f = (arr8.astype(np.float32) / 255.0) ** (1.0 / GAMMA)
            arr8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)
            if prof["post_contrast"] != 1.0:
                im = Image.fromarray(arr8, "RGB")
                im = ImageEnhance.Contrast(im).enhance(prof["post_contrast"])
                arr8 = np.array(im)

        arr8 = apply_smoothing_uint8(arr8, smooth)
        write_png(out_png, arr8)
        print(f"[OK] Quicklook saved: {out_png}  ({prof['profile']})")

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
            yield Window(x, y, w, h)

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
    smooth: str = "none",
    on_error: str = "skip",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(in_tif))[0]
    with rasterio.open(in_tif) as src:
        print(debug_info(src))
        prof = choose_profile(src, in_tif)
        if bands is None:
            bands = prof["bands"]

        lo, hi, is_single = percentiles_from_thumbnail(
            src, bands, thumb_max=2048,
            p_low=prof["p_low"], p_high=prof["p_high"],
            use_log_for_stats=prof["use_log_for_stats"]
        )

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
            data = robust_read(src, bands, win, on_error=on_error, fill_value=0.0)
            if data is None:
                continue
            data = np.transpose(data, (1, 2, 0))
            if data.shape[2] == 1:
                data = data[:, :, 0]
            arr8 = scale_to_uint8(data, lo, hi, is_single_band=is_single)
            if skip_if_low_variance:
                if float(np.var(arr8)) < var_threshold:
                    continue
            arr8 = apply_smoothing_uint8(arr8, smooth)
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
                raise ValueError("fmt must be 'png' or 'jpg")
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

# ---------------- CLI ----------------
def parse_bands(bands_str: Optional[str]) -> Optional[List[int]]:
    if bands_str is None:
        return None
    toks = [b.strip() for b in bands_str.split(",") if b.strip()]
    return [int(t) for t in toks]

def main():
    ap = argparse.ArgumentParser(description="Large GeoTIFF preview & tiler (robust + sensor-aware).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_prev = sub.add_parser("preview", help="Create a quicklook PNG (and optionally crop an ROI).")
    ap_prev.add_argument("--in_tif", required=True)
    ap_prev.add_argument("--out_png", required=True)
    ap_prev.add_argument("--bands", default=None)
    ap_prev.add_argument("--thumb_max", type=int, default=2048)
    ap_prev.add_argument("--roi", default=None, help="x,y,w,h")
    ap_prev.add_argument("--smooth", default="none", choices=["none","mean3","mean5","mean7","median3","gauss1","gauss2"])
    ap_prev.add_argument("--on_error", default="skip", choices=["skip","fill"], help="How to handle read errors")

    ap_tile = sub.add_parser("tile", help="Cut tiles and write optional CSV manifest.")
    ap_tile.add_argument("--in_tif", required=True)
    ap_tile.add_argument("--out_dir", required=True)
    ap_tile.add_argument("--bands", default=None)
    ap_tile.add_argument("--tile", type=int, default=512)
    ap_tile.add_argument("--stride", type=int, default=None)
    ap_tile.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg"])
    ap_tile.add_argument("--quality", type=int, default=95)
    ap_tile.add_argument("--skip_if_low_variance", action="store_true")
    ap_tile.add_argument("--var_threshold", type=float, default=3.0)
    ap_tile.add_argument("--csv_manifest", default=None)
    ap_tile.add_argument("--smooth", default="none", choices=["none","mean3","mean5","median3","gauss1","gauss2"])
    ap_tile.add_argument("--on_error", default="skip", choices=["skip","fill"], help="How to handle read errors")

    args = ap.parse_args()

    if args.cmd == "preview":
        bands = None if args.bands in (None, "None", "") else [int(v) for v in args.bands.split(",")]
        if args.roi:
            x, y, w, h = (int(v) for v in args.roi.split(","))
            # For brevity, reuse quicklook scaling for ROI via crop + manual steps if needed
            # (You can call crop_roi here if you want full ROI function.)
            build_quicklook(args.in_tif, args.out_png, bands=bands, thumb_max=args.thumb_max, smooth=args.smooth)
        else:
            build_quicklook(args.in_tif, args.out_png, bands=bands, thumb_max=args.thumb_max, smooth=args.smooth)

    elif args.cmd == "tile":
        bands = None if args.bands in (None, "None", "") else [int(v) for v in args.bands.split(",")]
        tile_image(
            args.in_tif, args.out_dir,
            tile=args.tile, stride=args.stride,
            bands=bands, fmt=args.fmt, quality=args.quality,
            skip_if_low_variance=args.skip_if_low_variance, var_threshold=args.var_threshold,
            csv_manifest=args.csv_manifest, smooth=args.smooth, on_error=args.on_error,
        )

if __name__ == "__main__":
    main()
