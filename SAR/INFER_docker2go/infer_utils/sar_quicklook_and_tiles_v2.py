
#!/usr/bin/env python3
import os
import csv
import argparse
from typing import Tuple, List, Optional, Any

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

try:
    from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# def percentiles_from_thumbnail(src: Any, bands: List[int], thumb_max: int = 2048) -> Tuple[np.ndarray, np.ndarray, bool]:
#     width, height = src.width, src.height
#     scale = max(width, height) / float(thumb_max)
#     if scale < 1.0:
#         scale = 1.0
#     out_h = max(1, int(round(height / scale)))
#     out_w = max(1, int(round(width / scale)))
#     data = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.average)
#     data = np.transpose(data, (1, 2, 0)).astype(np.float32)
#     is_single = data.shape[2] == 1
#     if is_single:
#         data = np.log1p(np.maximum(0, data))
#         lo = np.percentile(data, 2)
#         hi = np.percentile(data, 98)
#         lo = np.array([lo], dtype=np.float32)
#         hi = np.array([hi], dtype=np.float32)
#     else:
#         lo = np.percentile(data, 2, axis=(0, 1)).astype(np.float32)
#         hi = np.percentile(data, 98, axis=(0, 1)).astype(np.float32)
#     hi = np.maximum(hi, lo + 1e-6)
#     return lo, hi, is_single
def percentiles_from_thumbnail(src: Any, bands: List[int], thumb_max: int = 2048):
    width, height = src.width, src.height
    scale = max(width, height) / float(thumb_max)
    if scale < 1.0:
        scale = 1.0
    out_h = max(1, int(round(height / scale)))
    out_w = max(1, int(round(width / scale)))

    # downsampled read
    data = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.average)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)  # HWC

    # mask: 0 = invalid. This respects nodata/alpha if present.
    masks = src.read_masks(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.nearest)
    masks = np.transpose(masks, (1, 2, 0)) > 0  # HWC

    is_single = data.shape[2] == 1

    if is_single:
        # SAR log compress before stats
        band = data[:, :, 0]
        band[~masks[:, :, 0]] = np.nan
        band = np.log1p(np.maximum(0, band))
        lo = np.nanpercentile(band, 2)
        hi = np.nanpercentile(band, 98)
        lo = np.array([lo], dtype=np.float32)
        hi = np.array([hi], dtype=np.float32)
    else:
        data[~masks] = np.nan
        lo = np.nanpercentile(data, 1, axis=(0, 1)).astype(np.float32)
        hi = np.nanpercentile(data, 99, axis=(0, 1)).astype(np.float32)

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

def _simple_box_blur_uint8(img: np.ndarray, k: int) -> np.ndarray:
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
        out_ch = []
        for c in range(img.shape[2]):
            out_ch.append(_simple_box_blur_uint8(img[..., c], k))
        return np.stack(out_ch, axis=2)

def apply_smoothing_uint8(img: np.ndarray, smooth: str) -> np.ndarray:
    if smooth == "none":
        return img
    if smooth.startswith("mean"):
        k = int(smooth.replace("mean", ""))
        if HAS_SCIPY:
            size = (k, k, 1) if img.ndim == 3 else (k, k)
            return uniform_filter(img, size=size).astype(np.uint8)
        return _simple_box_blur_uint8(img, k)
    if smooth.startswith("median"):
        k = int(smooth.replace("median", ""))
        if HAS_SCIPY:
            size = (k, k, 1) if img.ndim == 3 else (k, k)
            return median_filter(img, size=size).astype(np.uint8)
        return _simple_box_blur_uint8(img, k)
    if smooth.startswith("gauss"):
        sigma = float(smooth.replace("gauss", ""))
        if HAS_SCIPY:
            sigma_tuple = (sigma, sigma, 0) if img.ndim == 3 else (sigma, sigma)
            return gaussian_filter(img, sigma=sigma_tuple).astype(np.uint8)
        k = max(3, int(round(6 * sigma)) | 1)
        return _simple_box_blur_uint8(img, k)
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

def build_quicklook(in_tif: str, out_png: str, bands: Optional[List[int]] = None, thumb_max: int = 2048, smooth: str = "none") -> None:
    with rasterio.open(in_tif) as src:
        print(debug_info(src))
        if bands is None:
            bands = list(range(1, min(3, src.count) + 1)) if src.count >= 3 else [1]
        lo, hi, is_single = percentiles_from_thumbnail(src, bands, thumb_max=thumb_max)
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

        # brighten RGB a bit
        GAMMA = 1.6  # 1.0 = off, 1.4–1.8 = mild, 2.2 = strong
        if (not is_single) and (arr8.ndim == 3):
            f = (arr8.astype(np.float32) / 255.0) ** (1.0 / GAMMA)
            arr8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)

        arr8 = apply_smoothing_uint8(arr8, smooth)
        write_png(out_png, arr8)
        print(f"[OK] Quicklook saved: {out_png}")

def crop_roi(in_tif: str, out_png: str, x: int, y: int, w: int, h: int, bands: Optional[List[int]] = None, smooth: str = "none") -> None:
    with rasterio.open(in_tif) as src:
        print(debug_info(src))
        if bands is None:
            bands = list(range(1, min(3, src.count) + 1)) if src.count >= 3 else [1]
        x = max(0, min(x, src.width - 1))
        y = max(0, min(y, src.height - 1))
        w = max(1, min(w, src.width - x))
        h = max(1, min(h, src.height - y))
        lo, hi, is_single = percentiles_from_thumbnail(src, bands, thumb_max=2048)
        win = Window(x, y, w, h)
        data = src.read(bands, window=win)
        data = np.transpose(data, (1, 2, 0))
        if data.shape[2] == 1:
            data = data[:, :, 0]
        arr8 = scale_to_uint8(data, lo, hi, is_single_band=is_single)
        # brighten RGB a bit
        GAMMA = 1.6  # 1.0 = off, 1.4–1.8 = mild, 2.2 = strong
        if (not is_single) and (arr8.ndim == 3):
            f = (arr8.astype(np.float32) / 255.0) ** (1.0 / GAMMA)
            arr8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)

        arr8 = apply_smoothing_uint8(arr8, smooth)
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

def parse_bands(bands_str: Optional[str]) -> Optional[List[int]]:
    if bands_str is None:
        return None
    toks = [b.strip() for b in bands_str.split(",") if b.strip()]
    return [int(t) for t in toks]

def main():
    ap = argparse.ArgumentParser(description="Large GeoTIFF preview & tiler (SAR-friendly, optional smoothing).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_prev = sub.add_parser("preview", help="Create a quicklook PNG (and optionally crop an ROI).")
    ap_prev.add_argument("--in_tif", required=True)
    ap_prev.add_argument("--out_png", required=True)
    ap_prev.add_argument("--bands", default=None, help="1-based band indices, e.g. '1,2,3' or '1'")
    ap_prev.add_argument("--thumb_max", type=int, default=2048, help="Longest side in pixels for quicklook")
    ap_prev.add_argument("--roi", default=None, help="ROI crop as x,y,w,h (full-res). If set, only ROI is saved.")
    ap_prev.add_argument("--smooth", default="none", choices=["none","mean3","mean5","mean7","median3","gauss1","gauss2"], help="Optional smoothing for display")
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
    ap_tile.add_argument("--smooth", default="none", choices=["none","mean3","mean5","mean7","median3","gauss1","gauss2"], help="Optional smoothing for display/ML")
    args = ap.parse_args()
    if args.cmd == "preview":
        bands = parse_bands(args.bands)
        if args.roi:
            x, y, w, h = (int(v) for v in args.roi.split(","))
            crop_roi(args.in_tif, args.out_png, x, y, w, h, bands=bands, smooth=args.smooth)
        else:
            build_quicklook(args.in_tif, args.out_png, bands=bands, thumb_max=args.thumb_max, smooth=args.smooth)
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
            smooth=args.smooth,
        )
if __name__ == "__main__":
    main()
