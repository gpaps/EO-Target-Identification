#!/usr/bin/env python3
# img2vis_png.py — Make display-friendly PNG quicklooks (no resize)

import os
import glob
import warnings
import argparse
import numpy as np
from PIL import Image, ImageFile

# Allow huge images; silence Pillow's safety warnings
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

def to_uint8_grayscale(arr: np.ndarray, pclip: float) -> np.ndarray:
    """Percentile clip + min-max normalize to 0..255 (uint8)."""
    arr = arr.astype(np.float32, copy=False)
    hi = np.percentile(arr, pclip)
    lo = np.percentile(arr, 100 - pclip) if pclip > 50 else arr.min()
    arr = np.clip(arr, lo, hi)
    rng = (arr.max() - arr.min()) or 1.0
    arr = (255.0 * (arr - arr.min()) / rng).astype(np.uint8)
    return arr

def vis_from_pil(img: Image.Image, pclip: float, rgb_equalize: bool, band: int) -> Image.Image:
    """Return an 8-bit, viewable PIL.Image with same W×H."""
    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F", "L"):
        # Single-band (typical SAR or grayscale)
        arr = np.array(img)
        vis = to_uint8_grayscale(arr, pclip)
        return Image.fromarray(vis, mode="L")

    if img.mode in ("RGB", "RGBA"):
        arr = np.array(img)
        if rgb_equalize:
            # per-channel stretch
            out = np.empty_like(arr[..., :3])
            for c in range(3):
                out[..., c] = to_uint8_grayscale(arr[..., c], pclip)
            return Image.fromarray(out, mode="RGB")
        else:
            # simple 8-bit copy
            return img.convert("RGB")

    # Multi-band (e.g., >4) or unexpected modes: pick one band for quicklook
    try:
        arr = np.array(img)
        if arr.ndim == 3:
            b = max(0, min(band, arr.shape[2] - 1))
            vis = to_uint8_grayscale(arr[..., b], pclip)
            return Image.fromarray(vis, mode="L")
        else:
            vis = to_uint8_grayscale(arr, pclip)
            return Image.fromarray(vis, mode="L")
    except Exception:
        # Fallback
        return img.convert("L")

def convert_folder(src: str, dst: str, pattern: str, pclip: float, rgb_equalize: bool, band: int):
    os.makedirs(dst, exist_ok=True)
    exts = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(src, pattern, ext)))

    converted = 0
    errors = 0

    for tif_path in files:
        try:
            with Image.open(tif_path) as img:
                vis = vis_from_pil(img, pclip=pclip, rgb_equalize=rgb_equalize, band=band)
                # Keep same basename; write PNG next to dst
                rel = os.path.relpath(tif_path, src)
                out_dir = os.path.join(dst, os.path.dirname(rel))
                os.makedirs(out_dir, exist_ok=True)
                out_png = os.path.join(out_dir, os.path.splitext(os.path.basename(tif_path))[0] + ".png")
                vis.save(out_png, format="PNG", optimize=True)
                print(f"Converted: {tif_path} -> {out_png}")
                converted += 1
        except Exception as e:
            print(f"Error converting {tif_path}: {e}")
            errors += 1

    print("\n[OK] PNG quicklooks done")
    print(f"  Converted: {converted}")
    print(f"  Errors:    {errors}")
    print(f"  Output dir: {dst}")

def main():
    ap = argparse.ArgumentParser(description="TIFF → viewable PNG quicklooks (no resize).")
    ap.add_argument("--src", required=True, help="Source folder with TIFFs")
    ap.add_argument("--dst", required=True, help="Destination folder for PNGs")
    ap.add_argument("--pattern", default="", help="Optional subfolder pattern (e.g., '' or '**' for recursive)")
    ap.add_argument("--pclip", type=float, default=99.5, help="Upper percentile clip (e.g., 99.5)")
    ap.add_argument("--rgb-equalize", action="store_true", help="Per-channel stretch for RGB/RGBA")
    ap.add_argument("--band", type=int, default=0, help="Band index for multi-band (>4) inputs")
    args = ap.parse_args()

    convert_folder(
        src=args.src,
        dst=args.dst,
        pattern=args.pattern,
        pclip=args.pclip,
        rgb_equalize=args.rgb_equalize,
        band=args.band,
    )

if __name__ == "__main__":
    main()
