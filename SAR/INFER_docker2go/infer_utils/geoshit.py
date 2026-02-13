# convert_geotiff_to_png.py
import argparse, os, numpy as np
import rasterio
from PIL import Image

def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    # percentile stretch to 0-255 per band
    out = np.zeros_like(img, dtype=np.uint8)
    for b in range(img.shape[2]):
        band = img[:, :, b].astype(np.float32)
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:  # fallback
            lo, hi = band.min(), band.max()
        band = np.clip((band - lo) / (hi - lo + 1e-6) * 255, 0, 255)
        out[:, :, b] = band.astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bands", default="1,2,3", help="1-based band indices, e.g. 1,2,3")
    ap.add_argument("--fmt", default="png", choices=["png","jpg"])
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality if fmt=jpg")
    ap.add_argument("--no_stretch", action="store_true", help="Keep original dtype (uint8 only)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    band_idx = [int(x.strip()) for x in args.bands.split(",")]

    for name in os.listdir(args.in_dir):
        if not name.lower().endswith((".tif",".tiff")):
            continue
        in_path = os.path.join(args.in_dir, name)
        out_name = os.path.splitext(name)[0] + f".{args.fmt}"
        out_path = os.path.join(args.out_dir, out_name)

        with rasterio.open(in_path) as src:
            arr = src.read(band_idx)  # C,H,W
            arr = np.transpose(arr, (1, 2, 0))  # H,W,C

        if not args.no_stretch:
            arr = to_uint8(arr)
        else:
            if arr.dtype != np.uint8:
                raise ValueError("no_stretch requires input to be uint8.")

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]  # (H,W,1) -> (H,W)

        img = Image.fromarray(arr)
        if args.fmt == "jpg":
            img.save(out_path, "JPEG", quality=args.quality, subsampling=0)
        else:
            img.save(out_path, "PNG")
        print(f"[OK] {in_path} -> {out_path}")

if __name__ == "__main__":
    main()
