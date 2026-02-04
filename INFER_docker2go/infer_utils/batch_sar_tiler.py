
import os, math, glob
from pathlib import Path

# ==== CONFIG (edit these) ====
INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/sar/*.tif'  # supports spaces; keep quotes
OUT_ROOT   = './sar_outputs'    # where to write per-image folders
TILE       = 1000               # tile size (e.g., 800 or 1000)
STRIDE     = 1000               # stride (same as TILE for no overlap)
SMOOTH     = 'none'             # 'none','mean3','mean5','median3','gauss1','gauss2' (display only)
ON_ERROR   = 'skip'             # 'skip' or 'fill' tiles that fail to read
THUMB_MAX  = 2048               # quicklook longest side in pixels
# ============================

# import v3 first (robust), fallback to v2
try:
    from sar_quicklook_and_tiles_v3 import build_quicklook, tile_image
except ImportError:
    from sar_quicklook_and_tiles_v2 import build_quicklook, tile_image

import rasterio

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def expected_windows(width, height, stride):
    import math
    return int(math.ceil(width / float(stride)) * math.ceil(height / float(stride)))

def main():
    tif_paths = sorted(glob.glob(INPUT_GLOB))
    if not tif_paths:
        print(f"[WARN] No files matched: {INPUT_GLOB}")
        return

    ensure_dir(OUT_ROOT)
    summary_rows = []
    for tif in tif_paths:
        base = Path(tif).stem
        out_dir = os.path.join(OUT_ROOT, base)
        tiles_dir = os.path.join(out_dir, f"tiles_{TILE}")
        ensure_dir(out_dir)
        ensure_dir(tiles_dir)

        # read dims
        with rasterio.open(tif) as src:
            W, H, C = src.width, src.height, src.count

        # quicklook
        ql_path = os.path.join(out_dir, f"{base}_quicklook.png")
        try:
            build_quicklook(tif, ql_path, bands=None, thumb_max=THUMB_MAX, smooth=SMOOTH)
        except Exception as e:
            print(f"[WARN] Quicklook failed for {tif}: {e}")

        # tiles
        manifest = os.path.join(tiles_dir, "manifest.csv")
        try:
            tile_image(
                tif, tiles_dir,
                tile=TILE, stride=STRIDE,
                bands=None, fmt='png', quality=95,
                skip_if_low_variance=False, var_threshold=3.0,
                csv_manifest=manifest,
                smooth=SMOOTH, on_error=ON_ERROR
            )
        finally:
            # count actual tiles
            actual = len([p for p in os.listdir(tiles_dir) if p.endswith(".png") and p != "manifest.csv"])
            exp = expected_windows(W, H, STRIDE)
            summary_rows.append((tif, W, H, TILE, STRIDE, exp, actual))

    # write summary
    import csv
    summary_csv = os.path.join(OUT_ROOT, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "width", "height", "tile", "stride", "expected_tiles", "actual_tiles"])
        for row in summary_rows:
            w.writerow(row)

    print(f"[OK] Done. Summary -> {summary_csv}")

if __name__ == "__main__":
    main()
