#!/usr/bin/env python3
import os, time
import rasterio
from pathlib import Path
from sar_quicklook_and_tiles_v4 import build_quicklook, tile_image, choose_profile

# -------------------------------------------------
#  INPUT PATH (edit only this)
# -------------------------------------------------
IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/images/ICEYE_X47_GRD_SLEDP_6077248_20250905T131046.tif"
OUT_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/"
# OUT_ROOT = "../dataset"
SMOOTH = "mean3"
ON_ERROR = "skip"       # "skip" or "fill"
THUMB_MAX = 4096
# leave None for auto-tiling
TILE = None
STRIDE = None
# -------------------------------------------------

def _round_to(x, base):
    return max(base, int(round(x / base)) * base)

def choose_tile_and_stride(width, height,
                           target_tiles_per_side=8,
                           min_tile=2048, max_tile=5120,
                           align=512, overlap_frac=0.10):
    longest = max(width, height)
    ideal_tile = max(min_tile, min(max_tile, longest // target_tiles_per_side))
    tile = _round_to(ideal_tile, align)
    stride = _round_to(int(tile * (1.0 - overlap_frac)), align)
    return tile, stride

def main():
    t0 = time.time()
    base = Path(IN_TIF).stem
    out_dir = os.path.join(OUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)

    # --- open image, get profile + dims ---
    with rasterio.open(IN_TIF) as src:
        W, H, C = src.width, src.height, src.count
        prof = choose_profile(src, IN_TIF)
    print(f"[INFO] Opened {IN_TIF}")
    print(f"[INFO] Size={W}x{H}, bands={C}, profile={prof['profile']}")

    # --- decide tile/stride ---
    tile = TILE
    stride = STRIDE
    if tile is None or stride is None:
        tile, stride = choose_tile_and_stride(W, H)
    tile = min(tile, max(W, H))
    stride = min(stride, tile)

    # special ICEYE huge frames
    name_uc = base.upper()
    local_on_error = ON_ERROR
    if "ICEYE" in name_uc and ("X46" in name_uc or "X47" in name_uc):
        tile, stride = choose_tile_and_stride(W, H,
                                              target_tiles_per_side=10,
                                              min_tile=1536, max_tile=4096,
                                              align=512, overlap_frac=0.10)
        local_on_error = "fill"

    # --- quicklook ---
    ql_path = os.path.join(out_dir, f"{base}_quicklook.png")
    try:
        build_quicklook(IN_TIF, out_png=ql_path,
                        bands=None, thumb_max=THUMB_MAX, smooth=SMOOTH)
        print(f"[OK] Quicklook saved → {ql_path}")
    except Exception as e:
        print(f"[WARN] Quicklook failed: {e}")

    # --- tiling ---
    tiles_dir = os.path.join(out_dir, f"tiles_{tile}")
    os.makedirs(tiles_dir, exist_ok=True)
    manifest = os.path.join(tiles_dir, "manifest.csv")
    try:
        tile_image(
            IN_TIF, tiles_dir,
            tile=tile, stride=stride,
            bands=None, fmt="png", quality=99,
            skip_if_low_variance=False, var_threshold=3.0,
            csv_manifest=manifest,
            smooth=SMOOTH, on_error=local_on_error
        )
        print(f"[OK] Tiling done → {tiles_dir}")
    except Exception as e:
        print(f"[WARN] Tiling failed: {e}")

    print(f"[DONE] Total elapsed {round(time.time()-t0,1)}s")

if __name__ == "__main__":
    main()
