import os, glob, time
from pathlib import Path
import rasterio

# ==== CONFIG (edit these) ====
# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/final_inference/Heraklion/Optical/Optical_Heraklion_skysatscene_basic_analytic_udm2_20251222/SkySatScene/*.tif'
# OUT_ROOT = '/media/gpaps/My Passport/CVRL-GeorgeP/_/final_inference/Heraklion/Optical/Optical_Heraklion_skysatscene_basic_analytic_udm2_20251222/patches_SkySatScene/1000x1000/'
INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/received_satellite_data/Optical/Athens-Airport/PelicanScene/*.tif'
OUT_ROOT = '/media/gpaps/My Passport/CVRL-GeorgeP/_/received_satellite_data/Optical/Athens-Airport/PelicanScene/1000x1000/'

SMOOTH = 'none'
ON_ERROR = 'skip'
THUMB_MAX = 5048
TILE = 1000
STRIDE = 1000
# ============================

# Try importing v5 (Haze Fixed), then v4, then v3
try:
    from sar_quicklook_and_tiles_v5 import build_quicklook, tile_image, choose_profile

    print("[INFO] Using sar_quicklook_and_tiles_v5 (Independent Stats / Haze Killer)")
except ImportError:
    try:
        from sar_quicklook_and_tiles_v4 import build_quicklook, tile_image, choose_profile

        print("[INFO] Using sar_quicklook_and_tiles_v4")
    except ImportError:
        from sar_quicklook_and_tiles_v3 import build_quicklook, tile_image

        choose_profile = None


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def expected_windows(width, height, stride):
    import math
    return int(math.ceil(width / float(stride)) * math.ceil(height / float(stride)))


def _round_to(x, base):
    return max(base, int(round(x / base)) * base)


def choose_tile_and_stride(width, height, target_tiles_per_side=8, min_tile=2048, max_tile=5120, align=512,
                           overlap_frac=0.10):
    longest = max(width, height)
    ideal_tile = max(min_tile, min(max_tile, longest // target_tiles_per_side))
    tile = _round_to(ideal_tile, align)
    stride = _round_to(int(tile * (1.0 - overlap_frac)), align)
    return tile, stride


def main():
    tif_paths = sorted(glob.glob(INPUT_GLOB))
    if not tif_paths:
        print(f"[WARN] No files matched: {INPUT_GLOB}")
        return

    ensure_dir(OUT_ROOT)
    summary_rows = []
    t0_all = time.time()

    for tif in tif_paths:
        base = Path(tif).stem
        out_dir = os.path.join(OUT_ROOT, base)
        ensure_dir(out_dir)

        # read dims
        try:
            with rasterio.open(tif) as src:
                W, H, C = src.width, src.height, src.count
                prof = choose_profile(src, tif) if choose_profile else None
        except Exception as e:
            print(f"[ERROR] Could not open {tif}: {e}")
            continue

        # decide tile/stride
        tile = TILE
        stride = STRIDE
        if tile is None or stride is None:
            tile, stride = choose_tile_and_stride(W, H)

        tile = min(tile, max(W, H))
        stride = min(stride, tile)

        # special-case ICEYE X46/X47
        name_uc = base.upper()
        local_on_error = ON_ERROR
        if "ICEYE" in name_uc and ("X46" in name_uc or "X47" in name_uc):
            tile, stride = choose_tile_and_stride(W, H, target_tiles_per_side=10)
            local_on_error = "fill"

        # quicklook
        ql_path = os.path.join(out_dir, f"{base}_quicklook.png")
        ql_ok = True
        t0 = time.time()
        try:
            build_quicklook(tif, ql_path, bands=None, thumb_max=THUMB_MAX, smooth=SMOOTH)
        except Exception as e:
            ql_ok = False
            print(f"[WARN] Quicklook failed for {tif}: {e}")
        ql_sec = time.time() - t0

        # tiles
        tiles_dir = os.path.join(out_dir, f"tiles_{tile}")
        ensure_dir(tiles_dir)
        manifest = os.path.join(tiles_dir, "manifest.csv")
        t0 = time.time()
        try:
            tile_image(
                tif, tiles_dir,
                tile=tile, stride=stride,
                bands=None, fmt='png', quality=99,
                skip_if_low_variance=False, var_threshold=3.0,
                csv_manifest=manifest,
                smooth=SMOOTH, on_error=local_on_error
            )
        except Exception as e:
            print(f"[WARN] Tiling failed for {tif}: {e}")
        tile_sec = time.time() - t0

        actual = len([p for p in os.listdir(tiles_dir) if p.endswith(".png") and p != "manifest.csv"])
        exp = expected_windows(W, H, stride)

        summary_rows.append((
            tif, W, H, C,
            prof["profile"] if prof else ("1-band" if C == 1 else "RGB"),
            tile, stride, exp, actual,
            ql_ok, round(ql_sec, 2), round(tile_sec, 2),
            local_on_error
        ))

    import csv
    summary_csv = os.path.join(OUT_ROOT, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "width", "height", "bands", "profile",
            "tile_px", "stride_px", "expected_tiles", "actual_tiles",
            "quicklook_ok", "quicklook_sec", "tiling_sec", "on_error"
        ])
        for row in summary_rows:
            w.writerow(row)

    print(f"[OK] Done in {round(time.time() - t0_all, 1)}s. Summary -> {summary_csv}")


if __name__ == "__main__":
    main()