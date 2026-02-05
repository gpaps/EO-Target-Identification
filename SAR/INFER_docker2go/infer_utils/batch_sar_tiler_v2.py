import os, glob, time
from pathlib import Path
import rasterio

from INFER_docker2go.infer_utils.batch_sar_tiler import INPUT_GLOB, OUT_ROOT

# ==== CONFIG (edit these) ====
# Ship Filepaths
# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/images/*.tif'
# OUT_ROOT   = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2'
# Aircraft Filepaths
# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/infra/*.tif'
# OUT_ROOT   = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/infra/_outputs_v2'
# Skysat Raw Filepaths
# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/optical/piraeus_skysat_mosaic/raw/*.tif'
# OUT_ROOT   = '/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/optical/piraeus_skysat_mosaic/raw/_outputs_640x480/'

# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/pansharpened_thessaloniki.tif'
# OUT_ROOT   = '/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/_outputs_640_640/'

# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/vehicles/SLEDP_6667842_461490/ICEYE_X49_GRD_SLEDP_6667842_20251021T125916.tif'
# OUT_ROOT   = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X49_GRD_SLEDP_6667842_20251021T125916/10048x10048'

# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/images/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218.tif'
# OUT_ROOT   = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218/'

# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/pansharpened.tif'
#THessaloniki OPtical fast
# INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/pansharpened_thessaloniki.tif'
# OUT_ROOT = '/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/Thessaloniki/'
INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/final_inference/Heraklion/Optical/Optical_Heraklion_skysatscene_basic_analytic_udm2_20251222/SkySatScene/20251222_064159_ssc1d1_0010_basic_analytic.tif'
OUT_ROOT = '/media/gpaps/My Passport/CVRL-GeorgeP/_/final_inference/Heraklion/Optical/Optical_Heraklion_skysatscene_basic_analytic_udm2_20251222/patches_SkySatScene/800x800/'

SMOOTH     = 'none'   # 'none','mean3','mean5','median3','gauss1','gauss2'
ON_ERROR   = 'skip'    # 'skip' or 'fill'
THUMB_MAX  = 5048
# If TILE/STRIDE are None -> auto-decide per-image
TILE       = 1000
STRIDE     = 1000
# ============================

# import v3 first (robust), fallback to v2
try:
    from sar_quicklook_and_tiles_v4 import build_quicklook, tile_image, choose_profile
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
        with rasterio.open(tif) as src:
            W, H, C = src.width, src.height, src.count
            prof = choose_profile(src, tif) if choose_profile else None

        # decide tile/stride
        tile = TILE
        stride = STRIDE
        # auto tile for very large or very small images
        if tile is None or stride is None:
            tile, stride = choose_tile_and_stride(W, H)
        # guard: tile cannot exceed image dims
        tile = min(tile, max(W, H))
        stride = min(stride, tile)

        # special-case ICEYE X46/X47 huge frames
        name_uc = base.upper()
        local_on_error = ON_ERROR
        if "ICEYE" in name_uc and ("X46" in name_uc or "X47" in name_uc):
            tile, stride = choose_tile_and_stride(W, H,
                                                  target_tiles_per_side=10,
                                                  min_tile=1536, max_tile=4096,
                                                  align=512, overlap_frac=0.10)
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

        # count actual tiles
        actual = len([p for p in os.listdir(tiles_dir) if p.endswith(".png") and p != "manifest.csv"])
        exp = expected_windows(W, H, stride)

        summary_rows.append((
            tif, W, H, C,
            prof["profile"] if prof else ("1-band" if C==1 else "RGB"),
            tile, stride, exp, actual,
            ql_ok, round(ql_sec,2), round(tile_sec,2),
            local_on_error
        ))

    # write summary
    import csv
    summary_csv = os.path.join(OUT_ROOT, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file","width","height","bands","profile",
            "tile_px","stride_px","expected_tiles","actual_tiles",
            "quicklook_ok","quicklook_sec","tiling_sec","on_error"
        ])
        for row in summary_rows:
            w.writerow(row)

    print(f"[OK] Done in {round(time.time()-t0_all,1)}s. Summary -> {summary_csv}")

if __name__ == "__main__":
    main()
