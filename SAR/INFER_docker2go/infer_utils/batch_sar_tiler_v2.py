import os, glob, time, csv
from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image

# ==== CONFIG ====
INPUT_GLOB = '/media/gpaps/My Passport/CVRL-GeorgeP/_/final_inference/Heraklion/Optical/Optical_Heraklion_skysatscene_basic_analytic_udm2_20251222/SkySatScene/*.tif'
OUT_ROOT = '/media/gpaps/My Passport/CVRL-GeorgeP/_/final_inference/Heraklion/Optical/Optical_Heraklion_skysatscene_basic_analytic_udm2_20251222/patches_SkySatScene/'

SMOOTH = 'none'
ON_ERROR = 'skip'
THUMB_MAX = 5048

TILE = 1000
STRIDE = 1000
# ============================

# Try to import QL builder (Optional)
try:
    from sar_quicklook_and_tiles_v4 import build_quicklook, choose_profile
except ImportError:
    try:
        from sar_quicklook_and_tiles_v3 import build_quicklook

        choose_profile = None
    except:
        build_quicklook = None
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


# --- Improved Normalization Logic ---
def normalize_band(band):
    """
    Normalize a band to 0-255 using 2%-98% percentile stretching.
    Ignores 0 values (nodata) so padding doesn't skew the stretch.
    """
    if band.size == 0: return band

    # Create a mask of valid pixels (assuming 0 is nodata)
    valid_mask = band > 0

    # If the tile is purely empty/black, return it as is
    if not np.any(valid_mask):
        return np.zeros_like(band, dtype=np.uint8)

    # Calculate percentiles ONLY on valid pixels
    # This prevents the black background from dragging down the P2 value
    p2, p98 = np.percentile(band[valid_mask], (2, 98))

    if p98 == p2:
        return np.zeros_like(band, dtype=np.uint8)

    # Clip and stretch (convert to float for precision)
    band_float = band.astype(np.float32)
    band_stretched = np.clip((band_float - p2) / (p98 - p2), 0, 1)

    return (band_stretched * 255).astype(np.uint8)


# --- Robust Tiling Function ---
def tile_image_normalized(tif_path, out_dir, tile, stride, manifest_path):
    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height

        # SKIP CHECK: Ignore UDM masks or non-RGB files
        if src.count < 3:
            print(f"[SKIP] {os.path.basename(tif_path)} has {src.count} bands (needs 3+ for RGB).")
            return

        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'col_off', 'row_off', 'width', 'height'])

            for row in range(0, H, stride):
                for col in range(0, W, stride):

                    # 1. Calculate Valid Window Dimensions
                    # We still calculate the "valid" area to pass to Window
                    valid_w = int(min(tile, W - col))
                    valid_h = int(min(tile, H - row))

                    # 2. Strict Integer Window
                    window = Window(col, row, valid_w, valid_h)

                    # 3. Read with boundless=True
                    # This is the FIX for "Bounds and transform inconsistent"
                    # We read bands [3, 2, 1] for SkySat (Red, Green, Blue)
                    try:
                        data = src.read([3, 2, 1], window=window, boundless=True)
                    except Exception as e:
                        print(f"Read error at {col},{row}: {e}")
                        continue

                    # If completely empty (black tile), skip
                    if data.max() == 0:
                        continue

                    # 4. Normalize (R, G, B)
                    r = normalize_band(data[0])
                    g = normalize_band(data[1])
                    b = normalize_band(data[2])
                    rgb = np.dstack((r, g, b))

                    # 5. Pad to full Tile size if we are at the edge
                    # Because we read 'valid_w/h', the array might be smaller than 'tile'
                    h_curr, w_curr = rgb.shape[:2]
                    if h_curr != tile or w_curr != tile:
                        pad_img = np.zeros((tile, tile, 3), dtype=np.uint8)
                        pad_img[:h_curr, :w_curr, :] = rgb
                        rgb = pad_img

                    # Save
                    tile_name = f"tile_{col}_{row}.png"
                    save_path = os.path.join(out_dir, tile_name)
                    Image.fromarray(rgb).save(save_path)

                    writer.writerow([tile_name, col, row, tile, tile])


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

        try:
            with rasterio.open(tif) as src:
                W, H, C = src.width, src.height, src.count
                prof = choose_profile(src, tif) if choose_profile else None
        except Exception as e:
            print(f"[ERROR] Could not open {tif}: {e}")
            continue

        print(f"[INFO] Processing {base}: Size {W}x{H}, bands: {C}")

        # Tile Logic
        tile = TILE
        stride = STRIDE
        if tile is None or stride is None:
            tile, stride = choose_tile_and_stride(W, H)

        # Preserve ICEYE override
        name_uc = base.upper()
        if "ICEYE" in name_uc and ("X46" in name_uc or "X47" in name_uc):
            tile, stride = choose_tile_and_stride(W, H, target_tiles_per_side=10)

        # Quicklook
        ql_path = os.path.join(out_dir, f"{base}_quicklook.png")
        ql_ok = True
        t0 = time.time()
        if build_quicklook and C >= 3:
            try:
                build_quicklook(tif, ql_path, bands=None, thumb_max=THUMB_MAX, smooth=SMOOTH)
                print(f"[OK] Quicklook saved: {ql_path}")
            except Exception as e:
                ql_ok = False
                print(f"[WARN] Quicklook failed for {tif}: {e}")
        else:
            ql_ok = False
        ql_sec = time.time() - t0

        # Tiling
        tiles_dir = os.path.join(out_dir, f"tiles_{tile}")
        ensure_dir(tiles_dir)
        manifest = os.path.join(tiles_dir, "manifest.csv")

        t0 = time.time()
        actual = 0
        try:
            tile_image_normalized(tif, tiles_dir, tile, stride, manifest)
            actual = len([p for p in os.listdir(tiles_dir) if p.endswith(".png") and p != "manifest.csv"])
        except Exception as e:
            print(f"[ERROR] Tiling failed for {tif}: {e}")
        tile_sec = time.time() - t0

        exp = expected_windows(W, H, stride)
        summary_rows.append((
            tif, W, H, C,
            prof["profile"] if prof else ("1-band" if C == 1 else "RGB"),
            tile, stride, exp, actual,
            ql_ok, round(ql_sec, 2), round(tile_sec, 2),
            ON_ERROR
        ))
        print(f"Processed {base}: {actual} tiles generated in {round(tile_sec, 2)}s")

    # Summary CSV
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