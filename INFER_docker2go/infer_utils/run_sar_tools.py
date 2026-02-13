from sar_quicklook_and_tiles_v2 import build_quicklook, crop_roi, tile_image

# IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/sar/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318.tif"
# IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/optical/piraeus_skysat_mosaic/raw/skysat_2025_09_05_piraeus.tif"
IN_TIFF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/pansharpened/pansharpened.tif"
# Quicklook
# build_quicklook(IN_TIF, out_png="./pansharpened_quicklook.png", bands=None, thumb_max=4048)
build_quicklook(IN_TIF, out_png="./pansharpened_quicklook",
                bands=[3,2,1], thumb_max=4048)  # RGB = 3,2,1 for many 4-band stacks

# ROI (x, y, w, h) at native resolution
# crop_roi(IN_TIF, out_png="./roi.png", x=120000, y=80000, w=4096, h=4096, bands=None) #SAR40kx40k
# crop_roi(IN_TIF, "./roi.png", x=6000, y=4500, w=4096, h=4096, bands=[2,1,3]) #RGBskysat

# Tiles
tile_image(
    IN_TIF,
    out_dir="./ICEYE_X55_GRD_SLEDP_6067614_20250904T232318_tiles_5000a",
    tile=5000,
    stride=5000,           # same as tile (no overlap). Set e.g. 256 for overlap
    bands=None,            # auto: 1 for SAR, 1-3 for RGB
    fmt="png",
    quality=99,
    skip_if_low_variance=False,
    var_threshold=3.0,
    smooth="mean3",
    csv_manifest="./skysat_2025_09_05_piraeus_manifest.csv",
)

# pansharpen
tile_image(
    IN_TIF,
    out_dir="./tiles_1024_s512",
    tile=1024,
    stride=512,
    bands=[3,2,1],
    fmt="png",
    quality=95,
    skip_if_low_variance=True,
    var_threshold=3.0,
    csv_manifest="./tiles_manifest.csv",
    smooth="none"
)

print("Done.")