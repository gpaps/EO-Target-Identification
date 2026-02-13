from sar_quicklook_and_tiles_v3 import build_quicklook, crop_roi, tile_image

# IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/sar/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318.tif"
# IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/optical/piraeus_skysat_mosaic/raw/skysat_2025_09_05_piraeus.tif"
# IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/sar/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318.tif"
# IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/sar/ICEYE_X47_GRD_SLEDP_6074302_20250905T080916.tif"
# IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/aircrafts/images/ICEYE_X25_GRD_SLEDF_5217751_20250708T001209.tif"
IN_TIF = "/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/-.tif"
# Quicklook
build_quicklook(IN_TIF, out_png="../dataset/ICEYE_X25_GRD_SLEDF_5217751_20250708T001209_16.png", bands=None, thumb_max=4000)
# build_quicklook(IN_TIF, out_png="./skysat_2025_09_05_piraeus.png",
#                 bands=[3,2,1], thumb_max=2048)  # RGB = 3,2,1 for many 4-band stacks

# ROI (x, y, w, h) at native resolution
# crop_roi(IN_TIF, out_png="./roi.png", x=120000, y=80000, w=4096, h=4096, bands=None) #SAR40kx40k
# crop_roi(IN_TIF, "./roi.png", x=6000, y=4500, w=4096, h=4096, bands=[3,2,1]) #RGBskysat

# Tiles
# tile_image(
#     IN_TIF,
#     out_dir="../dataset/ICEYE_X47_GRD_SLEDP_6074302_20250905T080916_tiles_5000",
    # out_dir="/home/gpaps/PycharmProject/Esa_Ships/INFER_docker2go/dataset/skysat_2025_09_05_piraeus/2318_tiles_512",
    # tile=5000,
    # stride=5000,           # same as tile (no overlap). Set e.g. 256 for overlap
    # bands=None,            # auto: 1 for SAR, 1-3 for RGB
    # fmt="png",
    # quality=99,
    # skip_if_low_variance=False,
    # var_threshold=3.0,
    # smooth="mean1",
    # csv_manifest="../dataset/manifest.csv",
# )
print(print("Done."))



