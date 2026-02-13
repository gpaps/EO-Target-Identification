import os
import numpy as np
from osgeo import gdal
import rasterio
from rasterio.enums import Resampling

# ---- Paths ----
raw_tif = "20250410_064754_ssc4d3_0009_basic_analytic.tif"
rpc_txt = "20250410_064754_ssc4d3_0009_basic_analytic_RPC.TXT"
rpc_vrt = "temp_rpc.vrt"
output_ortho = "corrected_rgb.tif"

# ---- Attach RPC Metadata and Warp ----
gdal.Translate(rpc_vrt, raw_tif, RPC_FILENAME=rpc_txt)

gdal.Warp(
    output_ortho,
    rpc_vrt,
    format="GTiff",
    resampleAlg="bilinear",
    dstSRS="EPSG:4326",  # or use a projected CRS
    multithread=True,
    options=["RPC_DEM=FALSE"]
)


# ---- Normalize and Save RGB Composite ----
def normalize(band):
    return ((band - band.min()) / (band.max() - band.min() + 1e-8) * 255).astype(np.uint8)


with rasterio.open(output_ortho) as src:
    r = normalize(src.read(3))
    g = normalize(src.read(2))
    b = normalize(src.read(1))

    rgb = np.stack([r, g, b])

    profile = src.profile
    profile.update(count=3, dtype='uint8')

    with rasterio.open("rgb_model_ready.tif", "w", **profile) as dst:
        dst.write(rgb)
