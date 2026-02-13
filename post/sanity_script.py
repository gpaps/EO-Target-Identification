import rasterio, cv2, json
import numpy as np
from shapely.geometry import shape

scene_tif = "/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/sar/ICEYE_X47_GRD_SLEDP_6074302_20250905T080916.tif"
quicklook_png = "/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/dataset/ICEYE_X47_GRD_SLEDP_6074302_20250905T080916.png"
geojson_path = "/home/gpaps/PycharmProject/Esa_Ships/post/geo_output/predictions.geojson"

# 1) make a quicklook
# 1) load existing quicklook (already bright) and compute scale to full scene
with rasterio.open(scene_tif) as src:
    transform = src.transform
    H_full, W_full = src.height, src.width

arr = cv2.imread(quicklook_png, cv2.IMREAD_COLOR)  # BGR
if arr is None:
    raise FileNotFoundError(f"Quicklook not found: {quicklook_png}")

# compute scaling from scene pixels -> quicklook pixels
q_h, q_w = arr.shape[:2]
sx = q_w / float(W_full)
sy = q_h / float(H_full)



# 2) draw predictions
with open(geojson_path) as f:
    gj = json.load(f)

for feat in gj["features"]:
    poly = shape(feat["geometry"])
    coords = np.array(poly.exterior.coords)

    # map -> scene pixel (invert affine)
    px_scene = ~transform * (coords[:, 0], coords[:, 1])  # (x,y) in scene pixels
    px_scene = np.array(px_scene).T

    # scene pixel -> quicklook pixel
    px_qkl = px_scene.copy()
    px_qkl[:, 0] *= sx   # scale x
    px_qkl[:, 1] *= sy   # scale y

    pts = px_qkl.astype(int)
    cv2.polylines(arr, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

cv2.imwrite(quicklook_png, arr)
print("Wrote overlay on", quicklook_png)
