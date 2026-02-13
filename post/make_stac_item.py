# post/make_stac_item.py
import json, time, os, subprocess
from pathlib import Path

def _read_geo_from_tif(scene_tif):
    """
    Returns (epsg:int|None, bbox:[minx,miny,maxx,maxy]|None, geometry:GeoJSON|None, source:"rasterio"|"gdalinfo"|None)
    Tries rasterio (if present), else gdalinfo -json. Falls back to None values.
    """
    # 1) try rasterio
    try:
        import rasterio
        from rasterio.features import dataset_bounds
        with rasterio.open(scene_tif) as src:
            epsg = src.crs.to_epsg() if src.crs else None
            b = src.bounds  # left, bottom, right, top
            bbox = [float(b.left), float(b.bottom), float(b.right), float(b.top)]
            # simple rectangle footprint in image CRS
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [b.left,  b.bottom],
                    [b.right, b.bottom],
                    [b.right, b.top],
                    [b.left,  b.top],
                    [b.left,  b.bottom]
                ]]
            }
            return epsg, bbox, geometry, "rasterio"
    except Exception:
        pass

    # 2) try gdalinfo -json
    try:
        proc = subprocess.run(["gdalinfo", "-json", str(scene_tif)], capture_output=True, text=True, check=True)
        info = json.loads(proc.stdout)
        # bbox from corner coordinates if present
        corners = info.get("cornerCoordinates", {})
        # gdal gives lon/lat if dataset is georeferenced in geographic; otherwise in projected CRS
        # derive bbox from corners if we have at least UL/LR
        keys = ["upperLeft","lowerRight"]
        if all(k in corners for k in keys):
            ul = corners["upperLeft"]; lr = corners["lowerRight"]
            minx, miny = float(ul[0]), float(lr[1])
            maxx, maxy = float(lr[0]), float(ul[1])
            bbox = [minx, miny, maxx, maxy]
            geometry = {
                "type":"Polygon","coordinates":[[
                    [minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy],[minx,miny]
                ]]
            }
        else:
            bbox = geometry = None
        # epsg (best-effort)
        epsg = None
        srs = info.get("coordinateSystem", {})
        auth = srs.get("authority", {})
        if isinstance(auth, dict) and auth.get("name") == "EPSG" and "code" in auth:
            try: epsg = int(auth["code"])
            except: epsg = None
        return epsg, bbox, geometry, "gdalinfo"
    except Exception:
        pass

    return None, None, None, None

def _read_metrics(metrics_path):
    try:
        return json.loads(Path(metrics_path).read_text())
    except Exception:
        return {}

def _rel_or_abs(base_dir: Path, path_str: str):
    """Return relative path if inside base_dir, else absolute path."""
    p = Path(path_str)
    try:
        return str(p.relative_to(base_dir))
    except Exception:
        return str(p.resolve())

def make_stac_item(
    scene_tif: str,
    predictions_geojson: str,
    metrics_json: str,
    out_item_path: str,
    collection_id: str = "SS-TI",
    browse_png: str | None = None,
    classes: list[str] = None,
    properties_extra: dict | None = None,
    item_id: str | None = None,
):
    """Create a STAC Item (Label extension) referencing your outputs."""
    base = Path(out_item_path).parent
    base.mkdir(parents=True, exist_ok=True)

    # 1) geo (epsg/bbox/geometry)
    epsg, bbox, geometry, geo_source = _read_geo_from_tif(scene_tif)

    # 2) load metrics for handy props
    metrics = _read_metrics(metrics_json)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 3) item id & class list
    if classes is None: classes = ["ship"]  # default; adjust per use-case later
    if item_id is None:
        # use scene name + timestamp
        item_id = f"{Path(scene_tif).stem}_ti_{int(time.time())}"

    # 4) core item
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/label/v1.0.1/schema.json",
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
            "https://stac-extensions.github.io/processing/v1.1.0/schema.json"
        ],
        "id": item_id,
        "collection": collection_id,
        "bbox": bbox if bbox else None,
        "geometry": geometry if geometry else None,
        "properties": {
            "datetime": now_iso,
            "label:tasks": ["detection"],
            "label:properties": ["class","score"],
            "label:classes": [{"name": c} for c in classes],
            "processing:software": "SW-03-01 Target Identification",
            "processing:version": "0.1.0"
        },
        "assets": {}
    }

    # 5) projection props if we know EPSG
    if epsg:
        item["properties"]["proj:epsg"] = epsg

    # 6) attach metrics summary into properties (lightweight)
    if metrics:
        item["properties"]["ti:detections_total"] = metrics.get("detections_total")
        item["properties"]["ti:avg_confidence"] = metrics.get("avg_confidence")
        item["properties"]["ti:per_class"] = metrics.get("per_class")
        item["properties"]["ti:georef_source"] = metrics.get("georef_source")

    # 7) extra props if caller wants to inject
    if properties_extra:
        item["properties"].update(properties_extra)

    # 8) assets (use relative paths if possible, else absolute)
    item["assets"]["predictions"] = {
        "href": _rel_or_abs(base, predictions_geojson),
        "type": "application/geo+json",
        "roles": ["labels"],
        "title": "Detections (vector)"
    }
    item["assets"]["metrics"] = {
        "href": _rel_or_abs(base, metrics_json),
        "type": "application/json",
        "roles": ["metadata"],
        "title": "Runtime & counts"
    }
    if browse_png and Path(browse_png).exists():
        item["assets"]["browse"] = {
            "href": _rel_or_abs(base, browse_png),
            "type": "image/png",
            "roles": ["overview"],
            "title": "Overlay quicklook"
        }
    # optionally reference the source image (if you keep/copy it next to outputs)
    if Path(scene_tif).exists():
        item["assets"]["image"] = {
            "href": _rel_or_abs(base, scene_tif),
            "type": "image/tiff; application=geotiff",
            "roles": ["data"],
            "title": "Source image"
        }

    # 9) write item
    Path(out_item_path).write_text(json.dumps(item, indent=2))
    return {"item_path": str(out_item_path), "geo_source": geo_source, "epsg": epsg, "bbox": bbox}
