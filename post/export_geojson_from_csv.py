# post/export_geojson_from_csv.py
# Minimal post-step: CSV (tile px) + manifest -> GeoJSON (map coords) + metrics.json
# Tries rasterio->gdalinfo->XML (bilinear corners). No new dependencies required.

import csv, json, math, time, subprocess, xml.etree.ElementTree as ET
from pathlib import Path


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row


def _try_rasterio_transform(tif_path):
    try:
        import rasterio  # optional
        with rasterio.open(tif_path) as src:
            tfm = src.transform  # Affine(a,b,c,d,e,f)
            crs = src.crs
            return (tfm.a, tfm.b, tfm.c, tfm.d, tfm.e, tfm.f), (crs.to_string() if crs else None)
    except Exception:
        return None, None


def _try_gdalinfo_transform(tif_path):
    try:
        proc = subprocess.run(["gdalinfo", "-json", str(tif_path)], capture_output=True, text=True, check=True)
        info = json.loads(proc.stdout)
        gt = info.get("geoTransform")  # [c,a,b,f,d,e] order in GDAL
        if gt and len(gt) == 6:
            c, a, b, f, d, e = gt
            # Convert GDAL order to (a,b,c,d,e,f)
            return (a, b, c, d, e, f), info.get("coordinateSystem", {}).get("wkt", None)
    except Exception:
        pass
    return None, None


def _xml_corner_mapping(xml_path):
    # Returns ((cols,rows),(lon_FN,lat_FN),(lon_FF,lat_FF),(lon_LN,lat_LN),(lon_LF,lat_LF))
    # ICEYE XML corner tags vary; we read the common ones.
    tree = ET.parse(xml_path);
    root = tree.getroot()

    def _get_text(tag):
        e = root.find(f".//{tag}")
        return e.text.strip() if e is not None and e.text else None

    # Dimensions
    n_rng = root.find(".//number_of_range_samples")
    n_az = root.find(".//number_of_azimuth_samples")
    if n_rng is None or n_az is None:
        return None
    W = int(n_rng.text.strip());
    H = int(n_az.text.strip())

    # Corners (first_near, first_far, last_near, last_far)
    # Expect elements like:
    # <coord_first_near><lat>..</lat><lon>..</lon></coord_first_near>
    def _corner(tag):
        node = root.find(f".//{tag}")
        if node is None: return None
        lat = node.findtext("lat")
        lon = node.findtext("lon")
        if lat is None or lon is None: return None
        return (float(lon), float(lat))  # (lon, lat)

    FN = _corner("coord_first_near")
    FF = _corner("coord_first_far")
    LN = _corner("coord_last_near")
    LF = _corner("coord_last_far")
    if not all([FN, FF, LN, LF]): return None

    return (W, H), FN, FF, LN, LF


def _pix_to_map_affine(x, y, A):
    # A = (a,b,c,d,e,f) maps pixel (x,y) to map (X,Y) at pixel corners
    a, b, c, d, e, f = A
    X = a * x + b * y + c
    Y = d * x + e * y + f
    return float(X), float(Y)


def _pix_to_lonlat_bilinear(x, y, W, H, FN, FF, LN, LF):
    # x in [0..W-1], y in [0..H-1]
    if W <= 1 or H <= 1:
        return None
    u = (x) / (W - 1)
    v = (y) / (H - 1)

    # Bilinear in lon/lat separately
    def lerp2(p00, p10, p01, p11, u, v):
        return (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + (1 - u) * v * p01 + u * v * p11

    lon = lerp2(FN[0], FF[0], LN[0], LF[0], u, v)
    lat = lerp2(FN[1], FF[1], LN[1], LF[1], u, v)
    return float(lon), float(lat)


def export_geojson(
        scene_tif,
        detections_csv,
        manifest_csv,
        out_geojson,
        out_metrics_json,
        xml_path=None,
        class_field_guess=("class", "pred_class", "label", "cls"),
        score_field_guess=("score", "conf", "confidence", "prob")
):
    """
    Convert per-tile detections (pixel) to a single GeoJSON (map coords), plus metrics.json.

    detections_csv columns (flexible): must include bbox + tile filename to join with manifest.
      bbox fields: any of x1/xmin/left, y1/ymin/top, x2/xmax/right, y2/ymax/bottom (case-insensitive)
      tile reference: one of ["tile","file","image","filename","path","img"] (string containing tile basename)

    manifest_csv columns: file_name, x, y, w, h (as written by your tiler)
    """
    t0 = time.time()
    scene_tif = Path(scene_tif);
    detections_csv = Path(detections_csv);
    manifest_csv = Path(manifest_csv)
    out_geojson = Path(out_geojson);
    out_geojson.parent.mkdir(parents=True, exist_ok=True)

    # 1) get pixel->map transform
    A, crs_str = _try_rasterio_transform(scene_tif)
    use = "rasterio" if A else None
    if not A:
        A, crs_str = _try_gdalinfo_transform(scene_tif)
        use = "gdalinfo" if A else None

    # 2) optional XML fallback (bilinear lon/lat)
    xml_info = None
    if not A and xml_path:
        xml_info = _xml_corner_mapping(xml_path)
        use = "xml_bilinear" if xml_info else None

    if not A and not xml_info:
        raise RuntimeError("No georeferencing available: rasterio/gdalinfo/XML all failed.")

    # 3) load manifest: map tile basename -> (x_off,y_off)
    tile_offsets = {}
    for r in _read_csv_rows(manifest_csv):
        tile_name = Path(r["file_name"]).name
        tile_offsets[tile_name] = (int(float(r["x"])), int(float(r["y"])))

    # 4) helpers to pick columns from detections CSV
    def pick(fieldnames, candidates):
        s = {f.lower(): f for f in fieldnames}
        for c in candidates:
            if c.lower() in s: return s[c.lower()]
        return None

    # likely bbox field name variants
    def find_bbox_cols(header):
        # accept many common variants, including your 'bbox_*' names
        lower_map = {h.lower(): h for h in header}

        def pick(*cands):
            for c in cands:
                if c in lower_map:
                    return lower_map[c]
            return None

        x1 = pick("bbox_x1", "x1", "xmin", "left")
        y1 = pick("bbox_y1", "y1", "ymin", "top")
        x2 = pick("bbox_x2", "x2", "xmax", "right")
        y2 = pick("bbox_y2", "y2", "ymax", "bottom")
        return x1, y1, x2, y2

    # 5) iterate detections, convert coords
    features = []
    per_class = {}
    scores = []

    with open(detections_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        header = rdr.fieldnames or []
        tile_field = pick(header, ("tile", "file", "image", "filename", "path", "img"))
        cls_field = pick(header, class_field_guess)
        scr_field = pick(header, score_field_guess)
        bx = find_bbox_cols(header)

        if not tile_field or not all(bx):
            raise ValueError(f"Detections CSV is missing required fields. Found header: {header}")

        for row in rdr:
            tile_base = Path(row[tile_field]).name
            if tile_base not in tile_offsets:
                # try just basename match (some rows may have full path)
                tile_base = Path(tile_base).name
                if tile_base not in tile_offsets:
                    continue
            dx, dy = tile_offsets[tile_base]

            # tile px -> scene px
            x1 = float(row[bx[0]]) + dx
            y1 = float(row[bx[1]]) + dy
            x2 = float(row[bx[2]]) + dx
            y2 = float(row[bx[3]]) + dy

            # scene px -> map coords
            if A:
                p = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                poly = [_pix_to_map_affine(x, y, A) for (x, y) in p]
            else:
                (W, H), FN, FF, LN, LF = xml_info
                p = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                poly = [_pix_to_lonlat_bilinear(x, y, W, H, FN, FF, LN, LF) for (x, y) in p]

            poly.append(poly[0])  # close ring

            klass = row.get(cls_field, "object") if cls_field else "object"
            score = float(row.get(scr_field, 0.0)) if scr_field in row else 0.0

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [poly]},
                "properties": {
                    "class": klass,
                    "score": score,
                    "tile": tile_base
                }
            })
            per_class[klass] = per_class.get(klass, 0) + 1
            scores.append(score)

    # 6) write GeoJSON
    crs_name = None
    if A and crs_str:
        crs_name = crs_str
    elif not A:
        crs_name = "WGS84"  # XML gives lon/lat

    fc = {
        "type": "FeatureCollection",
        "name": "predictions",
        "crs": {"type": "name", "properties": {"name": crs_name or "UNKNOWN"}},
        "features": features
    }
    out_geojson.write_text(json.dumps(fc, indent=2))

    # 7) metrics.json
    metrics = {
        "detections_total": len(features),
        "per_class": per_class,
        "avg_confidence": (sum(scores) / len(scores)) if scores else 0.0,
        "runtime_sec": round(time.time() - t0, 3),
        "georef_source": use
    }
    Path(out_metrics_json).write_text(json.dumps(metrics, indent=2))
