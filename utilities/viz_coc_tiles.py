#!/usr/bin/env python3
import os, json, random
from pathlib import Path
from PIL import Image, ImageDraw

COCO_JSON = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318/tiles_5120/coco/coco_annotationsv4.json"    # or coco_annotations.json
TILES_DIR  = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318/tiles_5120/"                          # folder with the tile PNGs
OUT_DIR    = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318/tiles_5120/_viz"
NUM        = 12                           # how many tiles to render

os.makedirs(OUT_DIR, exist_ok=True)

with open(COCO_JSON, "r") as f:
    coco = json.load(f)

id2img = {im["id"]: im for im in coco["images"]}
img2anns = {im["id"]: [] for im in coco["images"]}
for ann in coco["annotations"]:
    img2anns[ann["image_id"]].append(ann)

# choose a mix: some with anns, some without
with_anns = [i for i,a in img2anns.items() if a]
without  = [i for i,a in img2anns.items() if not a]
sample = random.sample(with_anns, min(len(with_anns), NUM//2)) + \
         random.sample(without, min(len(without), max(0, NUM-len(with_anns))))
for img_id in sample:
    imrec = id2img[img_id]
    fname = imrec["file_name"]
    path = os.path.join(TILES_DIR, fname)
    if not os.path.exists(path):
        continue
    im = Image.open(path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for ann in img2anns[img_id]:
        x,y,w,h = ann["bbox"]
        dr.rectangle([x, y, x+w-1, y+h-1], outline=(255,0,0), width=3)
    im.save(os.path.join(OUT_DIR, f"viz_{img_id}_{Path(fname).stem}.png"))

print(f"[OK] wrote previews to {OUT_DIR}")
