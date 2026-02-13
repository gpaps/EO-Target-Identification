import json

MIN_SIDE = 10  # or 12, we can tune this

with open("greek_harbors_raw.json", "r") as f:
    coco = json.load(f)

filtered_anns = []
for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    if min(w, h) >= MIN_SIDE:
        filtered_anns.append(ann)

coco["annotations"] = filtered_anns

with open("greek_harbors_filtered.json", "w") as f:
    json.dump(coco, f)


