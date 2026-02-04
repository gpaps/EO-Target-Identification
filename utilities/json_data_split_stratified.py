import json, os, random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from collections import Counter

# -------------------- CONFIG --------------------
# INPUT_JSON = "../updated_coco_with_hierarchy.json"
# INPUT_JSON = "/home/gpaps/PycharmProject/Esa_Ships/Optical/json/VHRShips_Imagenet_Consolidated.json"
INPUT_JSON = "/home/gpaps/PycharmProject/Esa_Ships/SAR/json/SAR_Ships_[HRSID-SSDD]_cleaned.json"
# OUTPUT_DIR = "Optical/docker2go/VHRships_Imagenet/json/"
OUTPUT_DIR = "tempSAR/"
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10
SEED = 42
# ------------------------------------------------

assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "Ratios must sum to 1."

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_JSON, 'r') as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
annotations = coco["annotations"]
categories = coco["categories"]

# Index by image
image_to_annots = defaultdict(list)
image_to_classes = defaultdict(set)
class_to_images = defaultdict(set)

for ann in annotations:
    img_id = ann["image_id"]
    cat_id = ann["category_id"]
    image_to_annots[img_id].append(ann)
    image_to_classes[img_id].add(cat_id)
    class_to_images[cat_id].add(img_id)

# Start val/test assignment: maintain coverage per class
val_ids, test_ids = set(), set()
for cat_id, img_ids in class_to_images.items():
    img_ids = list(img_ids)
    total = len(img_ids)
    val_n = max(1, int(total * VAL_RATIO))
    test_n = max(1, int(total * TEST_RATIO))

    sampled = random.sample(img_ids, val_n + test_n)
    val_ids.update(sampled[:val_n])
    test_ids.update(sampled[val_n:val_n + test_n])

# Resolve overlap
test_ids -= val_ids
train_ids = set(images.keys()) - val_ids - test_ids

# ---------- PATCH: Stratified BG-only images ----------
# Identify BG-only image IDs
bg_image_ids = set(images.keys()) - set(image_to_annots.keys())
bg_image_ids = list(bg_image_ids)
random.shuffle(bg_image_ids)

total_bg = len(bg_image_ids)
val_bg_n = int(VAL_RATIO * total_bg)
test_bg_n = int(TEST_RATIO * total_bg)

val_ids.update(bg_image_ids[:val_bg_n])
test_ids.update(bg_image_ids[val_bg_n:val_bg_n + test_bg_n])
train_ids.update(bg_image_ids[val_bg_n + test_bg_n:])


# ------------------------------------------------------


# Build output dicts
def build_subset(ids, exclude_bg_images=False):
    subset_anns = []
    image_ids_with_anns = set()

    for a in annotations:
        if a["image_id"] in ids:
            a["iscrowd"] = a.get("iscrowd", 0)
            if "area" not in a or a["area"] == 0:
                bbox = a["bbox"]
                a["area"] = bbox[2] * bbox[3]
            subset_anns.append(a)
            image_ids_with_anns.add(a["image_id"])

    if exclude_bg_images:
        ids = [i for i in ids if i in image_ids_with_anns]

    return {
        "images": [images[i] for i in ids],
        "annotations": subset_anns,
        "categories": categories
    }


train_data = build_subset(train_ids, exclude_bg_images=False)
val_data = build_subset(val_ids, exclude_bg_images=False)
test_data = build_subset(test_ids, exclude_bg_images=False)

val_image_ids = set(img['id'] for img in val_data['images'])
val_gt_image_ids = set(ann['image_id'] for ann in val_data['annotations'])
bg_in_val = val_image_ids - val_gt_image_ids


# print(f"❗ BG-only images in val: {len(bg_in_val)}")
def count_bg(images, annotations):
    image_ids = {img['id'] for img in images}
    gt_ids = {ann['image_id'] for ann in annotations}
    return len(image_ids - gt_ids)


print(f"❗ BG images in train: {count_bg(train_data['images'], train_data['annotations'])}")
print(f"❗ BG images in val:   {count_bg(val_data['images'], val_data['annotations'])}")
print(f"❗ BG images in test:  {count_bg(test_data['images'], test_data['annotations'])}")

# Save to disk
with open(os.path.join(OUTPUT_DIR, "coco_train.json"), "w") as f:
    json.dump(train_data, f)

with open(os.path.join(OUTPUT_DIR, "coco_val.json"), "w") as f:
    json.dump(val_data, f)

with open(os.path.join(OUTPUT_DIR, "coco_test.json"), "w") as f:
    json.dump(test_data, f)

# Summary
print("\n✅ Stratified Balanced Split Complete:")
print(f"Total Images: {len(images)}")
print(f"Train: {len(train_data['images'])}, Annotations: {len(train_data['annotations'])}")
print(f"Val:   {len(val_data['images'])}, Annotations: {len(val_data['annotations'])}")
print(f"Test:  {len(test_data['images'])}, Annotations: {len(test_data['annotations'])}")


def count_images_per_category(image_ids, annotations, categories):
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_to_images = defaultdict(set)

    for ann in annotations:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name[ann["category_id"]]
        cat_to_images[cat_name].add(img_id)

    return {cat: len(imgs & image_ids) for cat, imgs in cat_to_images.items()}


# Collect image ID sets
train_img_ids = {img["id"] for img in train_data["images"]}
val_img_ids = {img["id"] for img in val_data["images"]}
test_img_ids = {img["id"] for img in test_data["images"]}

# Count
train_counts = count_images_per_category(train_img_ids, train_data["annotations"], categories)
val_counts = count_images_per_category(val_img_ids, val_data["annotations"], categories)
test_counts = count_images_per_category(test_img_ids, test_data["annotations"], categories)

# Pretty print
print("\nPer-Category Image Counts:")
print(f"{'Category':<22} {'Train':>8} {'Val':>8} {'Test':>8}")
print("-" * 50)
for cat in sorted(train_counts.keys()):
    t = train_counts.get(cat, 0)
    v = val_counts.get(cat, 0)
    s = test_counts.get(cat, 0)
    print(f"{cat:<22} {t:>8} {v:>8} {s:>8}")
