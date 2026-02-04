import json
import os


def merge_multiple_coco_jsons(input_json_paths, output_json_path):
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    category_name_to_id = {}
    image_id_counter = 0
    annotation_id_counter = 0

    for json_idx, json_path in enumerate(input_json_paths):
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Handle categories
        for cat in data["categories"]:
            if cat["name"] not in category_name_to_id:
                new_id = len(category_name_to_id) + 1
                category_name_to_id[cat["name"]] = new_id
                merged_data["categories"].append({"id": new_id, "name": cat["name"]})

        # Map old category IDs to new ones
        old_cat_id_to_new = {
            cat["id"]: category_name_to_id[cat["name"]]
            for cat in data["categories"]
        }

        # Handle images
        old_image_id_to_new = {}
        for img in data["images"]:
            new_image_id = image_id_counter
            old_image_id_to_new[img["id"]] = new_image_id
            img["id"] = new_image_id
            merged_data["images"].append(img)
            image_id_counter += 1

        # Handle annotations
        for ann in data["annotations"]:
            ann["id"] = annotation_id_counter
            ann["image_id"] = old_image_id_to_new[ann["image_id"]]
            ann["category_id"] = old_cat_id_to_new[ann["category_id"]]
            merged_data["annotations"].append(ann)
            annotation_id_counter += 1

    # Save output
    with open(output_json_path, 'w') as f:
        json.dump(merged_data, f, indent=4)

    summary = {
        "output_json": output_json_path,
        "total_images": len(merged_data["images"]),
        "total_annotations": len(merged_data["annotations"]),
        "total_categories": len(merged_data["categories"]),
        "categories": list(category_name_to_id.keys())
    }
    print(json.dumps(summary, indent=2))


input_json_paths = [
    # "/media/gpaps/My Passport/CVRL-GeorgeP/_/Benchmark_Dataset/Ships/annotations/ship_BMP.json",
    # "/media/gpaps/My Passport/CVRL-GeorgeP/_/Benchmark_Dataset/Ships/annotations/ship_JPG.json"
    "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/CONSOLIDATED_SHIPS/instances_ships_consolidated.json",
    "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/VHRShips_ShipRSImageNet_master.json"
]
output_json_path = '/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/coco_dataset.json'


# Paths to your uploaded JSON files
# input_json_paths = [
#     "/coco_train.json",
#     "/coco_valid.json",
#     "/coco_test.json",
#     "/SAR_SADD_Airplane_Cleaned.json"
# ]
# output_json_path = "/mnt/data/merged_SADD_airplanes_all.json"

# input_json_paths = [
#     "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/bg_coco.json",
#     "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/VHRShips_Imagenet_Consolidated.json"
# ]
# output_json_path = '/Optical/json/VHRShips_Imagenet_Consolidated_greekBG.json'

# input_json_paths = [
#     "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X6_GRD_SLED_4410498_20241216T014014/annotations_corrected/ICEYE_X6_GRD_SLED_4410498_20241216T014014.json",
#     "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X13_GRD_SLED_4616902_20250415T144324/annotations_corrected/ICEYE_X13_GRD_SLED_4616902_20250415T144324.json",
#     "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218/annotations_corrected/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218.json",
#     "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X46_GRD_SLEDP_6083439_20250905T205736/annotations_corrected/ICEYE_X46_GRD_SLEDP_6083439_20250905T205736.json",
#     "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X47_GRD_SLEDP_6074302_20250905T080916/annotations_corrected/ICEYE_X47_GRD_SLEDP_6074302_20250905T080916.json",
#     "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X47_GRD_SLEDP_6113188_20250908T060147/annotations_corrected/ICEYE_X47_GRD_SLEDP_6113188_20250908T060147.json",
#     "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X47_GRD_SLEDP_6113189_20250908T060121/annotations_corrected/ICEYE_X47_GRD_SLEDP_6113189_20250908T060121.json",
#     "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318/annotations_corrected/ICEYE_X55_GRD_SLEDP_6067614_20250904T232318.json",
# ]
# output_json_path = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_json/sar_ships.json'


# Merge and summarize
merge_multiple_coco_jsons(input_json_paths, output_json_path)
