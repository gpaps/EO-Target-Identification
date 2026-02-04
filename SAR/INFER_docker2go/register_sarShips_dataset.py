import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Relative path to your JSON files and dataset
base_path = "./json"

splits = {
    "Ships_SAR_train": {
        # "json_file": f"/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/json/coco_train.json",
        # "image_root": "/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/dataset/png/"
        # "json_file": "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships.json",
        # "image_root":"/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships_dataset/",
        "json_file": "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/800x88sar_ships.json",
        "image_root": "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/800x800_Images_crop_400p/",
    },
    "Ships_SAR_val": {
        # "json_file": f"/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/json/coco_train.json",
        # "image_root": "/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/dataset/png/"
        "json_file": "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships.json",
        "image_root": "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships_dataset/",
    },
    "Ships_SAR_test": {
        # "json_file": f"/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/json/coco_train.json",
        # "image_root": "/home/gpaps/PycharmProject/Esa_Ships/SAR/INFER_docker2go/dataset/png/"
        "json_file": "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships.json",
        "image_root": "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/master_/sar_ships_dataset/",
    }
}


def register_datasets():
    for name, info in splits.items():
        if name in MetadataCatalog.list():
            print(f"{name} already registered.")
            continue
        assert os.path.exists(info["json_file"]), f"Missing {info['json_file']}"
        assert os.path.exists(info["image_root"]), f"Missing {info['image_root']}"
        register_coco_instances(name, {}, info["json_file"], info["image_root"])
        print(f"Registered {name}")


if __name__ == "__main__":
    register_datasets()
