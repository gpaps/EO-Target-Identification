import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Relative path to your JSON files and dataset
base_path = "json"

splits = {
    "Ships_SAR_train": {
        "json_file": f"{base_path}/coco_train.json",
        "image_root": "./dataset/cluster/"
    },
    "Ships_SAR_val": {
        "json_file": f"{base_path}/coco_val.json",
        "image_root": "./dataset/cluster/"
    },
    "Ships_SAR_test": {
        "json_file": f"{base_path}/coco_test.json",
        "image_root": "./dataset/cluster/"
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
