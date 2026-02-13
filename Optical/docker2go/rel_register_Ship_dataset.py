import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Use absolute paths to make it robust to working directory issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

splits = {
    "Ship_Optical_train": {
        "json_file": os.path.join(BASE_DIR, "json", "coco_train.json"),
        "image_root": os.path.join(BASE_DIR, "dataset", "Optical")
    },
    "Ship_Optical_val": {
        "json_file": os.path.join(BASE_DIR, "json", "coco_val.json"),
        "image_root": os.path.join(BASE_DIR, "dataset", "Optical")
    },
    "Ship_Optical_test": {
        "json_file": os.path.join(BASE_DIR, "json", "coco_test.json"),
        "image_root": os.path.join(BASE_DIR, "dataset", "Optical")
    }
}


def register_datasets():
    for name, info in splits.items():
        if name in MetadataCatalog.list():
            print(f"✅ {name} already registered.")
            continue
        assert os.path.exists(info["json_file"]), f"❌ Missing {info['json_file']}"
        assert os.path.exists(info["image_root"]), f"❌ Missing {info['image_root']}"
        register_coco_instances(name, {}, info["json_file"], info["image_root"])
        print(f"✅ Registered {name}")


if __name__ == "__main__":
    register_datasets()
