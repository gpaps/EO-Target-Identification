import os
import json
import torch
import argparse
import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.events import TensorboardXWriter
from torch.utils.tensorboard import SummaryWriter
from detectron2.data import MetadataCatalog, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper

from register_Ship_dataset import register_datasets


class DatasetMapperWithResizeOnly(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.augmentations = [
            T.ResizeShortestEdge(short_edge_length=(720, 1024), sample_style="range")
        ]

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        image, transforms = T.apply_augmentations(self.augmentations, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())

        if "annotations" in dataset_dict and self.is_train:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


class AugmentedTrainer(DefaultTrainer):
    def build_writers(self):
        return [TensorboardXWriter(os.path.join(self.cfg.OUTPUT_DIR, "tensorboard"))]

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapperWithResizeOnly(cfg, is_train=True))


def setup_and_train(output_dir, num_classes, lr=0.0001, batch=512, backbone="r50"):
    cfg = get_cfg()
    model_key = "faster_rcnn_R_101_FPN_3x.yaml" if backbone == "r101" else "faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_key}"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_key}")

    cfg.DATASETS.TRAIN = ("Ship_Optical_train",)
    cfg.DATASETS.TEST = ("Ship_Optical_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.OUTPUT_DIR = output_dir

    cfg.SOLVER.IMS_PER_BATCH = batch
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.STEPS = (20000, 25000)
    cfg.SOLVER.AMP.ENABLED = True
    cfg.TEST.EVAL_PERIOD = 5000

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RPN.NMS_THRESH = 0.6
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 5000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

    os.makedirs(output_dir, exist_ok=True)
    register_datasets()
    setup_logger()

    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    trainer.train()
    del trainer
    torch.cuda.empty_cache()

    evaluator = COCOEvaluator("Ship_Optical_val", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "Ship_Optical_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("\nEvaluation Results:", results)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    writer.add_scalar("Final_Val/AP", results["bbox"]["AP"], 0)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--backbone", type=str, default="r50", choices=["r50", "r101"])
    args = parser.parse_args()

    output_dir = f"./trained_models_Ship_Optical/safe/Optical_sweep_{args.name}"
    with open("../json/coco_train.json", "r") as f:
        categories = json.load(f)["categories"]
    num_classes = len(categories)

    setup_and_train(output_dir, num_classes, lr=args.lr, batch=args.batch, backbone=args.backbone)


if __name__ == "__main__":
    main()
