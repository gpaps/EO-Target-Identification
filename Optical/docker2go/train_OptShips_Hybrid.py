import os, json, warnings, argparse, torch, random
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader, MetadataCatalog, \
    DatasetFromList, MapDataset
from detectron2.data import get_detection_dataset_dicts
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter
from detectron2.checkpoint import DetectionCheckpointer
import torch.backends.cudnn as cudnn

# Custom Import (Ensure these files exist in your folder)
from utils.generate_heatmap import generate_class_distribution_heatmap
from utils.tensorboard_utils import log_image_to_tensorboard, logText
from samplers.balanced_sampler import BalancedSampler
from register_Ship_dataset import register_datasets

from yacs.config import CfgNode as CN

warnings.simplefilter(action='ignore', category=FutureWarning)
Image.MAX_IMAGE_PIXELS = 100_000_000

# 1. OPTIMIZATION
cudnn.benchmark = True
register_datasets()


# 2. SAFETY UTIL
def prepare_output_dir(output_dir, force=False):
    if os.path.exists(output_dir) and not force:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir = f"{output_dir}_{timestamp}"
        print(f"[I/O] Output directory exists. Switching to: {new_dir}")
        return new_dir
    return output_dir


# 3. CLASS AWARE MAPPER (Resolution Normalization Strategy)
class ClassAwareMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = cfg
        self.standard_aug = [
            # SCALING STRATEGY: 900-1200px (The "Sweet Spot" from our analysis)
            T.ResizeShortestEdge(short_edge_length=(900, 1000, 1100, 1200), max_size=1333, sample_style="choice"),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ]
        self.extra_aug = [
            # Only Rotation. No texture destruction.
            T.RandomApply(T.RandomRotation(angle=[-45, 45]), prob=0.5),
        ]
        self.target_class_ids = [2, 3, 4]

    def __call__(self, dataset_dict):
        # Validation Logic (Keep GT for Eval)
        if not self.is_train and getattr(self.cfg, "KEEP_GT_IN_VAL", False):
            self.is_train = True
            # Validate at 1000px to match training "eye"
            self.tfm_gens = [T.ResizeShortestEdge(short_edge_length=(1000, 1000), max_size=1333, sample_style="choice")]
            try:
                ret = super().__call__(dataset_dict)
            finally:
                self.is_train = False
            return ret

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            self.tfm_gens = [T.ResizeShortestEdge(short_edge_length=(1000, 1000), max_size=1333, sample_style="choice")]
            return super().__call__(dataset_dict)

        # Training Logic
        annos = dataset_dict.get("annotations", [])
        has_target = any(obj["category_id"] in self.target_class_ids for obj in annos)
        aug_list = self.standard_aug + self.extra_aug if has_target else self.standard_aug
        self.tfm_gens = aug_list

        try:
            return super().__call__(dataset_dict)
        except ValueError as e:
            if len(annos) == 0:
                dataset_dict["instances"] = []
                return dataset_dict
            else:
                return None


class AugmentedTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        val_loader = build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            mapper=ClassAwareMapper(self.cfg, is_train=False)
        )
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TEST[0]).thing_classes
        hooks.insert(-1, EvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            val_loader,
            self.cfg.OUTPUT_DIR,
            class_names,
            self.cfg.EARLY_STOP.PATIENCE
        ))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TRAIN,
                                                    filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
        dataset = MapDataset(DatasetFromList(dataset_dicts, copy=False), ClassAwareMapper(cfg, is_train=True))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=BalancedSampler(dataset, dataset_dicts, cfg.SOLVER.IMS_PER_BATCH, cfg,
                                    oversample_factor=cfg.get("OVERSAMPLE_FACTOR", 1)),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=lambda x: x
        )


# [Keeping your MiniAirEvaluator / EvalHook / dump_predictions_to_csv as is]
# ... (Paste your Eval classes here or ensure they are imported) ...
# To save space, I assume the Eval classes are identical to previous script.
# IF YOU NEED THEM RE-PASTED, LET ME KNOW.

# --- EVALUATION CLASSES (Condensed for execution) ---
class MiniAirEvaluator:
    def __init__(self, df, output_dir, class_names):
        self.df = df.copy()
        self.output_dir = output_dir
        self.class_names = class_names
        self.bg_idx = len(class_names)

    # ... (Standard logic) ...
    def plot_canonical_confusion_matrix(self, iou_thr=0.5):
        pass  # Placeholder: Ensure you have the full class definition from previous files


class EvalHook(HookBase):
    def __init__(self, eval_period, model, val_loader, output_dir, class_names, patience):
        self._period = eval_period
        self.model = model
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.class_names = class_names
        self.patience = patience
        self.best_ap = -1
        self.counter = 0

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if next_iter % self._period == 0 or is_final:
            results = inference_on_dataset(
                self.model,
                self.val_loader,
                COCOEvaluator(self.trainer.cfg.DATASETS.TEST[0], self.trainer.cfg, False, self.output_dir)
            )
            print(f"[EvalHook] Eval results at iter {next_iter}: {results}")
            current_ap = results.get("bbox", {}).get("AP", -1)

            if current_ap > self.best_ap:
                self.best_ap = current_ap
                self.counter = 0
                checkpointer = DetectionCheckpointer(self.model, self.output_dir)
                checkpointer.save("model_best")
            else:
                self.counter += 1
                print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(" Early stopping triggered.")
                self.trainer.storage.put_scalars(early_stop_iter=next_iter)
                raise Exception("EARLY_STOP")


def dump_predictions_to_csv(model, data_loader, output_path):
    pass  # Placeholder


# ---------------------------------------------------------
# The Setup Function (Dense RPN Strategy)
# ---------------------------------------------------------
def setup_and_train(output_dir, num_classes, args):
    cfg = get_cfg()

    # --- 1. ARCHITECTURE ---
    if args.backbone == "r101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("Ship_Optical_train",)
    cfg.DATASETS.TEST = ("Ship_Optical_val",)
    cfg.DATALOADER.NUM_WORKERS = 16

    # --- 2. SOLVER ---
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.base_iters
    cfg.SOLVER.WARMUP_ITERS = 2000

    step_1 = int(0.7 * cfg.SOLVER.MAX_ITER)
    step_2 = int(0.9 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.STEPS = (step_1, step_2)
    period = int(0.05 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.CHECKPOINT_PERIOD = period
    cfg.TEST.EVAL_PERIOD = period
    cfg.SOLVER.AMP.ENABLED = True

    # --- 3. INPUT (Resolution Normalization) ---
    cfg.INPUT.MIN_SIZE_TRAIN = (900, 1000, 1100, 1200)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "none"

    # --- 4. DENSE RPN UPGRADE (The New Strategy) ---
    # A. Anchors: Interleaved sizes to catch "In-Between" Recreational boats
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24], [32, 48], [64, 96], [128, 192], [256, 384]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]

    # B. High Batch Size: Force model to classify hard background
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.33

    # C. NMS: Keep 0.8 to preserve crowded ships (0.7 kills neighbors!)
    cfg.MODEL.RPN.NMS_THRESH = 0.8
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    # -----------------------------------------------

    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms

    cfg.EARLY_STOP = CN()
    cfg.EARLY_STOP.PATIENCE = 3
    cfg.OUTPUT_DIR = output_dir
    cfg.RARE_CLASS_IDS = [2, 4]
    cfg.OVERSAMPLE_FACTOR = 4
    cfg.KEEP_GT_IN_VAL = True
    cfg.TEST.ENABLE_AIR_EVAL = True
    cfg.TEST.ENABLE_TB_IMAGES = True

    os.makedirs(output_dir, exist_ok=True)
    logText(cfg, cfg.OUTPUT_DIR)

    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except Exception as e:
        if str(e) == "EARLY_STOP":
            print("✅ Early stopping exit cleanly.")
        else:
            raise

    # Final Eval logic (Same as before)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=ClassAwareMapper(cfg, is_train=False))
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, cfg.OUTPUT_DIR)
    inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="r101")
    parser.add_argument("--batch", type=int, default=16)  # Safe for A100
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--base-iters", type=int, default=40000)

    parser.add_argument("--name", type=str, default="R101_DenseRPN_Strategy")
    parser.add_argument("--nms", type=float, default=0.65)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--resume-weights", type=str, default="")
    parser.add_argument("--cont-lr", type=float, default=1e-4)
    parser.add_argument("--extra-iters", type=int, default=0)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    output_dir = f"./trained_models_/superdataset/Optical_{args.name}"
    output_dir = prepare_output_dir(output_dir, force=args.force)

    train_json = "./json/coco_train.json"
    with open(train_json, 'r') as f:
        data = json.load(f)
    num_classes = len(data["categories"])

    setup_and_train(output_dir, num_classes, args)