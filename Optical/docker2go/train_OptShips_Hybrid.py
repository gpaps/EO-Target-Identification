import os, json, warnings, argparse, torch
from datetime import datetime

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, MetadataCatalog, \
    DatasetFromList, MapDataset
from detectron2.data import get_detection_dataset_dicts
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
import torch.backends.cudnn as cudnn

# --- CUSTOM IMPORTS (Clean!) ---
from utils.tensorboard_utils import logText
from samplers.balanced_sampler import BalancedSampler
from register_Ship_dataset import register_datasets
# NEW IMPORTS
from utils.mappers import ClassAwareMapper
# Update this line to include plot_loss_curves
from utils.evaluation_utils import EvalHook, MiniAirEvaluator, dump_predictions_to_csv, plot_loss_curves
from yacs.config import CfgNode as CN

warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. OPTIMIZATION
cudnn.benchmark = True
register_datasets()

def prepare_output_dir(output_dir, force=False):
    if os.path.exists(output_dir) and not force:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir = f"{output_dir}_{timestamp}"
        print(f"[I/O] Output directory exists. Switching to: {new_dir}")
        return new_dir
    return output_dir

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
        dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
        dataset = MapDataset(DatasetFromList(dataset_dicts, copy=False), ClassAwareMapper(cfg, is_train=True))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=BalancedSampler(dataset, dataset_dicts, cfg.SOLVER.IMS_PER_BATCH, cfg, oversample_factor=cfg.get("OVERSAMPLE_FACTOR", 1)),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=lambda x: x
        )
# ---------------------------------------------------------
# The Setup Function (DCN + CIoU + Dense RPN)
# ---------------------------------------------------------
def setup_and_train(output_dir, num_classes, args):
    cfg = get_cfg()

    # 1. ARCHITECTURE
    if args.backbone == "r101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("Ship_Optical_train",)
    cfg.DATASETS.TEST = ("Ship_Optical_val",)
    cfg.DATALOADER.NUM_WORKERS = 16

    # 2. SOLVER
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

    # SAFETY: Clip gradients to prevent DCN/CIoU explosion
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    # 3. INPUT (Resolution Normalization)
    cfg.INPUT.MIN_SIZE_TRAIN = (900, 1000, 1100, 1200)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "none"

    # 4. DENSE RPN UPGRADE
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24], [32, 48], [64, 96], [128, 192], [256, 384]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.33
    cfg.MODEL.RPN.NMS_THRESH = 0.8
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000

    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms

    # --- LOSS UPGRADE: CIoU ---
    # Replaces SmoothL1. Forces small boxes to align perfectly.
    # CIoU = Overlap + Center Distance + Aspect Ratio.
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"
    # --------------------------

    cfg.EARLY_STOP = CN()
    cfg.EARLY_STOP.PATIENCE = 10  # Increased patience for DCN convergence
    cfg.OUTPUT_DIR = output_dir

    # 5. DATASET FIXES
    # Force inclusion of Background images (Fixes False Positives on Land)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    # Force diverse sampling of Commercial (0), Fishing (2), Sub (4)
    cfg.RARE_CLASS_IDS = [0, 2, 4]
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

        # --- POST-PROCESSING ---
        print("\n[Post-Processing] Generating Final Artifacts...")
        checkpointer = DetectionCheckpointer(trainer.model, cfg.OUTPUT_DIR)
        best_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
        if os.path.exists(best_path):
            checkpointer.load(best_path)
        else:
            print("[WARN] model_best.pth not found! Using current weights.")

        csv_path = os.path.join(cfg.OUTPUT_DIR, "prediction_log_final.csv")
        val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0],
                                                 mapper=ClassAwareMapper(cfg, is_train=False))

        try:
            df_preds = dump_predictions_to_csv(trainer.model, val_loader, csv_path)
            class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
            mini_eval = MiniAirEvaluator(df_preds, cfg.OUTPUT_DIR, class_names)
            mini_eval.plot_confusion_matrix()

            print("[Post-Processing] Plotting Loss Curves...")
            plot_loss_curves(cfg.OUTPUT_DIR)

            print("✅ Analysis Complete.")
        except Exception as e:
            print(f"[Error] Post-processing failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="r101")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--base-iters", type=int, default=50000)  # Increased for DCN

    parser.add_argument("--name", type=str, default="R101_DCN_CIoU_Final")
    parser.add_argument("--nms", type=float, default=0.65)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    output_dir = f"./trained_models_/superdataset/Optical_{args.name}"
    output_dir = prepare_output_dir(output_dir, force=args.force)

    train_json = "./json/coco_train.json"
    with open(train_json, 'r') as f:
        data = json.load(f)
    num_classes = len(data["categories"])

    setup_and_train(output_dir, num_classes, args)