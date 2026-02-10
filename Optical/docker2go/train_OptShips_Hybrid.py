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

# --- CUSTOM IMPORTS ---
from utils.tensorboard_utils import logText
from samplers.balanced_sampler import BalancedSampler
from register_Ship_dataset import register_datasets
from utils.mappers import ClassAwareMapper
# Added generate_class_distribution_heatmap to imports
from utils.evaluation_utils import EvalHook, MiniAirEvaluator, dump_predictions_to_csv, plot_loss_curves
from utils.generate_heatmap import generate_class_distribution_heatmap

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


# --- HELPER: GUARANTEED POST-PROCESSING ---
def run_final_analysis(cfg, model):
    print("\n[Post-Processing] Starting Final Analysis...")
    try:
        # 1. Loss Curves
        print("[1/4] Generating Loss Plots...")
        plot_loss_curves(cfg.OUTPUT_DIR)

        # 2. Class Distribution Heatmap (NEW)
        print("[2/4] Generating Class Distribution Heatmap...")
        csv_log_path = os.path.join(cfg.OUTPUT_DIR, "class_distribution.csv")
        # Check if BalancedSampler generated the CSV
        if os.path.exists(csv_log_path):
            generate_class_distribution_heatmap(csv_log_path, cfg.OUTPUT_DIR)
        else:
            print(f"[Warn] Class distribution CSV not found at {csv_log_path}. Skipping heatmap.")

        # 3. Validation Inference
        print("[3/4] Running Inference for Analysis (Threshold=0.05)...")
        # Lower threshold to capture even weak detections for analysis
        model.roi_heads.score_thresh_test = 0.05

        val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0],
                                                 mapper=ClassAwareMapper(cfg, is_train=False))
        csv_path = os.path.join(cfg.OUTPUT_DIR, "prediction_log_final.csv")

        # 4. Dump Predictions & Matrices
        print("[4/4] Processing Predictions...")
        df_preds = dump_predictions_to_csv(model, val_loader, csv_path)

        if not df_preds.empty:
            class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
            mini_eval = MiniAirEvaluator(df_preds, cfg.OUTPUT_DIR, class_names)
            mini_eval.plot_confusion_matrix()
            mini_eval.save_metrics_table()
            print("✅ Analysis Complete. All files generated.")
        else:
            print("⚠ No predictions generated. Skipping Matrix.")

    except Exception as e:
        print(f"❌ [Error] Final Analysis Failed: {e}")


def setup_and_train(output_dir, num_classes, args):
    cfg = get_cfg()

    # 1. ARCHITECTURE
    if args.backbone == "r101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # --- DISABLED DCN (To avoid runtime error) ---
    # cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, True, True, True]
    # cfg.MODEL.RESNETS.DEFORM_MODULATED = True
    # cfg.MODEL.RESNETS.NUM_GROUPS = 1

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

    # 3. INPUT
    cfg.INPUT.MIN_SIZE_TRAIN = (900, 1000, 1100, 1200)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "none"

    # 4. DENSE RPN
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24], [32, 48], [64, 96], [128, 192], [256, 384]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.33
    cfg.MODEL.RPN.NMS_THRESH = 0.8
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000

    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms

    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"

    cfg.EARLY_STOP = CN()
    cfg.EARLY_STOP.PATIENCE = 0
    cfg.OUTPUT_DIR = output_dir

    # --- CRITICAL CONFIGS ---
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.RARE_CLASS_IDS = [0, 2, 4]
    cfg.OVERSAMPLE_FACTOR = 4

    cfg.KEEP_GT_IN_VAL = True
    cfg.TEST.ENABLE_AIR_EVAL = True
    cfg.TEST.ENABLE_TB_IMAGES = True

    os.makedirs(output_dir, exist_ok=True)
    logText(cfg, cfg.OUTPUT_DIR)

    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # --- ROBUST EXECUTION BLOCK ---
    try:
        trainer.train()
    except Exception as e:
        if str(e) == "EARLY_STOP":
            print("✅ Early stopping triggered.")
        else:
            print(f"❌ Training Crashed: {e}")
            raise
    finally:
        # Load best model if available
        checkpointer = DetectionCheckpointer(trainer.model, cfg.OUTPUT_DIR)
        best_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
        if os.path.exists(best_path):
            print(f"[Info] Loading Best Weights: {best_path}")
            checkpointer.load(best_path)
        else:
            print("[Warn] model_best.pth not found. Using current weights.")

        run_final_analysis(cfg, trainer.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="r101")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--base-iters", type=int, default=40000)
    parser.add_argument("--name", type=str, default="R101_Hybrid_Refactored")
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