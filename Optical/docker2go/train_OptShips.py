import os, json, warnings, argparse, torch, random
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
#Custom Import
from utils.generate_heatmap import generate_class_distribution_heatmap
from utils.tensorboard_utils import log_image_to_tensorboard, logText
from utils.augmentations import GaussianBlurAll, T_DownsampleAug
from samplers.balanced_sampler import BalancedSampler
from register_Ship_dataset import register_datasets

from yacs.config import CfgNode as CN

warnings.simplefilter(action='ignore', category=FutureWarning)
Image.MAX_IMAGE_PIXELS = 100_000_000

register_datasets()


class ClassAwareMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = cfg
        self.standard_aug = [
            # Scaling: Handle resolution differences (SkySat vs xView)
            T.ResizeShortestEdge(short_edge_length=(800, 1024), sample_style="choice"),

            # Orientation: Space has no "Up". Flip BOTH ways.
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ]

        self.extra_aug = [
            # Rotation: Ships don't align to a grid.
            T.RandomApply(T.RandomRotation(angle=[-45, 45]), prob=0.5),

            # Atmospheric/Sensor Simulation: Haze (Blur) and GSD (Downsample)
            T.RandomApply(GaussianBlurAll(sigma_range=(0.8, 1.5)), prob=0.3),

            # CRITICAL: This mimics "bad" satellite days or lower-res sensors
            # T_DownsampleAug uses your augmentations.py logic
            T.RandomApply(T_DownsampleAug(scale_factor=2, prob=1.0), prob=0.3),

            # Lighting: Sun glint and cloud shadows
            T.RandomApply(T.RandomBrightness(0.8, 1.2), prob=0.4),
            T.RandomApply(T.RandomContrast(0.8, 1.2), prob=0.4),
        ]
        self.target_class_ids = [2, 4]  # Fishing ships, submarines

    def __call__(self, dataset_dict):
        if not self.is_train and not getattr(self.cfg, "KEEP_GT_IN_VAL", False):
            dataset_dict.pop("annotations", None)

        annos = dataset_dict.get("annotations", [])

        # Strategy: If it's a rare class, hit it with the full augmentation stack
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
                print(f"Unhandled error for {dataset_dict['file_name']}: {e}")
                return None


class AugmentedTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
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
        dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=False)
        dataset = MapDataset(DatasetFromList(dataset_dicts, copy=False), ClassAwareMapper(cfg, is_train=True))

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=BalancedSampler(dataset, dataset_dicts, cfg.SOLVER.IMS_PER_BATCH, cfg,
                                    oversample_factor=cfg.get("OVERSAMPLE_FACTOR", 1)),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=lambda x: x
        )


class MiniAirEvaluator:
    def __init__(self, df, output_dir, class_names):
        self.df = df.copy()
        self.output_dir = output_dir
        self.class_names = class_names
        self.bg_idx = len(class_names)

    @staticmethod
    def _iou_matrix(a, b):
        """
        Vectorized IoU calculation.
        a: [Na, 4] (x1, y1, x2, y2)
        b: [Nb, 4] (x1, y1, x2, y2)
        Returns: [Na, Nb] IoU matrix
        """
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)

        ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

        inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
        inter_y1 = np.maximum(ay1[:, None], by1[None, :])
        inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
        inter_y2 = np.minimum(ay2[:, None], by2[None, :])

        inter_w = np.clip(inter_x2 - inter_x1, 0, None)
        inter_h = np.clip(inter_y2 - inter_y1, 0, None)
        inter = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a[:, None] + area_b[None, :] - inter

        return np.where(union > 0, inter / union, 0.0)

    def plot_canonical_confusion_matrix(self, iou_thr=0.5):
        df = self.df
        y_true, y_pred = [], []

        # Group by image to handle matching per-image
        for img in df["image"].unique():
            df_img = df[df["image"] == img]

            # 1. Extract Ground Truths (gt_class != -1)
            gt_rows = df_img[df_img["gt_class"] != -1]
            g_boxes = gt_rows[["gx1", "gy1", "gx2", "gy2"]].dropna().to_numpy(dtype=np.float32)
            g_cls = gt_rows["gt_class"].astype(int).to_numpy()

            # 2. Extract Predictions (pred_class != -1)
            pr_rows = df_img[df_img["pred_class"] != -1]
            p_boxes = pr_rows[["px1", "py1", "px2", "py2"]].dropna().to_numpy(dtype=np.float32)
            p_cls = pr_rows["pred_class"].astype(int).to_numpy()
            p_scr = pr_rows["score"].to_numpy(dtype=np.float32)

            # --- Edge Cases ---
            if len(g_boxes) == 0:
                # No GT in image -> All preds are False Positives (Background)
                for c in p_cls:
                    y_true.append(self.bg_idx)
                    y_pred.append(int(c))
                continue

            if len(p_boxes) == 0:
                # No Preds in image -> All GT are False Negatives (Missed)
                for c in g_cls:
                    y_true.append(int(c))
                    y_pred.append(self.bg_idx)
                continue

            # --- Greedy Matching ---
            # Sort preds by confidence (high to low)
            order = np.argsort(-p_scr)
            iou = self._iou_matrix(p_boxes, g_boxes)
            matched_g = set()

            for pi in order:
                gi = int(np.argmax(iou[pi]))
                # Check threshold and ensure GT hasn't been used yet
                if iou[pi, gi] >= iou_thr and gi not in matched_g:
                    # True Positive (or Misclassification between classes)
                    y_true.append(int(g_cls[gi]))
                    y_pred.append(int(p_cls[pi]))
                    matched_g.add(gi)
                else:
                    # False Positive (Ghost detection)
                    y_true.append(self.bg_idx)
                    y_pred.append(int(p_cls[pi]))

            # --- Handle Missed GT (False Negatives) ---
            for gi in range(len(g_cls)):
                if gi not in matched_g:
                    y_true.append(int(g_cls[gi]))
                    y_pred.append(self.bg_idx)

        # 3. Save Matrix
        labels = list(range(self.bg_idx + 1))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Save as CSV for the table generator
        cm_df = pd.DataFrame(
            cm,
            index=self.class_names + ["background"],
            columns=self.class_names + ["background"],
        )
        cm_csv = os.path.join(self.output_dir, "confusion_matrix_val.csv")
        cm_df.to_csv(cm_csv)

        # 4. Plot Heatmap
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names + ["background"]
        )
        disp.plot(xticks_rotation=45, cmap="Purples", values_format="d")
        plt.title("Canonical Confusion Matrix (Val Eval)")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "confusion_matrix_val.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        if os.path.exists(fig_path):
            log_image_to_tensorboard(fig_path, "Val/Confusion_Matrix", os.path.join(self.output_dir, "tensorboard"))

    def plot_precision_recall_f1_table(self):
        cm_path = os.path.join(self.output_dir, "confusion_matrix_val.csv")
        if os.path.exists(cm_path):
            try:
                cm_df = pd.read_csv(cm_path, index_col=0)
                stats = []
                for cls in self.class_names:
                    if cls in cm_df.index and cls in cm_df.columns:
                        # Diagonal is True Positive
                        tp = cm_df.loc[cls, cls]
                        # Sum of Row = All Ground Truths for this class (TP + FN)
                        total_gt = cm_df.loc[cls].sum()
                        # Sum of Column = All Predictions for this class (TP + FP)
                        total_pred = cm_df[cls].sum()

                        precision = tp / total_pred if total_pred > 0 else 0
                        recall = tp / total_gt if total_gt > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                        stats.append(
                            {"Class": cls, "P": round(precision, 2), "R": round(recall, 2), "F1": round(f1, 2)})

                print("\n=== Derived Performance Metrics ===")
                df_results = pd.DataFrame(stats)
                print(df_results)

                # Save plot of table
                plt.figure(figsize=(10, len(df_results) * 0.6))
                sns.heatmap(df_results.set_index("Class"), annot=True, fmt=".2f", cmap="Blues", cbar=False)
                plt.title("Precision / Recall / F1 on Validation")
                plt.tight_layout()
                fig_path = os.path.join(self.output_dir, "val_metrics_table.png")
                plt.savefig(fig_path, dpi=300)
                plt.close()

                if os.path.exists(fig_path):
                    log_image_to_tensorboard(fig_path, "Val/F1_Precision_Table",
                                             os.path.join(self.output_dir, "tensorboard"))

            except Exception as e:
                print(f"Skipping F1 table: {e}")


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

            # TODO later I need to change cfg methods to custom
            # COCOEvaluator(self.trainer.cfg.DATASETS.TEST[0], self.trainer.cfg, False, self.output_dir )#Deprecated
            # COCOEvaluator(
            #     dataset_name=dataset_name,
            #     tasks=("bbox",),
            #     distributed=False,
            #     output_dir=output_dir
            # )

            print(f"[EvalHook] Eval results at iter {next_iter}: {results}")
            current_ap = results.get("bbox", {}).get("AP", -1)
            print(f"[EvalHook] Iter {next_iter} AP: {current_ap:.2f}")

            # ✅ Use the same path as the dump
            csv_name = f"prediction_log_iter{next_iter}.csv"
            csv_path = os.path.join(self.output_dir, csv_name)
            if self.trainer.cfg.TEST.ENABLE_TB_IMAGES:
                log_val_predictions_to_tensorboard(self.trainer.cfg, self.model, self.val_loader,
                                                   os.path.join(self.output_dir, "tensorboard"),
                                                   max_images=5,
                                                   global_step=next_iter)

                # log_val_predictions_to_tensorboard(self.trainer.cfg, self.model, self.val_loader,
                #                                    os.path.join(self.output_dir, "tensorboard"),
                #                                    max_images=5  # or whatever number you prefer
                #                                    )
                log_confusion_and_f1(csv_path, self.class_names, os.path.join(self.output_dir, "tensorboard"), next_iter)
            # Save predictions as CSV;if
            if self.trainer.cfg.TEST.ENABLE_AIR_EVAL:
                dump_predictions_to_csv(self.model, self.val_loader, csv_path)

                if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                    try:
                        df = pd.read_csv(csv_path)
                        evaluator = MiniAirEvaluator(df, self.output_dir, self.class_names)
                        evaluator.plot_canonical_confusion_matrix()
                        evaluator.plot_precision_recall_f1_table()
                    except Exception as e:
                        print(f"⚠ Could not read or evaluate {csv_path}: {e}")
                else:
                    print(f"⚠ Warning: CSV {csv_path} is empty or missing — skipping eval plot.")

            # ✅ Early stopping logic
            if current_ap > self.best_ap:
                self.best_ap = current_ap
                self.counter = 0
                # torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model_best.pth"))
                checkpointer = DetectionCheckpointer(self.model, self.output_dir)
                checkpointer.save("model_best")  # This creates model_best.pth and updates last_checkpoint
            else:
                self.counter += 1
                print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(" Early stopping triggered.")
                self.trainer.storage.put_scalars(early_stop_iter=next_iter)
                raise Exception("EARLY_STOP")


def dump_predictions_to_csv(model, data_loader, output_path):
    was_training = model.training
    model.eval()  # Temporarily switch to eval

    rows = []
    print(f"Dumping predictions to: {output_path}")
    for inputs in data_loader:
        outputs = model(inputs)
        for inp, out in zip(inputs, outputs):
            gt_classes = [x["category_id"] for x in inp.get("annotations", [])]
            pred_classes = out["instances"].pred_classes.cpu().tolist()
            # print(f"Image: {inp['file_name']}, GTs: {len(gt_classes)}, Preds: {len(pred_classes)}")
            for gt, pred in zip(gt_classes, pred_classes):
                rows.append({"image": inp["file_name"], "gt_class": gt, "pred_class": pred, "iou": 0.0})

    if not rows:
        print("⚠ Warning: No predictions met the threshold — CSV will be empty.")

    model.train(was_training)  # Restore model mode

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


# def log_val_predictions_to_tensorboard(cfg, model, data_loader, tb_dir, max_images=4):
def log_val_predictions_to_tensorboard(cfg, model, data_loader, tb_dir, max_images=8, global_step=0):
    was_training = model.training
    model.eval()  # Avoids RPN assertion when gt_instances are missing

    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    print(f" Logging up to {max_images} validation images to TensorBoard at: {tb_dir}")

    for idx, inputs in enumerate(data_loader):
        if idx >= max_images:
            break

        with torch.no_grad():
            outputs = model(inputs)

        if not outputs or "instances" not in outputs[0]:
            print(f"⚠ Warning: No instances predicted for image {inputs[0]['file_name']}")
            continue

        image_np = inputs[0]["image"].permute(1, 2, 0).cpu().numpy()
        if image_np.max() <= 1.0:
            image_np *= 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        v = Visualizer(image_np, metadata=metadata)
        out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))

        writer.add_image(f"val_prediction_{idx}", out.get_image(), global_step, dataformats="HWC")

    writer.close()
    model.train(was_training)  # Restore original mode


def log_confusion_and_f1(csv_path, class_names, tb_dir, global_step):
    if not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
        if "true_label" not in df.columns or "pred_label" not in df.columns:
            print("CSV missing required columns.")
            return

        f1 = f1_score(df["true_label"], df["pred_label"], average="macro")
        cm = confusion_matrix(df["true_label"], df["pred_label"])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        image = Image.open(buf).convert("RGB")
        image_np = np.array(image)

        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_image("Val/ConfusionMatrix", image_np, global_step, dataformats="HWC")
        writer.add_scalar("Val/F1_score", f1, global_step)
        writer.close()

    except Exception as e:
        print(f"Failed to log confusion matrix or F1: {e}")


def setup_and_train(output_dir, num_classes, lr=0.0001, roi_batch=512, nms=0.5, score=0.5,  backbone="r50"):
    cfg = get_cfg()
    # Backbone selection
    if backbone == "r101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("Ship_Optical_train",)
    cfg.DATASETS.TEST = ("Ship_Optical_val",)
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.SOLVER.IMS_PER_BATCH = 8

    # Core Solver Params
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WARMUP_ITERS = 1000
    # cfg.SOLVER.STEPS = (35000, 42000)
    # cfg.SOLVER.MAX_ITER = 45000
    cfg.SOLVER.STEPS = (12000, 16000)
    cfg.SOLVER.MAX_ITER = 20000
    # cfg.SOLVER.GAMMA = 0.1
    # cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.TEST.EVAL_PERIOD = 2000
    cfg.SOLVER.AMP.ENABLED = True
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    print(f" FILTER_EMPTY_ANNOTATIONS: {cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS}")

    # Input
    # FIX: Set to 1024 to match your tiled data
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # ROI Head
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    # cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"  # testcase
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score  # 7Drop weak/confused detections
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms  # 1Suppress overlapping duplicate boxes
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.75
    # cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = [0.5, 0.6, 0.7]

    # Replaced 512 with 16 to catch tiny fishing boats
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    # Added 0.2 (1:5) and 5.0 (5:1) for long Tankers/Subs
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]

    cfg.MODEL.RPN.NMS_THRESH = 0.6
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 8000   # more proposals to not drop tiny ships
    # cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000


    cfg.EARLY_STOP = CN()
    cfg.EARLY_STOP.PATIENCE = 4  # We can override this in argparse later if needed

    cfg.OUTPUT_DIR = output_dir
    cfg.RARE_CLASS_IDS = [2, 4]
    cfg.OVERSAMPLE_FACTOR = 3
    cfg.KEEP_GT_IN_VAL = True  # Retain annotations during val
    cfg.TEST.ENABLE_AIR_EVAL = True  # To skip during long sweeps|mid-evaluation|created a prediction log csv F1 etc
    cfg.TEST.ENABLE_TB_IMAGES = True  # Viz images in TB

    os.makedirs(output_dir, exist_ok=True)

    logText(cfg, cfg.OUTPUT_DIR)


    # cfg.MODEL.WEIGHTS = "/home/gpaps/esa-train/Ships_/Optical/trained_models_Ship_Optical/run_1/Optical_sweep_lr0.00015_b512_s0.5_nms0.4/model_final.pth"
    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except Exception as e:
        if str(e) == "EARLY_STOP":
            print(" Early stopping exit cleanly.")
        else:
            raise

    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=ClassAwareMapper(cfg, is_train=False))
    print("'\n'  Running final validation and visual analysis...'\n'")

    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, cfg.OUTPUT_DIR)  # (deprecated)
    #TODO update to the latest version
    # evaluator = COCOEvaluator(
    #     dataset_name=cfg.DATASETS.TEST[0],
    #     tasks=("bbox",),
    #     distributed=False,
    #     output_dir=cfg.OUTPUT_DIR
    # )

    metrics = inference_on_dataset(trainer.model, val_loader, evaluator)

    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, "tensorboard"))
    writer.add_scalar("Final_Val/AP", metrics["bbox"]["AP"], 0)
    writer.close()

    print("\nFinal Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    try:
        # log_val_predictions_to_tensorboard(cfg, trainer.model, val_loader, os.path.join(output_dir, "tensorboard"))
        log_val_predictions_to_tensorboard(self.trainer.cfg, self.model, self.val_loader,
                                           os.path.join(self.output_dir, "tensorboard"),
                                           max_images=10,
                                           global_step=next_iter)

    except Exception as e:
        print(f"⚠ Could not write tensorboard predictions: {e}")

    # ✅ Save final prediction CSV for summary inspection
    final_pred_csv = os.path.join(cfg.OUTPUT_DIR, "prediction_log.csv")
    dump_predictions_to_csv(trainer.model, val_loader, final_pred_csv)

    # Generate heatmap of batch-level class imbalance (if CSV is available)
    class_count_csv = os.path.join(cfg.OUTPUT_DIR, "batch_class_distribution.csv")
    if os.path.exists(class_count_csv):
        try:
            generate_class_distribution_heatmap(class_count_csv, cfg.OUTPUT_DIR)
        except Exception as e:
            print(f"[Heatmap] Failed to generate: {e}")

    prediction_log_path = os.path.join(output_dir, "prediction_log.csv")
    if os.path.exists(prediction_log_path):
        df = pd.read_csv(prediction_log_path)
        class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
        evaluator = MiniAirEvaluator(df, output_dir, class_names)
        evaluator.plot_canonical_confusion_matrix()
        evaluator.plot_precision_recall_f1_table()

    # ✅ Append sweep summary to CSV (optional but helpful)
    summary_csv = os.path.join(cfg.OUTPUT_DIR, "..", "sweep_summary.csv")
    summary_row = {
        "run_name": os.path.basename(cfg.OUTPUT_DIR),
        "AP": metrics["bbox"]["AP"],
        "LR": cfg.SOLVER.BASE_LR,
        "Batch": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        "RareClasses": cfg.RARE_CLASS_IDS,
        "OversampleFactor": cfg.OVERSAMPLE_FACTOR,
    }
    df_summary = pd.DataFrame([summary_row])
    df_summary.to_csv(summary_csv, mode="a", header=not os.path.exists(summary_csv), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help="Base learning rate")
    parser.add_argument("--batch", type=int, default=512, help="ROI-batch size per image")
    parser.add_argument("--name", type=str, default="default", help="Sweep run name")
    parser.add_argument("--nms", type=float, default=0.5, help="ROI-NMS Threshold")
    parser.add_argument("--score", type=float, default=0.5, help="ROI-Score Threshold")
    parser.add_argument("--backbone", type=str, choices=["r50", "r101"], default="r50")

    args = parser.parse_args()

    output_dir = f"./trained_models_/superdataset/Optical_sweep_{args.name}"

    with open("../json/coco_train.json", 'r') as f:
        data = json.load(f)
    num_classes = len(data["categories"])

    setup_and_train(output_dir,
                    num_classes,
                    lr=args.lr,
                    roi_batch=args.batch,
                    nms=args.nms,
                    score=args.score,
                    backbone='r50'
                    )