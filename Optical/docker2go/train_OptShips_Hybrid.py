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

# Custom Import
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


# 3. CLASS AWARE MAPPER (Optimized for the 57% Small Image Cluster)
class ClassAwareMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = cfg
        self.standard_aug = [
            # SCALING STRATEGY:
            # Upsamples the 600-800px images to 1000-1200px (Crucial for Fishing Boats)
            # Keeps the 1280px images native.
            T.ResizeShortestEdge(short_edge_length=(900, 1000, 1100, 1200), max_size=1333, sample_style="choice"),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ]
        self.extra_aug = [
            # Only Rotation. No texture destruction.
            T.RandomApply(T.RandomRotation(angle=[-45, 45]), prob=0.5),
        ]
        self.target_class_ids = [2, 4]

    def __call__(self, dataset_dict):
        # Validation Logic (Keep GT for Eval)
        if not self.is_train and getattr(self.cfg, "KEEP_GT_IN_VAL", False):
            self.is_train = True
            # Validate at a solid 1000px to see small objects
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


# --- EVALUATION CLASSES ---
class MiniAirEvaluator:
    def __init__(self, df, output_dir, class_names):
        self.df = df.copy()
        self.output_dir = output_dir
        self.class_names = class_names
        self.bg_idx = len(class_names)

    @staticmethod
    def _iou_matrix(a, b):
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

        for img in df["image"].unique():
            df_img = df[df["image"] == img]
            gt_rows = df_img[df_img["gt_class"] != -1]
            g_boxes = gt_rows[["gx1", "gy1", "gx2", "gy2"]].dropna().to_numpy(dtype=np.float32)
            g_cls = gt_rows["gt_class"].astype(int).to_numpy()

            pr_rows = df_img[df_img["pred_class"] != -1]
            p_boxes = pr_rows[["px1", "py1", "px2", "py2"]].dropna().to_numpy(dtype=np.float32)
            p_cls = pr_rows["pred_class"].astype(int).to_numpy()
            p_scr = pr_rows["score"].to_numpy(dtype=np.float32)

            if len(g_boxes) == 0:
                for c in p_cls:
                    y_true.append(self.bg_idx)
                    y_pred.append(int(c))
                continue
            if len(p_boxes) == 0:
                for c in g_cls:
                    y_true.append(int(c))
                    y_pred.append(self.bg_idx)
                continue

            order = np.argsort(-p_scr)
            iou = self._iou_matrix(p_boxes, g_boxes)
            matched_g = set()
            for pi in order:
                gi = int(np.argmax(iou[pi]))
                if iou[pi, gi] >= iou_thr and gi not in matched_g:
                    y_true.append(int(g_cls[gi]))
                    y_pred.append(int(p_cls[pi]))
                    matched_g.add(gi)
                else:
                    y_true.append(self.bg_idx)
                    y_pred.append(int(p_cls[pi]))

            for gi in range(len(g_cls)):
                if gi not in matched_g:
                    y_true.append(int(g_cls[gi]))
                    y_pred.append(self.bg_idx)

        labels = list(range(self.bg_idx + 1))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        cm_df = pd.DataFrame(
            cm,
            index=self.class_names + ["background"],
            columns=self.class_names + ["background"],
        )
        cm_csv = os.path.join(self.output_dir, "confusion_matrix_val.csv")
        cm_df.to_csv(cm_csv)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names + ["background"])
        plt.figure(figsize=(10, 8))
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
                        tp = cm_df.loc[cls, cls]
                        total_gt = cm_df.loc[cls].sum()
                        total_pred = cm_df[cls].sum()

                        precision = tp / total_pred if total_pred > 0 else 0
                        recall = tp / total_gt if total_gt > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                        stats.append(
                            {"Class": cls, "P": round(precision, 2), "R": round(recall, 2), "F1": round(f1, 2)})

                print("\n=== Derived Performance Metrics ===")
                df_results = pd.DataFrame(stats)
                print(df_results)

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

            print(f"[EvalHook] Eval results at iter {next_iter}: {results}")
            current_ap = results.get("bbox", {}).get("AP", -1)
            print(f"[EvalHook] Iter {next_iter} AP: {current_ap:.2f}")

            csv_name = f"prediction_log_iter{next_iter}.csv"
            csv_path = os.path.join(self.output_dir, csv_name)

            if self.trainer.cfg.TEST.ENABLE_TB_IMAGES:
                log_val_predictions_to_tensorboard(self.trainer.cfg, self.model, self.val_loader,
                                                   os.path.join(self.output_dir, "tensorboard"),
                                                   max_images=5,
                                                   global_step=next_iter)

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
    from detectron2.structures import pairwise_iou
    was_training = model.training
    model.eval()

    rows = []
    print(f"Dumping predictions to: {output_path}")

    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            for inp, out in zip(inputs, outputs):

                gt_instances = inp.get("instances")
                gt_boxes = gt_instances.gt_boxes if gt_instances is not None and gt_instances.has("gt_boxes") else None
                gt_classes = gt_instances.gt_classes.tolist() if gt_instances is not None and gt_instances.has(
                    "gt_classes") else []

                pred_instances = out["instances"]
                pred_boxes = pred_instances.pred_boxes
                pred_classes = pred_instances.pred_classes.tolist()
                pred_scores = pred_instances.scores.tolist()

                # 1. Match GT and Preds
                if len(gt_classes) > 0 and len(pred_classes) > 0:
                    iou_matrix = pairwise_iou(gt_boxes.to(pred_boxes.device), pred_boxes).cpu().numpy()

                    matched_gt = set()
                    matched_pred = set()

                    while True:
                        if iou_matrix.size == 0 or iou_matrix.max() < 0.5: break
                        g_idx, p_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

                        rows.append({
                            "image": inp["file_name"],
                            "gt_class": gt_classes[g_idx], "pred_class": pred_classes[p_idx],
                            "score": pred_scores[p_idx],
                            "gx1": gt_boxes.tensor[g_idx, 0].item(), "gy1": gt_boxes.tensor[g_idx, 1].item(),
                            "gx2": gt_boxes.tensor[g_idx, 2].item(), "gy2": gt_boxes.tensor[g_idx, 3].item(),
                            "px1": pred_boxes.tensor[p_idx, 0].item(), "py1": pred_boxes.tensor[p_idx, 1].item(),
                            "px2": pred_boxes.tensor[p_idx, 2].item(), "py2": pred_boxes.tensor[p_idx, 3].item(),
                        })
                        matched_gt.add(g_idx)
                        matched_pred.add(p_idx)
                        iou_matrix[g_idx, :] = -1
                        iou_matrix[:, p_idx] = -1

                    # FN (Missed)
                    for i, cls in enumerate(gt_classes):
                        if i not in matched_gt:
                            rows.append({
                                "image": inp["file_name"], "gt_class": cls, "pred_class": -1, "score": -1,
                                "gx1": gt_boxes.tensor[i, 0].item(), "gy1": gt_boxes.tensor[i, 1].item(),
                                "gx2": gt_boxes.tensor[i, 2].item(), "gy2": gt_boxes.tensor[i, 3].item(),
                                "px1": -1, "py1": -1, "px2": -1, "py2": -1
                            })
                    # FP (False Pos)
                    for i, cls in enumerate(pred_classes):
                        if i not in matched_pred:
                            rows.append({
                                "image": inp["file_name"], "gt_class": -1, "pred_class": cls, "score": pred_scores[i],
                                "gx1": -1, "gy1": -1, "gx2": -1, "gy2": -1,
                                "px1": pred_boxes.tensor[i, 0].item(), "py1": pred_boxes.tensor[i, 1].item(),
                                "px2": pred_boxes.tensor[i, 2].item(), "py2": pred_boxes.tensor[i, 3].item(),
                            })

                # 2. Only GT (All FN)
                elif len(gt_classes) > 0:
                    for i, cls in enumerate(gt_classes):
                        rows.append({
                            "image": inp["file_name"], "gt_class": cls, "pred_class": -1, "score": -1,
                            "gx1": gt_boxes.tensor[i, 0].item(), "gy1": gt_boxes.tensor[i, 1].item(),
                            "gx2": gt_boxes.tensor[i, 2].item(), "gy2": gt_boxes.tensor[i, 3].item(),
                            "px1": -1, "py1": -1, "px2": -1, "py2": -1
                        })

                # 3. Only Preds (All FP)
                elif len(pred_classes) > 0:
                    for i, cls in enumerate(pred_classes):
                        rows.append({
                            "image": inp["file_name"], "gt_class": -1, "pred_class": cls, "score": pred_scores[i],
                            "gx1": -1, "gy1": -1, "gx2": -1, "gy2": -1,
                            "px1": pred_boxes.tensor[i, 0].item(), "py1": pred_boxes.tensor[i, 1].item(),
                            "px2": pred_boxes.tensor[i, 2].item(), "py2": pred_boxes.tensor[i, 3].item(),
                        })

    model.train(was_training)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def log_val_predictions_to_tensorboard(cfg, model, data_loader, tb_dir, max_images=8, global_step=0):
    was_training = model.training
    model.eval()

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
    model.train(was_training)


# ---------------------------------------------------------
# The Setup Function (ResNet101 + Res Normalization)
# ---------------------------------------------------------
def setup_and_train(output_dir, num_classes, args):
    cfg = get_cfg()

    # --- 1. ARCHITECTURE: ResNet-101 ---
    # Essential for distinguishing texture-less Fishing boats vs Rec boats
    if args.backbone == "r101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    else:
        # Fallback (but R101 is requested in CLI)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("Ship_Optical_train",)
    cfg.DATASETS.TEST = ("Ship_Optical_val",)
    cfg.DATALOADER.NUM_WORKERS = 16

    # --- 2. SOLVER (Calibrated for Batch 16) ---
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
    # Forces small images (600px) up to 900-1200px range.
    cfg.INPUT.MIN_SIZE_TRAIN = (900, 1000, 1100, 1200)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "none"

    # --- 4. RPN (Micro-Object Tuning) ---
    cfg.MODEL.RPN.NMS_THRESH = 0.8
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]

    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms

    cfg.EARLY_STOP = CN()
    cfg.EARLY_STOP.PATIENCE = 5
    cfg.OUTPUT_DIR = output_dir
    cfg.RARE_CLASS_IDS = [2, 4]
    cfg.OVERSAMPLE_FACTOR = 4
    cfg.KEEP_GT_IN_VAL = True
    cfg.TEST.ENABLE_AIR_EVAL = True
    cfg.TEST.ENABLE_TB_IMAGES = True

    os.makedirs(output_dir, exist_ok=True)
    logText(cfg, cfg.OUTPUT_DIR)

    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)  # Fresh start for R101

    try:
        trainer.train()
    except Exception as e:
        if str(e) == "EARLY_STOP":
            print("✅ Early stopping exit cleanly.")
        else:
            raise

    # Final Eval
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=ClassAwareMapper(cfg, is_train=False))
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, cfg.OUTPUT_DIR)
    inference_on_dataset(trainer.model, val_loader, evaluator)
    final_pred_csv = os.path.join(cfg.OUTPUT_DIR, "prediction_log.csv")
    dump_predictions_to_csv(trainer.model, val_loader, final_pred_csv)

    # Heatmap
    class_count_csv = os.path.join(cfg.OUTPUT_DIR, "batch_class_distribution.csv")
    if os.path.exists(class_count_csv):
        try:
            generate_class_distribution_heatmap(class_count_csv, cfg.OUTPUT_DIR)
        except Exception as e:
            pass

    # Confusion Matrix
    if os.path.exists(final_pred_csv):
        df = pd.read_csv(final_pred_csv)
        class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
        evaluator = MiniAirEvaluator(df, output_dir, class_names)
        evaluator.plot_canonical_confusion_matrix()
        evaluator.plot_precision_recall_f1_table()

    # Summary
    summary_csv = os.path.join(cfg.OUTPUT_DIR, "..", "sweep_summary.csv")
    summary_row = {
        "run_name": os.path.basename(cfg.OUTPUT_DIR),
        "LR": cfg.SOLVER.BASE_LR,
        "Batch": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        "RareClasses": cfg.RARE_CLASS_IDS,
    }
    df_summary = pd.DataFrame([summary_row])
    df_summary.to_csv(summary_csv, mode="a", header=not os.path.exists(summary_csv), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # R101 Defaults
    parser.add_argument("--backbone", type=str, default="r101")
    parser.add_argument("--batch", type=int, default=16)  # Safe A100 batch for high res
    parser.add_argument("--lr", type=float, default=0.002)  # Scaled for Batch 16
    parser.add_argument("--base-iters", type=int, default=60000)  # Deeper model = more iters

    parser.add_argument("--name", type=str, default="R101_ResNorm_Strategy")
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