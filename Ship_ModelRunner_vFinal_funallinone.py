import os, io, contextlib, random, sys, time
from glob import glob
import json
import torch
import detectron2
import pandas as pd
import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import time

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, DatasetFromList, MapDataset
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, Instances
# from utilities.Vehi_ModelRunner import JSON_PATH  # not strictly needed but kept
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

# Version-proof collate: list[dict] -> list[dict]
try:
    from detectron2.data.common import trivial_batch_collator as d2_collate
except Exception:
    def d2_collate(batch):
        return batch

# TRAIN_ROOT= "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirOpt/test/Onlythis/fine_tune/Optical_sweep_2default/"
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirOpt/test/Onlythis/fine_tune/Optical_sweep_2default/"
# TRAIN_ROOT= "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirOpt/test/Onlythis/fine_tune/Optical_sweep_2default/"
TRAIN_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirOpt/test/Onlythis/fine_tune/Optical_sweep_2default/"
MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirOpt/test/Onlythis/fine_tune/Optical_sweep_2default/"
OUTPUT_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/images/output_pred_bench_5/"
JSON_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/images/temp256/opt_Air.json"
IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/images/temp256/"

# ========= Centralized CONFIG SAR =========
# TRAIN_ROOT  = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december/finetune_base_v5_lr0.0002_b512_v21/"
# MODEL_ROOT  = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december/finetune_base_v5_lr0.0002_b512_v21/"
# OUTPUT_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december/finetune_base_v5_lr0.0002_b512_v21/output_2/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Dataset-SAR
# JSON_PATH   = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218/annotations_corrected/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218.json"
# IMAGE_DIR   = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X25_GRD_SLEDF_5243129_20250708T234218/tiles_3072/"

#  ========= Centralized CONFIG Optical =========
# TRAIN_ROOT  = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/"
# MODEL_ROOT  = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/"
# OUTPUT_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/fine_tune768/"
# os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Dataset-Optical
# IMAGE_DIR   = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/temp768/"
# JSON_PATH   = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/temp768/opt_ships.json"

CLASS_NAMES = ["ship"]
# CLASS_NAMES = ["Commercial", "Military", "Submarines", "Recreational Boats", "Fishing Boats"]
# CLASS_NAMES = ["aircraft", "helicopter"]

# Inference settings
BATCH_SIZE = 4
FORCE_RUN = False
SWEEP_MODE = False
SHOW_GT_SIDE_BY_SIDE = True

# If MAX_IMAGES <= 0 → use all images in the dataset for COCO eval
MAX_IMAGES = 100
RANDOM_SEED = 42

# Threshold grids for sweeps
# Low score threshold so COCO can build a proper PR curve.
SCORE_THRESHOLDS = [0.60]  # , 0.1, 0.3]
# Reasonable NMS IOU for dense scenes
NMS_THRESHOLDS = [0.3]
# IOU used for our custom TP/FP/FN accounting (not COCO)
IOU_THRESHOLD = 0.6
# IOU used for per-class PR / confusion stats
METRIC_IOU = 0.5

# Use full dataset for COCO eval (set False if you want subset-only for debug)
FULL_COCO_EVAL = True

# ---- Collect weights once (prefer model_best.pth, else model_final.pth) ----
_cands = glob(os.path.join(MODEL_ROOT, "**", "model_best.pth"), recursive=True) \
         + glob(os.path.join(MODEL_ROOT, "**", "model_final.pth"), recursive=True)
_run_to_weight = {}
for _w in _cands:
    _d = os.path.dirname(_w)
    if _d not in _run_to_weight or _w.endswith("model_best.pth"):
        _run_to_weight[_d] = _w
WEIGHT_FILES = list(_run_to_weight.values())
print(f"[config] MODEL_ROOT='{MODEL_ROOT}' | discovered runs={len(WEIGHT_FILES)}")


# =====================================


# --- best-model scanner (handles nested JSON + JSON-lines) ---
def scan_and_report_best(output_root: str):
    print("\n Scanning for best model based on COCO AP...")

    def extract_ap(path: str):
        """
        Read AP from a metrics.json that may be:
        - single JSON object with nested keys: {"bbox": {"AP": ..., "AP50": ...}}
        - JSON-lines (Detectron2 training): many lines with flat keys: {"bbox/AP": ...}
        Returns float AP or None.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.read(1)
                f.seek(0)

                # Case A: try single JSON object
                if first == "{":
                    try:
                        data = json.load(f)
                        if isinstance(data.get("bbox"), dict) and "AP" in data["bbox"]:
                            return float(data["bbox"]["AP"])
                        if "bbox/AP" in data:
                            return float(data["bbox/AP"])
                    except json.JSONDecodeError:
                        f.seek(0)

                # Case B: JSON-lines (training)
                ap = None
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if "bbox/AP" in rec:
                        ap = float(rec["bbox/AP"])
                return ap
        except Exception:
            return None

    metrics_paths = glob(os.path.join(output_root, "**", "metrics.json"), recursive=True)
    if not metrics_paths:
        print(f" No metrics.json files under: {output_root}")
        return

    best_ap, best_model = -1.0, None
    for p in metrics_paths:
        ap = extract_ap(p)
        run_dir = os.path.dirname(p)
        name = os.path.basename(run_dir)
        if ap is None:
            print(f" {name}: AP = n/a")
            continue
        print(f" {name}: AP = {ap:.2f}")
        if ap > best_ap:
            best_ap, best_model = ap, run_dir

    if best_model is None:
        print(f" No valid metrics parsed under: {output_root}")
        return

    print(f"\n Best model based on COCO AP: {os.path.basename(best_model)} with AP = {best_ap:.2f}")


def trivial_batch_collator(batch):
    return batch


def compute_iou(boxA, boxB):
    x1, y1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    x2, y2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area != 0 else 0


def compute_detection_stats(df: pd.DataFrame, num_classes: int, iou_thresh: float, out_dir: str):
    """
    Compute per-class TP/FP/FN, precision, recall, F1 and a (num_classes+1)x(num_classes+1)
    confusion matrix (last index = background).

    NOTE:
      - gt_class == -1 → background (no GT object)
      - pred_class == -1 → no prediction
      - We treat a GT as "correctly detected for class c" if:
          gt_class == c AND pred_class == c AND iou >= iou_thresh
      - Everything else counts as FP / FN accordingly.
    """
    os.makedirs(out_dir, exist_ok=True)
    eval_rows = df.copy()

    # Keep only rows that have at least GT or prediction
    eval_rows = eval_rows[(eval_rows["gt_class"] >= 0) | (eval_rows["pred_class"] >= 0)]

    stats = []
    for c in range(num_classes):
        is_gt_c = eval_rows["gt_class"] == c
        is_pred_c = eval_rows["pred_class"] == c

        # True positives for class c
        good_match = is_gt_c & is_pred_c & (eval_rows["iou"] >= iou_thresh)
        tp = int(good_match.sum())

        # Any prediction of c that is not a good_match is FP for c
        fp = int((is_pred_c & ~good_match).sum())

        # Any GT of c that is not a good_match is FN for c
        fn = int((is_gt_c & ~good_match).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        stats.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    # Micro-averaged overall stats
    total_tp = sum(s["tp"] for s in stats)
    total_fp = sum(s["fp"] for s in stats)
    total_fn = sum(s["fn"] for s in stats)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    # Multi-class confusion matrix (including background as last class index)
    def _map_class(x):
        return x if x >= 0 else num_classes

    gt_labels = eval_rows["gt_class"].apply(_map_class)
    pred_labels = eval_rows["pred_class"].apply(_map_class)
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(num_classes + 1)))

    # Save stats as CSV
    stats_df = pd.DataFrame(stats + [{
        "class_id": -1,
        "class_name": "micro_avg",
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": micro_prec,
        "recall": micro_rec,
        "f1": micro_f1,
    }])
    stats_csv = os.path.join(out_dir, "detection_stats_per_class.csv")
    stats_df.to_csv(stats_csv, index=False)

    # Save confusion matrix as CSV (last row/col = background)
    cm_csv = os.path.join(out_dir, "confusion_matrix_classes_plus_bg.csv")
    cm_df = pd.DataFrame(cm, index=[*CLASS_NAMES, "bg"], columns=[*CLASS_NAMES, "bg"])
    cm_df.to_csv(cm_csv, index_label="gt \\ pred")

    print(f"Saved per-class stats to {stats_csv}")
    print(f"Saved confusion matrix to {cm_csv}")
    print("Per-class F1:")
    for s in stats:
        print(f"  {s['class_name']}: F1={s['f1']:.3f} (P={s['precision']:.3f}, R={s['recall']:.3f})")
    print(f"Micro F1={micro_f1:.3f} (P={micro_prec:.3f}, R={micro_rec:.3f})")


def run_inference(model_path, output_dir, score_thresh, nms_thresh):
    with open(JSON_PATH) as f:
        coco_data = json.load(f)

    gt_annotations = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        bbox = ann["bbox"]
        cat_id = ann["category_id"] - 1
        gt_annotations.setdefault(img_id, []).append({"bbox": bbox, "category_id": cat_id})
    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    csv_out_path = os.path.join(output_dir, "prediction_log.csv")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = model_path

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
    # Finetune params.
    # example that gives 3 anchors/location:
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] * 5
    # 1 size * 3 aspect ratios = 3 anchors/location

    # cfg.MODEL.ANCHOR_GENERATOR.SIZES         = [[8, 12], [12, 16], [16, 24], [24, 32], [32, 48]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [12], [16], [24], [32]] # VANILLA worked
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]] * 5
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    # cfg.MODEL.RPN.IN_FEATURES   = ["p2", "p3", "p4", "p5", "p6"]
    # cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    # cfg.TEST.DETECTIONS_PER_IMAGE = 400
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    predictor = DefaultPredictor(cfg)
    predictions = []

    img_items = list(id_to_filename.items())
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(img_items)
    # Only limit if MAX_IMAGES > 0; otherwise use all
    if MAX_IMAGES is not None and MAX_IMAGES > 0:
        img_items = img_items[:MAX_IMAGES]
    selected_img_ids = [img_id for img_id, _ in img_items]

    # --- Build a batched inference loader from the selected items ---
    records = [{"image_id": img_id, "file_name": os.path.join(IMAGE_DIR, img_name)}
               for img_id, img_name in img_items]

    mapper = DatasetMapper(cfg, is_train=False)
    mapped = MapDataset(DatasetFromList(records, copy=False), mapper)
    infer_loader = torch.utils.data.DataLoader(
        mapped,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=max(8, getattr(cfg.DATALOADER, "NUM_WORKERS", 0)),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=d2_collate,
    )

    print(f"[loader] batch_size={getattr(infer_loader, 'batch_size', 'n/a')}, "
          f"num_workers={infer_loader.num_workers}, dataset_len={len(mapped)}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    first_batch = True
    total_imgs = 0

    for batch in tqdm(infer_loader, desc="Running Inference (batched)"):
        if first_batch:
            print(f"[runtime] first batch len={len(batch)}  (expected <= {BATCH_SIZE})")
            assert 0 < len(batch) <= BATCH_SIZE
            first_batch = False

        with torch.no_grad():
            outputs = predictor.model(batch)
            total_imgs += len(batch)

        for inp, out in zip(batch, outputs):
            img_id = inp["image_id"]
            img_path = inp["file_name"]
            if not os.path.exists(img_path):
                continue
            im = cv2.imread(img_path)

            instances = out["instances"].to("cpu")
            pred_boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            pred_classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
            scores = instances.scores.numpy() if instances.has("scores") else []

            gt_ann = gt_annotations.get(img_id, [])
            matched_gt_ids = set()

            # ---- match predictions to GT (simple best-IoU per prediction) ----
            for box, cls, score in zip(pred_boxes, pred_classes, scores):
                best_iou, best_gt_idx, assigned_gt_class = 0.0, -1, -1
                for j, gt in enumerate(gt_ann):
                    gt_box = BoxMode.convert(gt["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                    iou = compute_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, j
                        assigned_gt_class = gt["category_id"]

                if best_iou >= IOU_THRESHOLD and best_gt_idx >= 0:
                    matched_gt_ids.add(best_gt_idx)
                    predictions.append({
                        "image": os.path.basename(img_path),
                        "pred_class": int(cls),
                        "score": float(score),
                        "gt_class": int(assigned_gt_class),
                        "iou": float(best_iou),
                    })
                else:
                    predictions.append({
                        "image": os.path.basename(img_path),
                        "pred_class": int(cls),
                        "score": float(score),
                        "gt_class": -1,
                        "iou": float(best_iou),
                    })

            # mark missed GT
            for idx, gt in enumerate(gt_ann):
                if idx not in matched_gt_ids:
                    predictions.append({
                        "image": os.path.basename(img_path),
                        "pred_class": -1,
                        "score": 0.0,
                        "gt_class": int(gt["category_id"]),
                        "iou": 0.0,
                    })

            # no-pred & no-gt case
            if len(gt_ann) == 0 and len(pred_boxes) == 0:
                predictions.append({
                    "image": os.path.basename(img_path),
                    "pred_class": -1,
                    "score": 0.0,
                    "gt_class": -1,
                    "iou": 0.0,
                })

            # ---- Visualization ----
            custom_metadata = MetadataCatalog.get("__unused__")
            custom_metadata.thing_classes = CLASS_NAMES

            if SHOW_GT_SIDE_BY_SIDE:
                gt_instances = []
                for gt in gt_ann:
                    gt_box = BoxMode.convert(gt["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                    gt_instances.append({"bbox": gt_box, "category_id": gt["category_id"]})
                if gt_instances:
                    gt_vis = Visualizer(im[:, :, ::-1], metadata=custom_metadata, scale=1.0)
                    gt_vis_instances = Instances(im.shape[:2])
                    gt_vis_instances.pred_boxes = Boxes(torch.tensor([g["bbox"] for g in gt_instances]))
                    gt_vis_instances.pred_classes = torch.tensor([g["category_id"] for g in gt_instances])
                    gt_image = gt_vis.overlay_instances(
                        boxes=gt_vis_instances.pred_boxes,
                        labels=["GT"] * len(gt_vis_instances),
                        assigned_colors=[(0.0, 0.0, 1.0)] * len(gt_vis_instances),
                    ).get_image()
                else:
                    gt_image = im[:, :, ::-1]

            # pred_viz = Visualizer(im[:, :, ::-1], metadata=custom_metadata, scale=1.0)
            pred_viz = Visualizer(im, metadata=custom_metadata, scale=1.0)

            pred_image = pred_viz.draw_instance_predictions(instances).get_image()
            out_path = os.path.join(vis_dir, os.path.basename(img_path))
            if SHOW_GT_SIDE_BY_SIDE:
                gap = np.ones((im.shape[0], 5, 3), dtype=np.uint8) * 255
                combined = np.concatenate((gt_image, gap, pred_image), axis=1)
                cv2.imwrite(out_path, combined)
            else:
                cv2.imwrite(out_path, pred_image)

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(f"[summary] processed={total_imgs} imgs  | time={elapsed:.1f}s  | "
          f"avg={total_imgs / elapsed:.2f} img/s  | peakVRAM={peak_gb:.2f} GB  | batch={BATCH_SIZE}")

    df = pd.DataFrame(predictions)

    if df.empty:
        print(" No predictions/GT rows collected. Writing empty prediction_log.csv and skipping binary metrics.")
        df.to_csv(csv_out_path, index=False)
        print(f" Saved EMPTY predictions to {csv_out_path}")
    else:
        if "gt_class" in df.columns:
            df["gt_bool"] = df["gt_class"] != -1
            df["pred_bool"] = df["pred_class"] != -1
        else:
            print(" 'gt_class' column missing — skipping GT-specific logic")
            df["gt_bool"] = False
            df["pred_bool"] = False

        try:
            roc_auc = roc_auc_score(df["gt_bool"], df["score"])
        except Exception as e:
            print(f"⚠ Could not compute ROC-AUC: {e}")
            roc_auc = -1.0

        try:
            cm_bin = confusion_matrix(df["gt_bool"], df["pred_bool"], labels=[False, True])
            if cm_bin.size == 4:
                tn, fp, fn, tp = cm_bin.ravel()
            else:
                tn, fp, fn, tp = (0, 0, 0, 0)
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        except Exception as e:
            print(f" Could not compute binary confusion matrix: {e}")
            tnr = 0.0

        df["roc_auc"] = roc_auc
        df["tnr"] = tnr
        df.to_csv(csv_out_path, index=False)
        print(f"✅ Saved predictions to {csv_out_path}")

    try:
        if os.path.getsize(csv_out_path) == 0:
            print(" Total predicted instances: 0 (CSV empty)")
        else:
            df_pred = pd.read_csv(csv_out_path)
            print(f" Total predicted instances: {len(df_pred)}")
    except pd.errors.EmptyDataError:
        print(" Total predicted instances: 0 (CSV had no rows)")
    except Exception as e:
        print(f" Total predicted instances: n/a (could not read CSV: {e})")

    # --- Extra: per-class PR/F1 + confusion matrix ---
    compute_detection_stats(df, num_classes=len(CLASS_NAMES), iou_thresh=METRIC_IOU, out_dir=output_dir)

    # --- COCO eval ---
    USE_COCO_EVAL = True
    if USE_COCO_EVAL:
        dataset_name = "Temp_Eval"
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.pop(dataset_name, None)

        register_coco_instances(dataset_name, {}, JSON_PATH, IMAGE_DIR)
        MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_NAMES)

        test_records = load_coco_json(JSON_PATH, IMAGE_DIR)
        print(f"Loaded test set: {len(test_records)} images from {JSON_PATH}")

        all_annos = [ann["category_id"] for rec in test_records for ann in rec.get("annotations", [])]
        anno_counts = Counter(all_annos)
        print("🔍 Annotations per class:")
        OFFSET = 1
        for cls_id, count in sorted(anno_counts.items()):
            cls_name = CLASS_NAMES[cls_id - OFFSET] if OFFSET <= cls_id < len(
                CLASS_NAMES) + OFFSET else f"Unknown({cls_id})"
            print(f"  - {cls_name}: {count} annotations")

        coco_eval_dir = os.path.join(output_dir, "coco_eval")
        print(f"🔍 GT images in subset: {len(selected_img_ids)}")

        def build_subset_loader(cfg, dataset_name, selected_img_ids, batch_size=None):
            bs = batch_size
            full_dataset = DatasetCatalog.get(dataset_name)
            subset = [d for d in full_dataset if d.get("image_id") in selected_img_ids]

            mapper = DatasetMapper(cfg, is_train=False)
            mapped = MapDataset(DatasetFromList(subset, copy=False), mapper)

            return torch.utils.data.DataLoader(
                mapped,
                batch_size=bs,
                shuffle=False,
                num_workers=max(8, getattr(cfg.DATALOADER, "NUM_WORKERS", 0)),
                pin_memory=True,
                persistent_workers=True,
                collate_fn=d2_collate,
            )

        if FULL_COCO_EVAL or MAX_IMAGES is None or MAX_IMAGES <= 0:
            val_loader = build_detection_test_loader(cfg, dataset_name)
        else:
            val_loader = build_subset_loader(cfg, dataset_name, selected_img_ids, batch_size=BATCH_SIZE)

        evaluator = COCOEvaluator(dataset_name, tasks=["bbox"], distributed=False, output_dir=coco_eval_dir)

        log_path = os.path.join(coco_eval_dir, "coco_metrics_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f, contextlib.redirect_stdout(f):
            model_name = os.path.basename(output_dir.rstrip("/"))
            print(f" Model: {model_name}")
            print(f" Output Directory: {output_dir}\n")
            print(" COCO Evaluation Results:")
            results = inference_on_dataset(predictor.model, val_loader, evaluator)
            for k, v in results.items():
                print(f"{k}: {v}")
            metrics_out = os.path.join(output_dir, "metrics.json")
            with open(metrics_out, "w") as jf:
                json.dump(results, jf, indent=2)
            print(f" Saved metrics to {metrics_out}")


if __name__ == "__main__":

    if SWEEP_MODE:
        if not WEIGHT_FILES:
            print(f"❌ No model_best.pth or model_final.pth found under: {MODEL_ROOT}")
            sys.exit(0)
        else:
            print(f"🔎 Found {len(WEIGHT_FILES)} weight files.")

        for w in WEIGHT_FILES:
            model_folder = os.path.dirname(w)
            model_name = os.path.basename(model_folder)
            for score_thresh in SCORE_THRESHOLDS:
                for nms_thresh in NMS_THRESHOLDS:
                    run_dir = f"{model_name}_s{score_thresh}_n{nms_thresh}"
                    output_dir = os.path.join(OUTPUT_ROOT, run_dir)
                    os.makedirs(output_dir, exist_ok=True)

                    print(f"→ Running {model_name} | s={score_thresh}, n={nms_thresh} → {output_dir}")
                    t0 = time.perf_counter()
                    run_inference(w, output_dir, score_thresh, nms_thresh)
                    print(f"Inference duration for sweep {model_name} | score={score_thresh}, "
                          f"nms={nms_thresh}: {time.perf_counter() - t0:.2f} sec")

        print("All sweeps completed.")
        print("[infer] Scanning inference outputs…")
        scan_and_report_best(OUTPUT_ROOT)
        print("[train] Scanning training runs…")
        scan_and_report_best(TRAIN_ROOT)

    else:
        candidate_paths = glob(os.path.join(MODEL_ROOT, "**"), recursive=True)
        model_file = None
        for path in candidate_paths:
            best_p = os.path.join(path, "model_best.pth")
            final_p = os.path.join(path, "model_final.pth")
            if os.path.isfile(best_p):
                model_file = best_p
                break
            if os.path.isfile(final_p) and model_file is None:
                model_file = final_p
        if not model_file:
            print(f" No model_best.pth or model_final.pth found under {MODEL_ROOT}")
            sys.exit(1)

        score_thresh = SCORE_THRESHOLDS[0]
        nms_thresh = NMS_THRESHOLDS[0]

        model_folder = os.path.dirname(model_file)
        relative_folder = os.path.relpath(model_folder, MODEL_ROOT).replace(os.sep, "_")
        run_dir = f"{relative_folder}_s{score_thresh}_n{nms_thresh}"
        output_dir = os.path.join(OUTPUT_ROOT, run_dir)

        start_time = time.perf_counter()
        run_inference(model_file, output_dir, score_thresh, nms_thresh)
        duration = time.perf_counter() - start_time
        print(f"Inference duration (single run) {relative_folder} | score={score_thresh}, "
              f"nms={nms_thresh}: {duration:.2f} sec")

        scan_and_report_best(OUTPUT_ROOT)
