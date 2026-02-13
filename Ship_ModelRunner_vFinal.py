import os, io, contextlib, random, sys
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
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, Instances

# Settings
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_noval/"
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/no_aumentation/"
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/safe/res50/"
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/boostAP_/"
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/"
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/clutterMixed_/"
MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/model_best.pth"
#SAR
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/"#singleRun_run_0/"

# Where to save the results
# OUTPUT_ROOT = "/home/gpaps/PycharmProject/Esa_Ships/Inference_result/SAR/KCP3_/run_0/"
OUTPUT_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/"

# JSON path
# JSON_PATH = "SAR/docker2go/json/coco_test.json"
# JSON_PATH = "Optical/docker2go/json/coco_test.json"
# JSON_PATH = "/home/gpaps/PycharmProject/Esa_Ships/Bench_sat/XML9/coco_sky.json"
# JSON_PATH = "/home/gpaps/PycharmProject/Esa_Ships/Bench_sat/XML9__Images_crop_256p/XML9satt_bench_OptShips_256p.json"
JSON_PATH = "/home/gpaps/PycharmProject/Esa_Ships/Bench_sat/satt_bench_OptShips.json"
# JSON_PATH = "/home/gpaps/PycharmProject/Esa_Ships/Bench_sarships/sar_ship.json"

# IMAGE_DIR = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/"
# IMAGE _DIR = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_SAR-Ship/HRSID-SSDD/"
# IMAGE_DIR = "/home/gpaps/PycharmProject/Esa_Ships/Bench_sarships/images/"
IMAGE_DIR = "/home/gpaps/PycharmProject/Esa_Ships/INFER_docker2go/dataset/skysat_2025_09_05_piraeus_tiles_500/"
# Comment/uncomment this section for Optical/SAR
CLASS_NAMES = ["Commercial", "Military", "Submarines", "Recreational Boats", "Fishing Boats"]
# CLASS_NAMES = ["ship"]


# Optimal Params for Inference across all models change these affects the output/performance of the model
CONFIDENCE_THRESHOLD = 0.60
IOU_THRESHOLD = 0.5

BATCH_SIZE = 8
FORCE_RUN = True
SWEEP_MODE = False
SHOW_GT_SIDE_BY_SIDE = True  # Set too False to only visualize predictions

MAX_IMAGES = 30000000  #
RANDOM_SEED = 42  # Set None for non-reproducible random runs

# Sweeps over these thresholds to optimize further the output!
SCORE_THRESHOLDS = [0.1]
NMS_THRESHOLDS = [0.7]


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
    vis_dir = os.path.join(output_dir, "vis3")
    os.makedirs(vis_dir, exist_ok=True)

    csv_out_path = os.path.join(output_dir, "prediction_log.csv")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = model_path


    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 4000  # default: 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 4000  # default: 1000

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
    # cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]  # or multiple levels
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    # cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 4
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]

    '''TODO BoostAP'''
    # empty placeholder

    '''TODO cluttered MIX enable'''
    # cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]

    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    USE_TTA = False
    if USE_TTA:
        from detectron2.modeling import build_model
        from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA

        model = build_model(cfg)
        model.eval()
        predictor = DefaultPredictor(cfg)  # to initialize metadata + transforms
        predictor.model = GeneralizedRCNNWithTTA(cfg, model)  # ✅ simple default TTA

    else:
        predictor = DefaultPredictor(cfg)

    predictions = []

    img_items = list(id_to_filename.items())

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(img_items)
    img_items = img_items[:MAX_IMAGES]
    selected_img_ids = [img_id for img_id, _ in img_items]

    for img_id, img_name in tqdm(img_items, desc="Running Inference"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path): continue
        im = cv2.imread(img_path)
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        pred_classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
        scores = instances.scores.numpy() if instances.has("scores") else []
        gt_ann = gt_annotations.get(img_id, [])
        matched_gt_ids = set()

        for i, (box, cls, score) in enumerate(zip(pred_boxes, pred_classes, scores)):
            if score < CONFIDENCE_THRESHOLD:
                continue

            best_iou, best_gt_idx, assigned_gt_class = 0, -1, -1
            for j, gt in enumerate(gt_ann):
                gt_box = BoxMode.convert(gt["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                iou = compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, j
                    assigned_gt_class = gt["category_id"]

            if best_iou >= IOU_THRESHOLD:
                matched_gt_ids.add(best_gt_idx)
                predictions.append({"image": img_name, "pred_class": int(cls), "score": float(score),
                                    "gt_class": int(assigned_gt_class), "iou": float(best_iou)})
            else:
                predictions.append({"image": img_name, "pred_class": int(cls), "score": float(score), "gt_class": -1,
                                    "iou": float(best_iou)})

        for idx, gt in enumerate(gt_ann):
            if idx not in matched_gt_ids:
                predictions.append(
                    {"image": img_name, "pred_class": -1, "score": 0.0, "gt_class": int(gt["category_id"]), "iou": 0.0})

        if len(gt_ann) == 0 and len(pred_boxes) == 0:
            predictions.append({
                "image": img_name,
                "pred_class": -1,
                "score": 0.0,
                "gt_class": -1,
                "iou": 0.0
            })

        custom_metadata = MetadataCatalog.get("__unused__")
        custom_metadata.thing_classes = CLASS_NAMES
        if SHOW_GT_SIDE_BY_SIDE:
            gt_vis = Visualizer(im[:, :, ::-1], metadata=custom_metadata, scale=1.0)
            gt_instances = []

            for gt in gt_ann:
                gt_box = BoxMode.convert(gt["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                gt_instances.append({
                    "bbox": gt_box,
                    "category_id": gt["category_id"]
                })

            if gt_instances:  # ONLY visualize if GT exists
                gt_vis_instances = Instances(im.shape[:2])
                gt_boxes_tensor = torch.tensor([g["bbox"] for g in gt_instances])
                gt_classes_tensor = torch.tensor([g["category_id"] for g in gt_instances])
                gt_vis_instances.pred_boxes = Boxes(gt_boxes_tensor)
                gt_vis_instances.pred_classes = gt_classes_tensor

                gt_labels = ["GT"] * len(gt_vis_instances)
                gt_vis = gt_vis.overlay_instances(
                    boxes=gt_vis_instances.pred_boxes,
                    labels=gt_labels,
                    assigned_colors=[(0.0, 0.0, 1.0)] * len(gt_vis_instances)
                )

                gt_image = gt_vis.get_image()
            else:
                gt_image = im[:, :, ::-1]  # fallback to raw image if no GT

        # -- Draw Predictions --
        pred_viz = Visualizer(im[:, :, ::-1], metadata=custom_metadata, scale=1.0)
        instances = outputs["instances"].to("cpu")
        instances = instances[instances.scores >= CONFIDENCE_THRESHOLD]
        pred_vis = pred_viz.draw_instance_predictions(instances)
        pred_image = pred_vis.get_image()

        # -- Save Visualization --
        if SHOW_GT_SIDE_BY_SIDE:
            gap = np.ones((im.shape[0], 5, 3), dtype=np.uint8) * 255
            combined = np.concatenate((gt_image, gap, pred_image), axis=1)
            cv2.imwrite(os.path.join(vis_dir, img_name), combined)
            # cv2.imwrite(os.path.join(vis_dir, img_name), combined[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(vis_dir, img_name), pred_image)

    df = pd.DataFrame(predictions)
    if "gt_class" in df.columns:
        df["gt_bool"] = df["gt_class"] != -1
        df["pred_bool"] = df["pred_class"] != -1
    else:
        print(" 'gt_class' column missing — skipping GT-specific logic")
        df["gt_bool"] = False
        df["pred_bool"] = False

    try:
        roc_auc = roc_auc_score(df["gt_bool"], df["score"])
    except:
        roc_auc = -1.0
    cm = confusion_matrix(df["gt_bool"], df["pred_bool"], labels=[False, True])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    df["roc_auc"] = roc_auc
    df["tnr"] = tnr
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Saved predictions to {csv_out_path}")
    USE_COCO_EVAL = True
    if USE_COCO_EVAL:
        dataset_name = f"Temp_Eval_{score_thresh}_{nms_thresh}".replace('.', '')

        register_coco_instances(dataset_name, {}, JSON_PATH, IMAGE_DIR)

        # ✅ Set class names and fix ID mapping
        MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_NAMES)

        # MetadataCatalog.get(dataset_name).thing_classes = CLASS_NAMES
        # MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id = {1: 0}
        # MetadataCatalog.get(dataset_name).thing_contiguous_id_to_dataset_id = {0: 1}

        # ✅ Load test records just for statistics (optional)
        test_records = load_coco_json(JSON_PATH, IMAGE_DIR)
        print(f"Loaded test set: {len(test_records)} images from {JSON_PATH}")

        # ✅ Category annotation stats display
        all_annos = [ann["category_id"] for rec in test_records for ann in rec.get("annotations", [])]
        anno_counts = Counter(all_annos)

        print("🔍 Annotations per class:")
        OFFSET = 1  # Only needed for printing — NOT for evaluator
        for cls_id, count in sorted(anno_counts.items()):
            cls_name = CLASS_NAMES[cls_id - OFFSET] if OFFSET <= cls_id < len(
                CLASS_NAMES) + OFFSET else f"Unknown({cls_id})"
            print(f"  - {cls_name}: {count} annotations")

        # ✅ COCO Evaluator
        coco_eval_dir = os.path.join(output_dir, "coco_eval")

        print(f"🔍 GT images in subset: {len(selected_img_ids)}")
        evaluator = COCOEvaluator(dataset_name, tasks=["bbox"], distributed=False, output_dir=coco_eval_dir)
        print(
            f" Total predicted instances: {sum(len(p['instances'].pred_classes) for p in predictions if 'instances' in p)}")

        # val_loader = build_detection_test_loader(cfg, dataset_name)
        def build_subset_loader(cfg, dataset_name, selected_img_ids):
            mapper = DatasetMapper(cfg, is_train=False)
            full_dataset = DatasetCatalog.get(dataset_name)
            subset = [d for d in full_dataset if d["image_id"] in selected_img_ids]
            # Apply mapper to each image dict
            mapped_subset = list(map(mapper, subset))
            return torch.utils.data.DataLoader(
                mapped_subset, batch_size=1, shuffle=False,
                collate_fn=trivial_batch_collator, num_workers=0
            )

        val_loader = build_subset_loader(cfg, dataset_name, selected_img_ids)

        log_path = os.path.join(coco_eval_dir, "coco_metrics_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f, contextlib.redirect_stdout(f):
            model_name = os.path.basename(output_dir.rstrip("/"))
            print(f" Model: {model_name}")
            print(f" Output Directory: {output_dir}\\n")
            print(" COCO Evaluation Results:")
            results = inference_on_dataset(predictor.model, val_loader, evaluator)
            for k, v in results.items():
                print(f"{k}: {v}")
            metrics_out = os.path.join(output_dir, "metrics.json")
            with open(metrics_out, "w") as jf:
                json.dump(results, jf, indent=2)
            print(f" Saved metrics to {metrics_out}")


if __name__ == "__main__":

    from glob import glob

    if SWEEP_MODE:
        # Recursively find all folders with model_final.pth or model_best.pth
        candidate_paths = glob(os.path.join(MODEL_ROOT, "**"), recursive=True)
        model_files = []

        for path in candidate_paths:
            final_path = os.path.join(path, "model_final.pth")
            best_path = os.path.join(path, "model_best.pth")

            if os.path.isfile(final_path):
                model_files.append(final_path)
            elif os.path.isfile(best_path):
                model_files.append(best_path)

        for model_file in model_files:
            model_folder = os.path.dirname(model_file)
            relative_folder = os.path.relpath(model_folder, MODEL_ROOT).replace(os.sep, "_")

            for score_thresh in SCORE_THRESHOLDS:
                for nms_thresh in NMS_THRESHOLDS:
                    run_dir = f"{relative_folder}_s{score_thresh}_n{nms_thresh}"
                    output_dir = os.path.join(OUTPUT_ROOT, run_dir)

                    start_time = time.perf_counter()
                    run_inference(model_file, output_dir, score_thresh, nms_thresh)
                    end_time = time.perf_counter()
                    duration = end_time - start_time

                    print(
                        f"Inference duration for sweep {relative_folder} | score={score_thresh}, nms={nms_thresh}: {duration:.2f} sec")
                    with open("sweep_timing_log.txt", "a") as f:
                        f.write(
                            f"{relative_folder} | score={score_thresh}, nms={nms_thresh} | time: {duration:.2f} sec\n")

        print("✅ All sweeps completed.")

        # Optional summary of best AP
        print("\n Scanning for best model based on COCO AP...")
        best_ap = -1.0
        best_model_info = ""
        metrics_paths = glob(os.path.join(OUTPUT_ROOT, "*/metrics.json"))

        for path in metrics_paths:
            with open(path, "r") as f:
                try:
                    metrics = json.load(f)
                    ap = metrics.get("bbox", {}).get("AP", -1)
                    model_name = os.path.basename(os.path.dirname(path))
                    print(f" {model_name}: AP = {ap:.2f}")
                    if ap > best_ap:
                        best_ap = ap
                        best_model_info = model_name
                except Exception as e:
                    print(f"❌ Error reading {path}: {e}")

        print(f"\n🏆 Best model based on COCO AP: {best_model_info} with AP = {best_ap:.2f}")
