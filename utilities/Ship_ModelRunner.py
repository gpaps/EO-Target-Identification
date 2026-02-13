import os, io, contextlib
import json
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from collections import Counter
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ========== CONFIGURATION ==========
SWEEP_MODE = True
# MODEL_ROOT = "trained_models/_model_sweeps/"
MODEL_ROOT = "trained_models/SAR/cluster1/"

IMAGE_DIR = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_SAR-Ship/HRSID-SSDD/"
JSON_PATH = "../SAR/json/coco_test.json"

MODEL_PATH = "./trained_models/_model_sweeps/Infra_sweep_lr0.0002_b256/model_final.pth"
OUTPUT_DIR = "./Inference_result/Infra_sweep_lr0.0002_b256/"

CLASS_NAMES = ["ship"]  # ["storage tank", "bridge", "harbor"]
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.5
BATCH_SIZE = 8

FORCE_RUN = True
SAVE_VISUALS = False
SHIFT_CATEGORY_ID = True
USE_COCO_EVAL = True
VISUALIZE_CONFUSION = True


def compute_iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area != 0 else 0


def run_inference(model_path, output_dir):
    with open(JSON_PATH) as f:
        coco_data = json.load(f)

    gt_annotations = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        bbox = ann["bbox"]
        category_id = ann["category_id"] - 1 if SHIFT_CATEGORY_ID else ann["category_id"]
        if img_id not in gt_annotations:
            gt_annotations[img_id] = []
        gt_annotations[img_id].append({"bbox": bbox, "category_id": category_id})

    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    os.makedirs(output_dir, exist_ok=True)
    if SAVE_VISUALS:
        os.makedirs(os.path.join(output_dir, "vis"), exist_ok=True)

    csv_out_path = os.path.join(output_dir, "prediction_log.csv")
    if os.path.exists(csv_out_path) and not FORCE_RUN:
        print(f"{csv_out_path} already exists. Skipping inference.")
        return

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    predictor = DefaultPredictor(cfg)
    predictions = []

    img_items = list(id_to_filename.items())
    for batch_start in tqdm(range(0, len(img_items), BATCH_SIZE), desc="Running Inference"):
        batch = img_items[batch_start:batch_start + BATCH_SIZE]
        for img_id, img_name in batch:
            img_path = os.path.join(IMAGE_DIR, img_name)
            if not os.path.exists(img_path):
                continue
            im = cv2.imread(img_path)
            outputs = predictor(im)
            instances = outputs["instances"].to("cpu")
            pred_boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            pred_classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
            scores = instances.scores.numpy() if instances.has("scores") else []

            gt_ann = gt_annotations.get(img_id, [])
            matched_gt_ids = set()

            for box, cls, score in zip(pred_boxes, pred_classes, scores):
                best_iou, best_gt_idx, assigned_gt_class = 0, -1, -1
                for i, gt in enumerate(gt_ann):
                    gt_box = BoxMode.convert(gt["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                    iou = compute_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, i
                        assigned_gt_class = gt["category_id"]

                if best_iou >= IOU_THRESHOLD:
                    matched_gt_ids.add(best_gt_idx)
                predictions.append({
                    "image": img_name,
                    "pred_class": int(cls),
                    "score": float(score),
                    "gt_class": int(assigned_gt_class) if best_iou >= IOU_THRESHOLD else -1,
                    "iou": float(best_iou),
                })

            for idx, gt in enumerate(gt_ann):
                if idx not in matched_gt_ids:
                    predictions.append({
                        "image": img_name,
                        "pred_class": -1,
                        "score": 0.0,
                        "gt_class": int(gt["category_id"]),
                        "iou": 0.0
                    })

            if SAVE_VISUALS:
                vis_path = os.path.join(output_dir, "vis", img_name)
                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite(vis_path, v.get_image()[:, :, ::-1])

    pd.DataFrame(predictions).to_csv(csv_out_path, index=False)
    print(f"✅ Saved predictions to {csv_out_path}")

    if USE_COCO_EVAL:

        dataset_name = "TempInfraEval"
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)

        register_coco_instances(dataset_name, {}, JSON_PATH, IMAGE_DIR)

        # ✅ Print basic test set info
        test_records = load_coco_json(JSON_PATH, IMAGE_DIR)
        print(f"Loaded test set: {len(test_records)} images from {JSON_PATH}")

        # ✅ Assign class names and print per-class annotation count
        MetadataCatalog.get(dataset_name).thing_classes = CLASS_NAMES
        class_names = CLASS_NAMES
        all_annos = [ann["category_id"] for rec in test_records for ann in rec.get("annotations", [])]
        anno_counts = Counter(all_annos)

        print("🔍 Annotations per class:")
        for cls_id, count in sorted(anno_counts.items()):
            OFFSET = 1
            cls_name = class_names[cls_id - OFFSET] if OFFSET <= cls_id < len(class_names) + OFFSET else f"Unknown({cls_id})"
            # cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown({cls_id})"
            print(f"  - {cls_name}: {count} annotations")

        # ✅ Setup evaluator and test loader
        evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=os.path.join(output_dir, "coco_eval"))
        val_loader = build_detection_test_loader(cfg, dataset_name)

        # ✅ Evaluate and log
        log_path = os.path.join(output_dir, "coco_eval", "coco_metrics_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f, contextlib.redirect_stdout(f):
            model_name = os.path.basename(output_dir.rstrip("/"))
            print(f"📄 Model: {model_name}")
            print(f"📂 Output Directory: {output_dir}\n")
            print("📊 COCO Evaluation Results:")

            results = inference_on_dataset(predictor.model, val_loader, evaluator)
            for k, v in results.items():
                print(f"{k}: {v}")

            metrics_out = os.path.join(output_dir, "metrics.json")
            with open(metrics_out, "w") as jf:
                json.dump(results, jf, indent=2)
            print(f"📦 Saved metrics to {metrics_out}")


if __name__ == "__main__":
    if SWEEP_MODE:
        model_dirs = [d for d in os.listdir(MODEL_ROOT) if os.path.isdir(os.path.join(MODEL_ROOT, d))]
        for model_folder in model_dirs:
            model_path = os.path.join(MODEL_ROOT, model_folder, "model_final.pth")
            if not os.path.exists(model_path):
                continue
            output_dir = os.path.join("./Inference_result/SAR/cluster1/", model_folder)
            run_inference(model_path, output_dir)
        print("\n✅ Sweep Inference Completed!")
    else:
        run_inference(MODEL_PATH, OUTPUT_DIR)
