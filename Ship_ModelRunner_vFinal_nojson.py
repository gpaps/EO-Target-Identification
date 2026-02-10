import os, io, contextlib, random, sys, time, json
from glob import glob
from collections import Counter

import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, DatasetFromList, MapDataset
from detectron2 import model_zoo

from Ship_ModelRunner_vFinal import OUTPUT_ROOT, IMAGE_DIR
# from Ship_ModelRunner_vFinal_funallinone import TRAIN_ROOT
from utilities.Ship_ModelRunner_bestmodel_v1 import CLASS_NAMES, MODEL_ROOT

# --------- Version-proof collate ----------
try:
    from detectron2.data.common import trivial_batch_collator as d2_collate
except Exception:
    def d2_collate(batch):
        return batch

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

# ========= Centralized CONFIG =========
# Where your training runs live (for training AP scan)
# Ships
# TRAIN_ROOT  = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/"

# TRAIN_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/SAR_sweep_v5_lr0.0002_b512_nms0.4_score0.4/"
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/SAR_sweep_v5_lr0.0002_b512_nms0.4_score0.4/"
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X49_GRD_SLEDP_6667842_20251021T125916/8048x8048/ICEYE_X49_GRD_SLEDP_6667842_20251021T125916/tiles_8048/"
# OUTPUT_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X49_GRD_SLEDP_6667842_20251021T125916/8048x8048/ICEYE_X49_GRD_SLEDP_6667842_20251021T125916/pred_run0_/"

# TRAIN_ROOT = '/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirSAR/run_9/fine_tune/SAR_sweep_default/'
# MODEL_ROOT = '/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirSAR/run_9/fine_tune/SAR_sweep_default/'
# IMAGE_DIR = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/aircrafts/_outputs_v2/ICEYE_X49_GRD_SLF_5222656_20250708T093706/tiles_2560/'
# OUTPUT_ROOT = '/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/aircrafts/_outputs_v2/ICEYE_X49_GRD_SLF_5222656_20250708T093706/output_v2/'
# os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Airplanes
# TRAIN_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirOpt/test/Onlythis/Optical_sweep_lr0.0001_b512_r50/"

# ''''Cars''''
# TRAIN_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/xView_640/xview_res50_640/Optical_sweep_r50_lr0002_b512/"
# TRAIN_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/xView_640/extension_40/Optical_sweep_cont40k_r50_lr0002_b512_cont40k/"
# MODEL_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/xView_640/xview_res50_640/Optical_sweep_r50_lr0002_b512/"
# MODEL_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/xView_640/extension_40/Optical_sweep_cont40k_r50_lr0002_b512_cont40k/"
# TRAIN_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/3cl/xview_res50_640/Optical_Final_R50_HighDensity_LongAnchors_40k/"
# MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/3cl/xview_res50_640/Optical_Final_R50_HighDensity_LongAnchors_40k/"
# IMAGE_DIR = '/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/3cl/xview_res50_640/Optical_Final_R50_HighDensity_LongAnchors_40k/SKYsat_Herackleion/640x512/20251222_064159_ssc1d1_0010_basic_analytic/tiles_640/'
# OUTPUT_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/VehiOpt/3cl/xview_res50_640/Optical_Final_R50_HighDensity_LongAnchors_40k/SKYsat_Herackleion/Prediction/"

'''Optical Ships'''
TRAIN_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/superdataset/Optical_R101_DenseRPN_1024ROI/"
MODEL_ROOT = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/superdataset/Optical_R101_DenseRPN_1024ROI/model_best.pth"
IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/final_inference/Heraklion/Optical/Optical_Heraklion_skysatscene_basic_analytic_udm2_20251222/patches_SkySatScene/test/20251222_064159_ssc1d1_0010_basic_analytic/"
OUTPUT_ROOT ="/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/superdataset/Optical_R101_DenseRPN_1024ROI/Prediction/"

# IMAGE_DIR = '/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/Peiraeus640x512/pansharpened/tiles_640/' Athens640x640
# Where to look for candidate checkpoints (early-stopped runs are inside here)
# MODEL_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/"
# MODEL_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/AirOpt/test/Onlythis/Optical_sweep_lr0.0001_b512_r50/"


# Where to write inference results
# OUTPUT_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/optical/piraeus_skysat_mosaic/raw/_outputs_640x480/skysat_2025_09_05_piraeus/pred/"
# OUTPUT_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/_outputs_640_640/pansharpened_thessaloniki/Air_pred_640x640/"

# Cars
# OUTPUT_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/_outputs_640_640/pansharpened_thessaloniki/Cars_pred_640x640/"
# OUTPUT_ROOT = r"/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/_outputs_768_768/pansharpened_thessaloniki/CarT_pred_768x768/"
# os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Dataset I/O (COCO mode)
# JSON_PATH = "Xview_Stride640_Json/coco_test.json"
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/geotiff_test/optical/piraeus_skysat_mosaic/raw/_outputs_640x480/skysat_2025_09_05_piraeus/tiles_640/"
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/_outputs_640_640/pansharpened_thessaloniki/tiles_640/"
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/inference_data/optical_inference_data/pansharpened/_outputs_768_768/pansharpened_thessaloniki/tiles_768/"
# Cars


# Pure image inference (no annotations)
INFER_FROM_IMAGES = True  # <<< set True to ignore JSON and run on raw images
IM_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# CLASS_NAMES = ["ship"]
# CLASS_NAMES = ["Car", "Truck", 'Bus']
CLASS_NAMES = ["Commercial Vessel", "Military", "Submarines", "Recreational Boat", "Fishing Boat"]
# CLASS_NAMES = ["aircraft", "helicopter"]

# Inference settings
BATCH_SIZE = 4
FORCE_RUN = True
SWEEP_MODE = True
SHOW_GT_SIDE_BY_SIDE = False  # no GT in pure-image mode; force False
MAX_IMAGES = 22000
RANDOM_SEED = 42

# Threshold grids for sweeps
SCORE_THRESHOLDS = [0.85]
NMS_THRESHOLDS = [0.5]

# ---- Collect weights (prefer model_best.pth, else model_final.pth) ----
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

def scan_and_report_best(output_root: str):
    print("\n Scanning for best model based on COCO AP... (only if metrics.json exist)")

    def extract_ap(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.read(1);
                f.seek(0)
                if first == "{":
                    try:
                        data = json.load(f)
                        if isinstance(data.get("bbox"), dict) and "AP" in data["bbox"]:
                            return float(data["bbox"]["AP"])
                        if "bbox/AP" in data:
                            return float(data["bbox/AP"])
                    except json.JSONDecodeError:
                        f.seek(0)
                ap = None
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if "bbox/AP" in rec: ap = float(rec["bbox/AP"])
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


def build_cfg(model_path, score_thresh, nms_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    # your small-object tweaks preserved
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] * 5
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 12], [12, 16], [16, 24], [24, 32], [32, 48]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    # cfg.MODEL.RPN.IN_FEATURES  = ["p2", "p3", "p4", "p5", "p6"]
    # cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    # cfg.TEST.DETECTIONS_PER_IMAGE = 400
    ###
    # cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    # cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True  # Required for Cascade
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    # cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 4  # detect tiny aircraft
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    ###
    # cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    # Slightly denser, still reasonable instead of the two rows below
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 12, 16, 24, 32]] # broadcast to all levels #(expect more compute/VRAM)
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 12], [12, 16], [16, 24], [24, 32],[32, 48]]  #anchors/pos/level = 2*3 = 6 everywhere
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [12], [16], [24], [32]] # VANILLA worked
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 12], [12, 16], [16, 24], [24, 32]]#, [32, 48]]  #anchors/pos/level = 2*3 = 6 everywhere

    #TODO for CARS
    # cfg.INPUT.MIN_SIZE_TEST = 640
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 12], [12, 16], [16, 32], [32, 64]]
    # --- 2. CLEANUP (The Secret Sauce) ---
    # We found that 0.50 cuts the pavement noise perfectly.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    # --- 3. DENSITY (Optional Tweak) ---
    # You can try raising this to 0.65 to see if it helps parking lots,
    # even though the model wasn't explicitly trained for it.
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # Default (Safe)
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.65 # Experimental (Dense)
    # --- 4. CAPACITY ---
    # Ensure you don't cap detections too low if testing on a parking lot
    # cfg.TEST.DETECTIONS_PER_IMAGE = 3000

   # TODO FOR OPTICAL SHIPS
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24], [32, 48], [64, 96], [128, 192], [256, 384]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]


    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 12], [12, 16], [16, 32], [32, 64]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    # Allow the pipeline to carry thousands of boxes
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 8000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 6000
    # cfg.TEST.DETECTIONS_PER_IMAGE = 3000
    ###
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg


def list_images_from_dir(img_dir, max_images=None, seed=42):
    files = []
    for root, _, fns in os.walk(img_dir):
        for fn in fns:
            if fn.lower().endswith(IM_EXTS):
                files.append(os.path.join(root, fn))
    if seed is not None:
        random.seed(seed)
        random.shuffle(files)
    if max_images is not None:
        files = files[:max_images]
    # Assign incremental image_ids (Detectron-style)
    records = []
    for i, fp in enumerate(files, start=1):
        records.append({"image_id": i, "file_name": fp})
    return records


def list_images_from_json(json_path, image_root, max_images=None, seed=42):
    with open(json_path, "r") as f:
        coco = json.load(f)
    id_to_file = {img["id"]: os.path.join(image_root, img["file_name"]) for img in coco["images"]}
    items = list(id_to_file.items())
    if seed is not None:
        random.seed(seed)
        random.shuffle(items)
    if max_images is not None:
        items = items[:max_images]
    records = [{"image_id": img_id, "file_name": fpath} for img_id, fpath in items]
    return records


def run_inference(model_path, output_dir, score_thresh, nms_thresh):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    csv_out_path = os.path.join(output_dir, "prediction_log.csv")

    # --- Build model ---
    cfg = build_cfg(model_path, score_thresh, nms_thresh)
    predictor = DefaultPredictor(cfg)

    # --- Build records (pure images or COCO) ---
    if INFER_FROM_IMAGES:
        records = list_images_from_dir(IMAGE_DIR, max_images=MAX_IMAGES, seed=RANDOM_SEED)
        selected_img_ids = [r["image_id"] for r in records]
        print(f"[input] Pure-image mode | found {len(records)} images under {IMAGE_DIR}")
    else:
        records = list_images_from_json(JSON_PATH, IMAGE_DIR, max_images=MAX_IMAGES, seed=RANDOM_SEED)
        selected_img_ids = [r["image_id"] for r in records]
        print(f"[input] COCO-JSON mode | selected {len(records)} images from {JSON_PATH}")

    # --- Build DataLoader (batched) ---
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

    # --- Inference ---
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    first_batch = True
    total_imgs = 0

    rows = []  # per-detection rows
    meta = MetadataCatalog.get("__unused__");
    meta.thing_classes = CLASS_NAMES

    for batch in tqdm(infer_loader, desc="Running Inference (batched)"):
        if first_batch:
            print(f"[runtime] first batch len={len(batch)}")
            assert 0 < len(batch) <= BATCH_SIZE
            first_batch = False

        with torch.no_grad():
            outputs = predictor.model(batch)
            total_imgs += len(batch)

        for inp, out in zip(batch, outputs):
            img_path = inp["file_name"]
            im = cv2.imread(img_path)
            if im is None:
                continue

            instances = out["instances"].to("cpu")
            pred_boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.zeros((0, 4))
            pred_classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros((0,),
                                                                                                         dtype=int)
            scores = instances.scores.numpy() if instances.has("scores") else np.zeros((0,), dtype=float)

            # Save detections row-wise (XYXY)
            for (x1, y1, x2, y2), cls_id, sc in zip(pred_boxes, pred_classes, scores):
                cls_id = int(cls_id)
                rows.append({
                    "image": os.path.basename(img_path),
                    "image_path": img_path,
                    "cls_name": CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}",
                    "cls_id": cls_id,
                    "score": float(sc),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                })

            # Visualization
            # vis = Visualizer(im[:, :, ::-1], metadata=meta, scale=1.0)
            vis = Visualizer(im, metadata=meta, scale=1.0)
            pred_image = vis.draw_instance_predictions(instances).get_image()
            out_path = os.path.join(vis_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, pred_image)

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(f"[summary] processed={total_imgs} imgs  | time={elapsed:.1f}s  | "
          f"avg={total_imgs / max(elapsed, 1e-6):.2f} img/s  | peakVRAM={peak_gb:.2f} GB  | batch={BATCH_SIZE}")

    # --- Write CSV ---
    df = pd.DataFrame(rows, columns=["image", "image_path", "cls_name", "cls_id", "score", "x1", "y1", "x2", "y2"])
    df.to_csv(csv_out_path, index=False)
    print(f" Saved detections to {csv_out_path} | rows={len(df)}")

    # --- COCO eval only when we actually used JSON/GT ---
    if not INFER_FROM_IMAGES:
        try:
            from detectron2.evaluation import COCOEvaluator, inference_on_dataset
            from detectron2.data.datasets import register_coco_instances, load_coco_json

            dataset_name = "Temp_Eval_NoJSON"
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
                MetadataCatalog.pop(dataset_name, None)

            register_coco_instances(dataset_name, {}, JSON_PATH, IMAGE_DIR)
            MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_NAMES)

            # build subset loader matching selected_img_ids
            def build_subset_loader(cfg, dataset_name, selected_img_ids, batch_size=None):
                full_dataset = DatasetCatalog.get(dataset_name)
                subset = [d for d in full_dataset if d.get("image_id") in selected_img_ids]
                mapped_local = MapDataset(DatasetFromList(subset, copy=False), DatasetMapper(cfg, is_train=False))
                return torch.utils.data.DataLoader(
                    mapped_local, batch_size=batch_size, shuffle=False,
                    num_workers=max(8, getattr(cfg.DATALOADER, "NUM_WORKERS", 0)),
                    pin_memory=True, persistent_workers=True, collate_fn=d2_collate)

            val_loader = build_subset_loader(cfg, dataset_name, selected_img_ids, batch_size=BATCH_SIZE)
            coco_eval_dir = os.path.join(output_dir, "coco_eval")
            os.makedirs(coco_eval_dir, exist_ok=True)
            log_path = os.path.join(coco_eval_dir, "coco_metrics_log.txt")

            with open(log_path, "w") as f, contextlib.redirect_stdout(f):
                model_name = os.path.basename(output_dir.rstrip("/"))
                print(f" Model: {model_name}")
                print(f" Output Directory: {output_dir}\n")
                print(" COCO Evaluation Results:")
                evaluator = COCOEvaluator(dataset_name, tasks=["bbox"], distributed=False, output_dir=coco_eval_dir)
                results = inference_on_dataset(predictor.model, val_loader, evaluator)
                for k, v in results.items():
                    print(f"{k}: {v}")
                metrics_out = os.path.join(output_dir, "metrics.json")
                with open(metrics_out, "w") as jf:
                    json.dump(results, jf, indent=2)
                print(f" Saved metrics to {metrics_out}")
        except Exception as e:
            print(f"[warn] COCO eval skipped due to error: {e}")


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
                    run_dir = f"{model_name}_s{score_thresh}_n{nms_thresh}" + (
                        "_imgs" if INFER_FROM_IMAGES else "_coco")
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
        # single-run mode (first found weight)
        candidate_paths = glob(os.path.join(MODEL_ROOT, "**"), recursive=True)
        model_file = None
        for path in candidate_paths:
            best_p = os.path.join(path, "model_best.pth")
            final_p = os.path.join(path, "model_final.pth")
            if os.path.isfile(best_p):
                model_file = best_p;
                break
            if os.path.isfile(final_p) and model_file is None:
                model_file = final_p
        if not model_file:
            print(f"❌ No model_best.pth or model_final.pth found under {MODEL_ROOT}")
            sys.exit(1)

        score_thresh = SCORE_THRESHOLDS[0]
        nms_thresh = NMS_THRESHOLDS[0]

        model_folder = os.path.dirname(model_file)
        relative_folder = os.path.relpath(model_folder, MODEL_ROOT).replace(os.sep, "_")
        run_dir = f"{relative_folder}_s{score_thresh}_n{nms_thresh}" + ("_imgs" if INFER_FROM_IMAGES else "_coco")
        output_dir = os.path.join(OUTPUT_ROOT, run_dir)

        start_time = time.perf_counter()
        run_inference(model_file, output_dir, score_thresh, nms_thresh)
        duration = time.perf_counter() - start_time
        print(
            f"Inference duration (single run) {relative_folder} | score={score_thresh}, nms={nms_thresh}: {duration:.2f} sec")
