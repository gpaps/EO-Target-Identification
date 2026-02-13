import os, cv2, json, torch
import pandas as pd
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# ==== CONFIGURATION ====
# $sar
IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/sar/ships/_outputs_v2/ICEYE_X6_GRD_SLED_4410498_20241216T014014/tiles_2048/"
MODEL_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december/finetune_base_v5_lr0.0002_b512_v21/model_final.pth"
# MODEL_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_SAR_sweep_v5_lr0.0002_b512/model_final.pth"
OUTPUT_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december/finetune_base_v5_lr0.0002_b512_v21/"
# $optical
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/1km_cropped_data/1km_cropped_optical_data/Piraeus_Optical/"
# MODEL_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/NOT_INFERED___/Optical_sweep_lr0.00015_b1024/model_best.pth"
# OUTPUT_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/NOT_INFERED___/Optical_sweep_lr0.00015_b1024/"
# IMAGE_DIR = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/00_Benchmark_validation/Optical_1km_Cropped/_all_territory/"
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/VHRShips_ShipRSImageNEt/"
# IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/"
# MODEL_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/model_best.pth"

# MODEL_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/safe/res50/Optical_sweep_lr0.0001_nms0.3_r50/model_best.pth"
# OUTPUT_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/"
# OUTPUT_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/ships/"
# CLASS_NAMES = ["Commercial", "Military", "Submarines", "Recreational Boats", "Fishing Boats"]
CLASS_NAMES = ["ship"]
CONFIDENCE_THRESHOLD = 0.75
SAVE_VISUALS = True

# ==== SETUP ====
os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_VISUALS:
    os.makedirs(os.path.join(OUTPUT_DIR, "vis_x6"), exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]] * 5
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get("__unused__")
metadata.thing_classes = CLASS_NAMES

# ==== INFERENCE LOOP ====
predictions = []
image_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".tiff", ".bmp"))]

for img_name in tqdm(image_list, desc="Predicting"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    im = cv2.imread(img_path)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []

    for box, cls, score in zip(boxes, classes, scores):
        predictions.append({
            "image": img_name,
            "pred_class": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Unknown({cls})",
            "score": float(score),
            "bbox_x1": float(box[0]),
            "bbox_y1": float(box[1]),
            "bbox_x2": float(box[2]),
            "bbox_y2": float(box[3]),
        })

    if SAVE_VISUALS:
        vis = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_output = vis.draw_instance_predictions(instances)
        vis_path = os.path.join(OUTPUT_DIR, "vis", img_name)
        cv2.imwrite(vis_path, vis_output.get_image()[:, :, ::-1])

# ==== SAVE CSV ====
df = pd.DataFrame(predictions)
csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Inference complete. Results saved to: {csv_path}")
