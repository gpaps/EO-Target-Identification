import os
import cv2
import json
import logging
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import glob

from pathlib import Path

import config
from Data_Ingestion_s3_SAR import data_ingestion


def run():
    ###
    # # ----------  TILE THE SCENE (v2 tiler) ----------
    from infer_utils.sar_quicklook_and_tiles_v2 import tile_image  # adjust import per your tree
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.TILES_DIR).mkdir(parents=True, exist_ok=True)
    Path(Path(config.output_dir) / config.VIS_SUBDIR).mkdir(parents=True, exist_ok=True)

    IMAGE_DIR   = config.TILES_DIR
    CLASS_NAMES = config.CLASSES

    tile_image(
        in_tif=config.SCENE_TIF,
        out_dir=config.TILES_DIR,
        tile=config.TILE,
        stride=config.STRIDE,
        bands=None if not config.BANDS else [int(b) for b in config.BANDS.split(",")],
        fmt="png",
        csv_manifest=config.MANIFEST_CSV,
        smooth=config.SMOOTH,
    )

    # ==== CONFIGURATION ====
    IMAGE_DIR   = config.TILES_DIR
    OUTPUT_DIR  = config.output_dir
    CLASS_NAMES = config.CLASSES
    FOLDER_NAME = config.VIS_SUBDIR
    SAVE_VISUALS = True
    CONFIDENCE_THRESHOLD = 0.60


    # ==== SETUP ====
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_VISUALS:
        os.makedirs(os.path.join(OUTPUT_DIR, FOLDER_NAME), exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = config.MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO to change this when you ARE NOT USING ROI In test runs
    # cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    # cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]] * 5

    # TODO threshold
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("__unused__")
    metadata.thing_classes = CLASS_NAMES

    # ==== INFERENCE LOOP ====
    predictions = []
    image_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".tiff"))]

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
            vis_path = os.path.join(OUTPUT_DIR, FOLDER_NAME, img_name)
            cv2.imwrite(vis_path, vis_output.get_image()[:, :, ::-1])

    # ==== SAVE CSV ====
    df = pd.DataFrame(predictions)
    csv_path = str(Path(config.output_dir) / config.CSV_NAME)
    df.to_csv(csv_path, index=False)
    logging.info(f"Inference complete. Results saved to: {csv_path}")

    # TODO to optimize the this test script.
    from post.export_geojson_from_csv import export_geojson

    pd.DataFrame(predictions).to_csv(csv_path, index=False)

    # Geo export
    from post.export_geojson_from_csv import export_geojson
    geojson_out = str(Path(config.output_dir) / "predictions.geojson")
    metrics_out = str(Path(config.output_dir) / "metrics.json")
    export_geojson(
        scene_tif=str(config.SCENE_TIF),
        detections_csv=csv_path,
        manifest_csv=config.MANIFEST_CSV,
        out_geojson=geojson_out,
        out_metrics_json=metrics_out,
        xml_path=config.XML_PATH or None,
    )

    # XXX stac_meta wants all the results in the CWD. It also deletes all the existing files in the CWD before creating new ones.
    # This is only ok when running in a container in a safe folder or using CWL (or cwltool) which is running in a temp folder.
    # To avoid deleting project files we check for existing py files in the CWD and abort if any are found.
    existing_py_files = glob.glob(os.path.join(config.STAC_FOLDER, '*.py'))
    if existing_py_files:
        logging.error(f"Existing .py files found in the {config.STAC_FOLDER}. Aborting stac_metadata to avoid deleting/overwriting files.")
    else:
        # Stac META data
        from post import STAC_Metadatav2TT
        STAC_Metadatav2TT.stac_metadata()    
     

    
if __name__ == "__main__":
    
    os.makedirs(config.log_path, exist_ok=True)

    # Prepare Logger
    os.makedirs(config.log_path, exist_ok=True)
    log_filename = datetime.now().strftime("detector_%Y%m%d_%H%M%S.log")
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(config.log_path+log_filename),
                            logging.StreamHandler()
                            ]
                        )
    
    # Parse CLI args
    config.parse_args()
    
    # pull data from s3
    # XXX Files are very large. For testing just check if they are already downloaded
    if not all((Path(config.download_dir) / f).exists() for f in config.S3_INPUT_FILES):
        logging.info("Downloading data from S3...")
        data_ingestion()
        logging.info("Download complete.")
    else:
        logging.info("Input files already exist, skipping download.")
    
    # run inference
    run()
    
