# SAR Object Detection Training Pipeline (Detectron2)

This repository provides modular, high-quality training scripts for object detection on SAR imagery, specifically for:
- 🛩 **SAR Aircraft Detection**
- 🚢 **SAR Ship Detection**

Both use Detectron2's Faster R-CNN (with Cascade ROI support for Aircraft), advanced evaluation, and 
training enhancements such as early stopping, sweep compatibility, and rich TensorBoard logging.

---

## 📁 Structure Overview

```bash
├── trainv4_SARAircraft_sweeps_v2_cleaned.py
├── trainv4_SARShips_sweeps_v2.py
├── utils/
│   └── tensorboard_utils.py
├── run_sweep_2.sh
├── json/
│   └── coco_train.json / coco_val.json
└── trained_models_SAR/

Features:

✅ Single-Class Training with configurable class name
✅ Detectron2-based model (Faster R-CNN + Cascade ROI Heads)
✅ Early Stopping based on AP stagnation
✅ Mid-training evaluation with:

    COCO AP, AP50, AP75

    Confusion Matrix (visual)

    Precision / Recall / F1 table

✅ Final evaluation on validation set
✅ TensorBoard Logging for:

    Training losses

    Final predictions

    Config parameters (text)

✅ Sweep-compatible CLI with:

    Learning Rate (--lr)

    Batch Size (--batch)

    NMS Threshold (--nms)

    Score Threshold (--score)

    Run Name (--name)

Training Script Example:

python3 trainv4_SARShips_sweeps_v2.py \
  --lr 0.0001 \
  --batch 512 \
  --nms 0.3 \
  --score 0.5 \
  --name sweep_lr0.0001_b512_nms0.3_score0.5

Sweep Script (example)

bash run_sweep_2.sh

Modifies learning rate, batch size, NMS, and score thresholds in combinations (24 total by default).
utils/tensorboard_utils.py:

Contains reusable logging utilities:

    log_val_predictions_to_tensorboard(...): logs annotated images
    log_image_to_tensorboard(...): logs arbitrary RGB images
    logText(...): dumps full training config into TensorBoard

Early Stopping:
Controlled via config;

cfg.EARLY_STOP = CN()
cfg.EARLY_STOP.PATIENCE = 2
When no AP improvement occurs for patience evaluations, training halts gracefully.

Output Artifacts:

    metrics.json: COCO metrics per iteration
    confusion_matrix.png: Mid-train and final
    precision_recall_table.png: Per-class F1 and PR
    tensorboard/: Scalars, visual preds, config text

Notes:

    Uses Detectron2's build_detection_test_loader with custom mappers
    Fully modular for reuse across Optical or SAR use cases
    Built to run on cluster (multi-GPU compatible)

Contact:
For questions or support, reach out to George Papadopoulos (FORTH, ICS).
gpaps@ics.forth.gr

-