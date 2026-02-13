#!/bin/bash

# --- Configuration ---
REMOTE_USER=gpaps
REMOTE_HOST=139.91.185.102
REMOTE_PATH="/home/gpaps/esa-train/Ships_/Optical/trained_models_/"
LOCAL_DEST="/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/"

#LOCAL_DEST=trained_models_home/cluster1

# --- Sync selective files while preserving directory structure ---
echo "Starting rsync from $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

rsync -av --include='*/' \
          --include='*.pth' \
          --include='metrics.json' \
          --include='class_distribution_heatmap.png' \
          --include='*.csv' \
          --include='*.png' \
          --include='*.json' \
          --include='coco_metrics_log.txt' \
          --include='tensorboard/*' \
          --include='last_checkpoint' \
          --include='instances_predictions.pth' \
          --include='coco_instances_results' \
          --exclude='*' \
          "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
          "$LOCAL_DEST"

echo "Done! Synced selected files into: $LOCAL_DEST"
