#!/usr/bin/env bash

# Config: 50k iters (Approx 40 Epochs for 10k images)
# Batch 8 (Per GPU - Adjust if using multi-gpu)
#BASE_ITERS=50000

echo "=================================================================="
echo " LAUNCH: SATELLITE SHIP DETECTION"
echo "=================================================================="

# --- RUN 1: Standard ResNet50 ---
#python3 train_OptShips_Hybrid.py \
#  --name R50_Satellite_Opt_v1 \
#  --backbone r50 \
#  --batch 32 \
#  --lr 0.004
#  --base-iters $BASE_ITERS \
#  --force
# --- RUN 2: Standard ResNet101 ---
#python3 train_OptShips_Hybrid.py \
#  --name R101_ResNorm_40k \
#  --backbone r101 \
#  --batch 16 \
#  --lr 0.002 \
#  --base-iters 40000 \
#  --force
# --- RUN 3: Standard ResNet101 ---
#python3 train_OptShips_Hybrid.py \
#  --name R101_DenseRPN_1024ROI \
#  --backbone r101 \
#  --batch 16 \
#  --lr 0.002 \
#  --base-iters 40000 \
#  --force

#python3 train_OptShips_Hybrid.py \
#  --name R101_Final_with_Background_RareCL0 \
#  --backbone r101 \
#  --batch 16 \
#  --lr 0.002 \
#  --base-iters 40000 \
#  --force
python3 train_OptShips_Hybrid.py \
  --name R101_Dense_Hybrid0_Bg_CIoU \
  --backbone r101 \
  --batch 16 \
  --lr 0.002 \
  --base-iters 40000 \
  --force
