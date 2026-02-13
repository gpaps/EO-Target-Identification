
# ESA Infrastructure & SAR Ships Training Pipeline - Summary

**Date:** 2025-05-07

---

## 🐳 Docker & Cluster Setup

### Actions:
- Built `docker-compose.yml` and `Dockerfile` for each use case (Infra and Ships).
- Ran containers on specific GPUs:
  ```bash
  CUDA_DEVICES=<GPU_ID> docker compose up -d
  CUDA_DEVICES=1 docker compose up -d
  docker exec -it SAR_AIR_model_trainer bash


  ```

- Ensured unique container names to avoid conflicts.
- Removed old containers:
  ```bash
  docker rm -f model_trainer
  docker rm -f sar_ship_model_trainer
  ```

### TensorBoard Access:
- Started server:
  ```bash
  ~/.local/bin/tensorboard --logdir trained_models_home/ --port 6010 --host 0.0.0.0
  ```
- Accessed remotely:
  ```bash
  ssh -L 6010:localhost:6010 gpaps@139.91.185.102
  ```

---

## 🏗️ Infrastructure Use Case

### Key Scripts:
- `trainv4_Infra_sweeps_v2.py`: Stable and used for production.
- `train_sweep.py`: Initially promising, but led to gradient explosion and NaNs.

### Fixes:
- Dropped `ClassAwareMapper` from Ships where it was unnecessary.
- Used gradient clipping:
  ```python
  cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
  cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
  cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
  ```

### Training Params:
- `MAX_ITER`: 30000
- `STEPS`: (18000, 25000)
- `BASE_LR`: 0.0002
- `BATCH_SIZE_PER_IMAGE`: 256
- `MIN_SIZE_TRAIN`: [640, 800, 1024, 1280]

---

## 🚢 SAR Ships Use Case

### Script:
- `trainv4_Ships_sweeps_v2.py` with `ShipSARMapper`

### Augmentations:
- ResizeShortestEdge
- Brightness, contrast
- Horizontal and vertical flip
- Random rotation

### Fixes:
- Replaced `ClassAwareMapper` (1-class only)
- Handled StopIteration via proper dataset mapping

### Params:
- `MAX_ITER`: 15000
- `STEPS`: (8000, 12000)
- `BASE_LR`: 0.0001
- `BATCH_SIZE_PER_IMAGE`: 256

---

## 🧠 Additional Enhancements

- Added TensorBoard text logging via:
  ```python
  logText(cfg, cfg.OUTPUT_DIR)
  ```
- Removed in-script dataset registration
- Evaluation hooks added for val metrics and output visual logging
- Avoided clutter: controlled logging output, model summaries

---

## 🧰 Handy Commands

```bash
# Run sweep
./run_sweep.sh

# Remove stuck containers
docker rm -f model_trainer
docker rm -f sar_ship_model_trainer

# TensorBoard from remote
~/.local/bin/tensorboard --logdir trained_models_home/ --port 6010 --host 0.0.0.0
ssh -L 6010:localhost:6010 gpaps@139.91.185.102
```

---

✅ You’re ready for repeatable, scalable training across use cases. Next up: monitoring and inference!
