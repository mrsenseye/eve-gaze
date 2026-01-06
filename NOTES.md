# Project Notes

## Goal

Build a strong single-task gaze estimation baseline on the EVE dataset before adding complexity (temporal modeling, multi-task learning, etc.).

The ultimate goal is accurate gaze direction prediction from face/eye images for eye-tracking applications.

## Current Status

**Baseline in training**: `gaze-singletask-v2`
- W&B: https://wandb.ai/seth-weisberg/eve-gaze/runs/qrdvsr3g
- Single-frame model (no temporal)
- 8x H200 GPUs, bf16-mixed precision

## File Overview

### Core Files

| File | Purpose |
|------|---------|
| `model_gaze_singletask.py` | Model architectures (GazeTRLite, GazeTRHybrid) |
| `eve_dataset.py` | EVE dataset loader with HDF5 reading and augmentations |
| `train_gaze_singletask.py` | PyTorch Lightning training script with W&B logging |

### model_gaze_singletask.py

Defines two model variants:

1. **GazeTRLite** (35.6M params) - Current baseline
   - 3 ResNet-18 encoders (face + left/right eyes)
   - 4-layer transformer for feature fusion
   - MLP head outputs pitch/yaw in radians

2. **GazeTRHybrid** (~100M params) - Higher capacity option
   - ViT-Base for face, ViT-Small for eyes
   - Same transformer fusion approach

### eve_dataset.py

- Loads pre-extracted crops from HDF5 files
- Expected structure: `/tmp/eve_dataset/{train,val}*/participant_*/frames_basler.h5`
- Each H5 file contains: `face`, `leye`, `reye` image arrays + `gaze` labels
- Augmentations: color jitter, blur, noise (training only)

### train_gaze_singletask.py

- PyTorch Lightning training loop
- Combined loss: Angular loss (cosine similarity) + MSE (0.1 weight)
- W&B logging with validation visualizations
- DDP multi-GPU support

## How to Use

### 1. Prepare Data

Extract EVE dataset to `/tmp/eve_dataset` with pre-cropped face/eye images in HDF5 format.

### 2. Train Baseline

```bash
python train_gaze_singletask.py \
    --model lite \
    --batch-size 64 \
    --max-epochs 30 \
    --run-name my-baseline
```

### 3. Monitor

- Check W&B dashboard for loss curves and angular error
- Validation visualizations show predicted (red) vs target (green) gaze vectors

### 4. Evaluate

Target: < 5° mean angular error on validation set.

## Next Steps

1. **Complete baseline training** - establish performance floor
2. **Temporal modeling** - add sequence_length=16 for video context
3. **Multi-task** - joint gaze + segmentation if beneficial

## Design Decisions

- **3 separate encoders** instead of single backbone: allows different input resolutions (224x224 face, 112x112 eyes) and lets the model learn eye-specific features
- **Transformer fusion** over simple concatenation: learns cross-attention between face context and eye details
- **Angular loss** as primary metric: directly optimizes the goal (gaze direction accuracy)
- **Single-frame first**: simpler to debug, establishes baseline before adding temporal complexity
