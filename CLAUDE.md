# EVE Gaze Estimation

## Overview

Single-task gaze estimation on the EVE dataset. Predicts gaze direction (pitch/yaw) from face and eye crops using transformer-based fusion.

## Environment

- Python: `/media/nas1/procrustes/conda_envs/mamba/bin/python`
- Fast storage: `/tmp` (tmpfs RAM disk - use for training data)
- Dataset: `/tmp/eve_dataset` (HDF5 files with pre-extracted crops)
- GPUs: 8x H200

## Commands

```bash
# Train baseline (single-frame)
python train_gaze_singletask.py --model lite --batch-size 64 --max-epochs 30 --run-name NAME

# Train with temporal modeling
python train_gaze_singletask.py --model lite --sequence-length 16 --batch-size 32 --run-name NAME

# Check GPU usage
nvidia-smi
```

## Architecture

**GazeTRLite** (35.6M params):
- 3 ResNet-18 encoders: face (224x224), left eye (112x112), right eye (112x112)
- 4-layer transformer fusion
- MLP head → pitch/yaw (radians)

**Loss**: Angular loss (cosine similarity) + 0.1 * MSE

**Target metric**: < 5° mean angular error

## Key Files

| File | Purpose |
|------|---------|
| `train_gaze_singletask.py` | Training script with PyTorch Lightning + W&B |
| `model_gaze_singletask.py` | GazeTRLite and GazeTRHybrid model definitions |
| `eve_dataset.py` | EVE dataset loader, reads HDF5, applies augmentations |

## Data Format

EVE dataset structure:
```
/tmp/eve_dataset/
├── train01/participant_001/frames_basler.h5
├── train02/participant_002/frames_basler.h5
└── val01/participant_XXX/frames_basler.h5
```

Each HDF5 contains:
- `face`: (N, 224, 224, 3) uint8
- `leye`: (N, 112, 112, 3) uint8
- `reye`: (N, 112, 112, 3) uint8
- `gaze`: (N, 2) float32 - pitch/yaw in radians

## Experiment Tracking

- Log all runs to W&B project `eve-gaze`
- Save checkpoints with metric in filename: `epoch{N}-val_angular_error{X.XX}.ckpt`
- Document runs in @NOTES.md

## Coding Style

- PyTorch Lightning for training loops
- timm for pretrained backbones
- albumentations for augmentation
- Type hints on function signatures
- Keep model code in `model_*.py`, training in `train_*.py`
