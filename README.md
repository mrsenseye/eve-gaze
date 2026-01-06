# EVE Gaze Estimation

Single-task gaze estimation models trained on the [EVE Dataset](https://ait.ethz.ch/eve).

## Architecture

**GazeTRLite**: A transformer-based gaze estimation model using face and eye crops.

- **Face encoder**: ResNet-18 (224x224 input)
- **Eye encoders**: ResNet-18 x2 (112x112 input for left/right eyes)
- **Fusion**: Concatenate features → 4-layer transformer → MLP head
- **Output**: Gaze direction as pitch/yaw (radians)
- **Parameters**: 35.6M

```
Face (224x224) ──→ ResNet-18 ──→ [512-dim]
                                    ↓
Left Eye (112x112) → ResNet-18 → [512-dim] → Concat → Transformer → MLP → [pitch, yaw]
                                    ↑
Right Eye (112x112) → ResNet-18 → [512-dim]
```

## Training

### Prerequisites

```bash
pip install torch torchvision pytorch-lightning timm wandb albumentations opencv-python
```

### Dataset

Download the [EVE Dataset](https://ait.ethz.ch/eve) and extract to `/tmp/eve_dataset` (or specify with `--eve-root`).

### Train

```bash
# Single-frame baseline (recommended first)
python train_gaze_singletask.py \
    --model lite \
    --batch-size 64 \
    --max-epochs 30 \
    --run-name gaze-baseline-v1

# With temporal modeling (after baseline)
python train_gaze_singletask.py \
    --model lite \
    --sequence-length 16 \
    --batch-size 32 \
    --max-epochs 30 \
    --run-name gaze-temporal-v1
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `lite` | Model type: `lite` (35M) or `full` (100M) |
| `--eve-root` | `/tmp/eve_dataset` | Path to EVE dataset |
| `--camera` | `basler` | Camera: `basler` (60fps) or `c` (webcam 30fps) |
| `--batch-size` | `64` | Batch size per GPU |
| `--sequence-length` | `1` | Frames per sample (1=single frame) |
| `--max-epochs` | `30` | Maximum training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--gpus` | `8` | Number of GPUs |

## Loss Function

Combined loss:
- **Angular Loss**: Cosine similarity between predicted and target 3D gaze vectors
- **MSE Loss**: Mean squared error on pitch/yaw values (0.1 weight)

## Metrics

- **Angular Error (degrees)**: Angle between predicted and target gaze vectors
- **Target**: < 5° angular error

## Model Variants

| Model | Backbone | Params | Notes |
|-------|----------|--------|-------|
| `lite` | ResNet-18 | 35.6M | Fast, good baseline |
| `full` | ViT-Base + ViT-Small | ~100M | Higher capacity |

## Files

- `train_gaze_singletask.py` - Training script with W&B logging
- `model_gaze_singletask.py` - GazeTRLite and GazeTRHybrid architectures
- `eve_dataset.py` - EVE dataset loader with augmentation

## Citation

If using the EVE dataset:
```bibtex
@inproceedings{park2020eve,
  title={Towards End-to-end Video-based Eye-Tracking},
  author={Park, Seonwook and Aksan, Emre and Zhang, Xucong and Hilliges, Otmar},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
