"""
Single-Task Gaze Estimation Training on EVE Dataset

Trains GazeTR-style models (face + eyes -> gaze pitch/yaw) on EVE data.
Target: <5° angular error on validation.

Usage:
    python train_gaze_singletask.py --model lite --batch-size 64 --max-epochs 30
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
import cv2

from eve_dataset import EVEDataset
from model_gaze_singletask import GazeTRHybrid, GazeTRLite, AngularLoss


def draw_gaze_on_face(face_img, gaze_pred, gaze_target, img_size=224):
    """Draw predicted and target gaze vectors on face image.

    Args:
        face_img: (3, H, W) tensor, normalized to [0,1]
        gaze_pred: (2,) pitch, yaw in radians
        gaze_target: (2,) pitch, yaw in radians
        img_size: image size
    Returns:
        (H, W, 3) numpy array with gaze vectors drawn
    """
    # Denormalize image
    img = face_img.cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.5 + 0.5) * 255  # Assuming normalized with mean=0.5, std=0.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Center of face
    cx, cy = img_size // 2, img_size // 2

    # Draw gaze vectors (length proportional to angle)
    scale = 80

    # Target gaze (green)
    pitch_t, yaw_t = gaze_target[0].item(), gaze_target[1].item()
    dx_t = -scale * np.sin(yaw_t) * np.cos(pitch_t)
    dy_t = -scale * np.sin(pitch_t)
    cv2.arrowedLine(img, (cx, cy), (int(cx + dx_t), int(cy + dy_t)), (0, 255, 0), 2, tipLength=0.3)

    # Predicted gaze (red)
    pitch_p, yaw_p = gaze_pred[0].item(), gaze_pred[1].item()
    dx_p = -scale * np.sin(yaw_p) * np.cos(pitch_p)
    dy_p = -scale * np.sin(pitch_p)
    cv2.arrowedLine(img, (cx, cy), (int(cx + dx_p), int(cy + dy_p)), (0, 0, 255), 2, tipLength=0.3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class GazeVisualizationCallback(Callback):
    """Log gaze prediction visualizations to W&B."""

    def __init__(self, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        self.val_samples = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Only collect from first few batches on rank 0
        if batch_idx < 2 and trainer.global_rank == 0:
            frames = batch['frames']
            face = frames['face']
            gaze_target = batch['gaze']

            with torch.no_grad():
                gaze_pred = pl_module(face, frames['left'], frames['right'])

            # Store samples
            for i in range(min(4, face.shape[0])):
                if len(self.val_samples) < self.num_samples:
                    self.val_samples.append({
                        'face': face[i].cpu(),
                        'gaze_pred': gaze_pred[i].cpu(),
                        'gaze_target': gaze_target[i].cpu(),
                    })

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0 or not self.val_samples:
            self.val_samples = []
            return

        # Create visualization grid
        images = []
        for sample in self.val_samples:
            img = draw_gaze_on_face(sample['face'], sample['gaze_pred'], sample['gaze_target'])

            # Compute angular error for this sample
            pred_vec = pl_module._pitchyaw_to_vector(sample['gaze_pred'].unsqueeze(0))
            target_vec = pl_module._pitchyaw_to_vector(sample['gaze_target'].unsqueeze(0))
            pred_vec = F.normalize(pred_vec, dim=-1)
            target_vec = F.normalize(target_vec, dim=-1)
            cos_sim = (pred_vec * target_vec).sum(dim=-1).clamp(-0.99999, 0.99999)
            error_deg = (torch.acos(cos_sim) * 180.0 / np.pi).item()

            # Add error text
            cv2.putText(img, f'{error_deg:.1f}°', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            images.append(wandb.Image(img, caption=f'Error: {error_deg:.1f}°'))

        # Log to W&B
        trainer.logger.experiment.log({
            'val/gaze_predictions': images,
            'global_step': trainer.global_step
        })

        self.val_samples = []


class GazeEstimationModule(pl.LightningModule):
    """PyTorch Lightning module for single-task gaze estimation."""

    def __init__(
        self,
        model_type: str = 'lite',
        face_backbone: str = 'resnet18',
        eye_backbone: str = 'resnet18',
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build model
        if model_type == 'lite':
            self.model = GazeTRLite(
                face_backbone=face_backbone,
                eye_backbone=eye_backbone,
                embed_dim=512,
                num_transformer_layers=4,
            )
        else:  # 'full'
            self.model = GazeTRHybrid(
                face_backbone='vit_base_patch16_224',
                eye_backbone='vit_small_patch16_224',
                embed_dim=768,
                num_cross_attn_layers=2,
            )

        self.angular_loss = AngularLoss()
        self.mse_loss = torch.nn.MSELoss()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # For logging
        self.val_errors = []

    def forward(self, face, left_eye, right_eye):
        return self.model(face, left_eye, right_eye)

    def _compute_angular_error_degrees(self, pred, target, validity=None):
        """Compute angular error in degrees."""
        # Convert pitch/yaw to 3D vectors
        pred_vec = self._pitchyaw_to_vector(pred)
        target_vec = self._pitchyaw_to_vector(target)

        # Normalize
        pred_vec = F.normalize(pred_vec, dim=-1)
        target_vec = F.normalize(target_vec, dim=-1)

        # Angular error
        cos_sim = (pred_vec * target_vec).sum(dim=-1).clamp(-0.99999, 0.99999)
        angular_error_rad = torch.acos(cos_sim)
        angular_error_deg = angular_error_rad * 180.0 / np.pi

        if validity is not None:
            validity = validity.float()
            if validity.sum() > 0:
                return (angular_error_deg * validity).sum() / validity.sum()
        return angular_error_deg.mean()

    @staticmethod
    def _pitchyaw_to_vector(pitchyaw):
        pitch = pitchyaw[..., 0]
        yaw = pitchyaw[..., 1]
        x = -torch.cos(pitch) * torch.sin(yaw)
        y = -torch.sin(pitch)
        z = -torch.cos(pitch) * torch.cos(yaw)
        return torch.stack([x, y, z], dim=-1)

    def training_step(self, batch, batch_idx):
        frames = batch['frames']
        face = frames['face']
        left_eye = frames['left']
        right_eye = frames['right']
        gaze_target = batch['gaze']
        validity = batch.get('gaze_validity', None)
        if validity is not None:
            validity = validity.squeeze(-1)  # (B, 1) -> (B,)

        # Handle sequences: flatten batch and sequence dims
        if face.dim() == 5:  # (B, T, C, H, W)
            B, T = face.shape[:2]
            face = face.view(B * T, *face.shape[2:])
            left_eye = left_eye.view(B * T, *left_eye.shape[2:])
            right_eye = right_eye.view(B * T, *right_eye.shape[2:])
            gaze_target = gaze_target.view(B * T, -1)
            if validity is not None:
                validity = validity.view(B * T)

        # Forward
        gaze_pred = self(face, left_eye, right_eye)

        # Loss
        angular_loss = self.angular_loss(gaze_pred, gaze_target, validity)
        mse_loss = self.mse_loss(gaze_pred, gaze_target)
        loss = angular_loss + 0.1 * mse_loss  # Combined loss

        # Metrics
        angular_error_deg = self._compute_angular_error_degrees(gaze_pred, gaze_target, validity)

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/angular_loss', angular_loss, sync_dist=True)
        self.log('train/angular_error_deg', angular_error_deg, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        frames = batch['frames']
        face = frames['face']
        left_eye = frames['left']
        right_eye = frames['right']
        gaze_target = batch['gaze']
        validity = batch.get('gaze_validity', None)
        if validity is not None:
            validity = validity.squeeze(-1)  # (B, 1) -> (B,)

        # Handle sequences
        if face.dim() == 5:
            B, T = face.shape[:2]
            face = face.view(B * T, *face.shape[2:])
            left_eye = left_eye.view(B * T, *left_eye.shape[2:])
            right_eye = right_eye.view(B * T, *right_eye.shape[2:])
            gaze_target = gaze_target.view(B * T, -1)
            if validity is not None:
                validity = validity.view(B * T)

        # Forward
        gaze_pred = self(face, left_eye, right_eye)

        # Loss
        angular_loss = self.angular_loss(gaze_pred, gaze_target, validity)

        # Metrics
        angular_error_deg = self._compute_angular_error_degrees(gaze_pred, gaze_target, validity)

        self.log('val/loss', angular_loss, prog_bar=True, sync_dist=True)
        self.log('val/angular_error_deg', angular_error_deg, prog_bar=True, sync_dist=True)

        return {'angular_error': angular_error_deg}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.trainer.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Train single-task gaze estimation')
    parser.add_argument('--eve-root', type=str, default='/tmp/eve_dataset',
                        help='Path to EVE dataset')
    parser.add_argument('--output-dir', type=str, default='checkpoints/gaze-singletask',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='lite', choices=['lite', 'full'],
                        help='Model type: lite (~15M) or full (~100M)')
    parser.add_argument('--face-backbone', type=str, default='resnet18',
                        help='Face backbone for lite model')
    parser.add_argument('--eye-backbone', type=str, default='resnet18',
                        help='Eye backbone for lite model')
    parser.add_argument('--camera', type=str, default='basler',
                        help='Camera type: basler (60fps) or c (webcam 30fps)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader workers')
    parser.add_argument('--max-epochs', type=int, default=30,
                        help='Max training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=1,
                        help='Frames per sample (1=single frame, >1=sequence)')
    parser.add_argument('--stride', type=int, default=4,
                        help='Frame stride within videos')
    parser.add_argument('--run-name', type=str, default='gaze-singletask-v1',
                        help='W&B run name')
    parser.add_argument('--gpus', type=int, default=8,
                        help='Number of GPUs')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print(f"Loading EVE dataset from {args.eve_root}...")
    train_dataset = EVEDataset(
        root_dir=args.eve_root,
        split='train',
        camera=args.camera,
        sequence_length=args.sequence_length,
        stride=args.stride,
        target_type='gaze',
        augment=True,
        patch_types=['face', 'left', 'right'],
        face_size=224,
        eye_size=112,
    )
    val_dataset = EVEDataset(
        root_dir=args.eve_root,
        split='val',
        camera=args.camera,
        sequence_length=args.sequence_length,
        stride=args.stride,
        target_type='gaze',
        augment=False,
        patch_types=['face', 'left', 'right'],
        face_size=224,
        eye_size=112,
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    model = GazeEstimationModule(
        model_type=args.model,
        face_backbone=args.face_backbone,
        eye_backbone=args.eye_backbone,
        lr=args.lr,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model}")
    print(f"Total params: {total_params/1e6:.2f}M")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")

    # W&B logger
    wandb_logger = WandbLogger(
        project='eve-gaze',
        name=args.run_name,
        save_dir='/tmp/wandb_logs',
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='{epoch:02d}-error{val/angular_error_deg:.2f}',
        monitor='val/angular_error_deg',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop = EarlyStopping(
        monitor='val/angular_error_deg',
        patience=10,
        mode='min',
    )
    gaze_viz = GazeVisualizationCallback(num_samples=8)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.gpus,
        strategy='ddp' if args.gpus > 1 else 'auto',
        precision='bf16-mixed',
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop, gaze_viz],
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining complete!")
    print(f"Best model saved to: {output_dir}")


if __name__ == '__main__':
    main()
