"""
Single-Task Gaze Estimation Model - GazeTR-Hybrid Style

Based on SOTA gaze estimation architectures:
- GazeTR: ResNet + ViT hybrid
- GazeSymCAT: Symmetric cross-attention transformer
- GazeCapsNet: Lightweight backbone + capsule routing

This model uses:
- Face encoder: ViT-Base (pretrained)
- Eye encoders: Shared ViT-Small for left/right eyes
- Cross-attention fusion: Eyes attend to face features
- Regression head: 2D gaze direction (pitch, yaw)

Target: <5° angular error on EVE validation
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ============================================================================
# Cross-Attention Module
# ============================================================================

class CrossAttention(nn.Module):
    """Cross-attention: query attends to key-value from another modality."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D) - queries (e.g., eye features)
            key_value: (B, N_kv, D) - keys/values (e.g., face features)
        Returns:
            (B, N_q, D) - attended features
        """
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]

        # Normalize inputs
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)

        # Project
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)

        return out


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with residual and FFN."""

    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # Cross-attention with residual
        query = query + self.cross_attn(query, key_value)
        # FFN with residual
        query = query + self.mlp(self.norm(query))
        return query


# ============================================================================
# GazeTR-Hybrid Model
# ============================================================================

class GazeTRHybrid(nn.Module):
    """
    GazeTR-Hybrid style gaze estimation model.

    Architecture:
    - Face encoder: ViT-Base (or ResNet-50) for global face features
    - Eye encoders: Shared ViT-Small for detailed eye features
    - Cross-attention: Eyes attend to face context
    - Regression head: MLP for pitch/yaw prediction

    Input:
    - face: (B, 3, 256, 256) face crop
    - left_eye: (B, 3, 128, 128) left eye crop
    - right_eye: (B, 3, 128, 128) right eye crop

    Output:
    - gaze: (B, 2) pitch/yaw in radians
    """

    def __init__(
        self,
        face_backbone: str = 'vit_base_patch16_224',
        eye_backbone: str = 'vit_small_patch16_224',
        face_size: int = 224,
        eye_size: int = 128,
        embed_dim: int = 768,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_face_backbone: bool = False,
        freeze_eye_backbone: bool = False,
    ):
        super().__init__()

        self.face_size = face_size
        self.eye_size = eye_size
        self.embed_dim = embed_dim

        # Face encoder (ViT-Base)
        self.face_encoder = timm.create_model(
            face_backbone,
            pretrained=True,
            num_classes=0,  # Remove classifier
            global_pool='',  # Return all tokens
        )
        face_embed_dim = self.face_encoder.embed_dim

        # Eye encoder (shared for left/right)
        self.eye_encoder = timm.create_model(
            eye_backbone,
            pretrained=True,
            num_classes=0,
            global_pool='',
        )
        eye_embed_dim = self.eye_encoder.embed_dim

        # Project to common embedding dimension
        self.face_proj = nn.Linear(face_embed_dim, embed_dim) if face_embed_dim != embed_dim else nn.Identity()
        self.eye_proj = nn.Linear(eye_embed_dim, embed_dim) if eye_embed_dim != embed_dim else nn.Identity()

        # Cross-attention layers: eyes attend to face
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])

        # Gaze regression head
        self.gaze_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),  # Left + right eye features
            nn.Linear(embed_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # pitch, yaw
        )

        # Optionally freeze backbones
        if freeze_face_backbone:
            for param in self.face_encoder.parameters():
                param.requires_grad = False
        if freeze_eye_backbone:
            for param in self.eye_encoder.parameters():
                param.requires_grad = False

        # Initialize head
        self._init_weights()

    def _init_weights(self):
        for module in self.gaze_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _resize_pos_embed(self, model, new_size: int):
        """Resize positional embeddings for different input sizes."""
        # This is handled by timm automatically during forward if needed
        pass

    def forward(
        self,
        face: torch.Tensor,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            face: (B, 3, H, W) face image
            left_eye: (B, 3, H, W) left eye image
            right_eye: (B, 3, H, W) right eye image
        Returns:
            gaze: (B, 2) pitch/yaw in radians
        """
        B = face.shape[0]

        # Resize inputs if needed
        if face.shape[-1] != self.face_size:
            face = F.interpolate(face, size=(self.face_size, self.face_size), mode='bilinear', align_corners=False)
        if left_eye.shape[-1] != self.eye_size:
            left_eye = F.interpolate(left_eye, size=(self.eye_size, self.eye_size), mode='bilinear', align_corners=False)
            right_eye = F.interpolate(right_eye, size=(self.eye_size, self.eye_size), mode='bilinear', align_corners=False)

        # Encode face -> (B, N_face, D_face)
        face_features = self.face_encoder.forward_features(face)
        face_features = self.face_proj(face_features)  # (B, N_face, embed_dim)

        # Encode eyes -> (B, N_eye, D_eye) each
        # Resize eyes to match ViT expected size (224)
        if left_eye.shape[-1] != 224:
            left_eye_resized = F.interpolate(left_eye, size=(224, 224), mode='bilinear', align_corners=False)
            right_eye_resized = F.interpolate(right_eye, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            left_eye_resized = left_eye
            right_eye_resized = right_eye

        left_features = self.eye_encoder.forward_features(left_eye_resized)
        right_features = self.eye_encoder.forward_features(right_eye_resized)

        left_features = self.eye_proj(left_features)  # (B, N_eye, embed_dim)
        right_features = self.eye_proj(right_features)

        # Cross-attention: eyes attend to face context
        for cross_attn in self.cross_attn_layers:
            left_features = cross_attn(left_features, face_features)
            right_features = cross_attn(right_features, face_features)

        # Pool to single vectors (use CLS token if available, else mean)
        left_pooled = left_features[:, 0]  # CLS token
        right_pooled = right_features[:, 0]

        # Concatenate and predict gaze
        combined = torch.cat([left_pooled, right_pooled], dim=-1)  # (B, 2*embed_dim)
        gaze = self.gaze_head(combined)  # (B, 2)

        return gaze

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        face_params = sum(p.numel() for p in self.face_encoder.parameters())
        eye_params = sum(p.numel() for p in self.eye_encoder.parameters())
        cross_attn_params = sum(p.numel() for p in self.cross_attn_layers.parameters())
        head_params = sum(p.numel() for p in self.gaze_head.parameters())
        proj_params = sum(p.numel() for p in self.face_proj.parameters()) + sum(p.numel() for p in self.eye_proj.parameters())

        return {
            'face_encoder': face_params,
            'eye_encoder': eye_params,
            'cross_attention': cross_attn_params,
            'projections': proj_params,
            'gaze_head': head_params,
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


# ============================================================================
# Lighter Alternative: ResNet + Transformer Hybrid
# ============================================================================

class GazeTRLite(nn.Module):
    """
    Lighter GazeTR model using ResNet-18 + small transformer.

    More efficient for training, still high quality.
    ~15M params vs ~100M for full model.
    """

    def __init__(
        self,
        face_backbone: str = 'resnet18',
        eye_backbone: str = 'resnet18',
        embed_dim: int = 512,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Face encoder (ResNet-18)
        self.face_encoder = timm.create_model(
            face_backbone,
            pretrained=True,
            num_classes=0,
            global_pool='',
        )
        face_feat_dim = self.face_encoder.num_features

        # Eye encoder (shared ResNet-18)
        self.eye_encoder = timm.create_model(
            eye_backbone,
            pretrained=True,
            num_classes=0,
            global_pool='',
        )
        eye_feat_dim = self.eye_encoder.num_features

        # Projections to flatten spatial features
        self.face_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (B, C, 4, 4)
            nn.Flatten(start_dim=2),  # -> (B, C, 16)
            nn.Linear(16, 1),  # -> (B, C, 1) then squeeze
        )
        self.eye_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # -> (B, C, 2, 2)
            nn.Flatten(start_dim=2),  # -> (B, C, 4)
            nn.Linear(4, 1),
        )

        # Project to embed_dim
        self.face_linear = nn.Linear(face_feat_dim, embed_dim)
        self.eye_linear = nn.Linear(eye_feat_dim, embed_dim)

        # Positional encoding for 3 tokens (face, left, right)
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Gaze head
        self.gaze_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        face: torch.Tensor,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
    ) -> torch.Tensor:
        B = face.shape[0]

        # Resize to standard sizes
        face = F.interpolate(face, size=(224, 224), mode='bilinear', align_corners=False)
        left_eye = F.interpolate(left_eye, size=(112, 112), mode='bilinear', align_corners=False)
        right_eye = F.interpolate(right_eye, size=(112, 112), mode='bilinear', align_corners=False)

        # Encode
        face_feat = self.face_encoder.forward_features(face)  # (B, C, H, W)
        left_feat = self.eye_encoder.forward_features(left_eye)
        right_feat = self.eye_encoder.forward_features(right_eye)

        # Pool and project
        face_vec = self.face_proj(face_feat).squeeze(-1)  # (B, C)
        left_vec = self.eye_proj(left_feat).squeeze(-1)
        right_vec = self.eye_proj(right_feat).squeeze(-1)

        face_vec = self.face_linear(face_vec)  # (B, embed_dim)
        left_vec = self.eye_linear(left_vec)
        right_vec = self.eye_linear(right_vec)

        # Stack as sequence: [face, left, right]
        tokens = torch.stack([face_vec, left_vec, right_vec], dim=1)  # (B, 3, embed_dim)
        tokens = tokens + self.pos_embed

        # Transformer fusion
        fused = self.transformer(tokens)  # (B, 3, embed_dim)

        # Use face token (index 0) for gaze prediction
        gaze = self.gaze_head(fused[:, 0])  # (B, 2)

        return gaze

    def count_parameters(self) -> Dict[str, int]:
        return {
            'face_encoder': sum(p.numel() for p in self.face_encoder.parameters()),
            'eye_encoder': sum(p.numel() for p in self.eye_encoder.parameters()),
            'transformer': sum(p.numel() for p in self.transformer.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }


# ============================================================================
# Loss Functions
# ============================================================================

class AngularLoss(nn.Module):
    """Angular error loss for gaze estimation."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        validity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 2) predicted pitch/yaw in radians
            target: (B, 2) target pitch/yaw in radians
            validity: (B,) optional validity mask
        Returns:
            Angular error in radians (scalar)
        """
        # Convert to 3D vectors
        pred_vec = self._pitchyaw_to_vector(pred)
        target_vec = self._pitchyaw_to_vector(target)

        # Normalize for numerical stability
        pred_vec = F.normalize(pred_vec, dim=-1)
        target_vec = F.normalize(target_vec, dim=-1)

        # Angular error via dot product
        cos_sim = (pred_vec * target_vec).sum(dim=-1).clamp(-0.99999, 0.99999)
        angular_error = torch.acos(cos_sim)

        if validity is not None:
            # Mask invalid samples
            validity = validity.float()
            if validity.sum() > 0:
                angular_error = (angular_error * validity).sum() / validity.sum()
            else:
                angular_error = angular_error.mean()
        else:
            angular_error = angular_error.mean()

        return angular_error

    @staticmethod
    def _pitchyaw_to_vector(pitchyaw: torch.Tensor) -> torch.Tensor:
        """Convert pitch/yaw to 3D unit vector."""
        pitch = pitchyaw[..., 0]
        yaw = pitchyaw[..., 1]

        x = -torch.cos(pitch) * torch.sin(yaw)
        y = -torch.sin(pitch)
        z = -torch.cos(pitch) * torch.cos(yaw)

        return torch.stack([x, y, z], dim=-1)


class GazeMSELoss(nn.Module):
    """Simple MSE loss on pitch/yaw angles."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        validity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.mse(pred, target).mean(dim=-1)  # (B,)

        if validity is not None:
            validity = validity.float()
            if validity.sum() > 0:
                loss = (loss * validity).sum() / validity.sum()
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()

        return loss


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing GazeTR-Hybrid model...")

    # Test full model
    model = GazeTRHybrid(
        face_backbone='vit_base_patch16_224',
        eye_backbone='vit_small_patch16_224',
        embed_dim=768,
        num_cross_attn_layers=2,
    )

    params = model.count_parameters()
    print(f"\nGazeTRHybrid parameters:")
    for k, v in params.items():
        print(f"  {k}: {v/1e6:.2f}M")

    # Test forward
    face = torch.randn(2, 3, 256, 256)
    left = torch.randn(2, 3, 128, 128)
    right = torch.randn(2, 3, 128, 128)

    gaze = model(face, left, right)
    print(f"\nOutput shape: {gaze.shape}")  # Should be (2, 2)

    # Test loss
    loss_fn = AngularLoss()
    target = torch.randn(2, 2) * 0.5  # Random gaze targets
    loss = loss_fn(gaze, target)
    print(f"Angular loss: {loss.item():.4f} rad = {loss.item() * 180 / 3.14159:.2f}°")

    # Test lite model
    print("\n" + "="*50)
    print("Testing GazeTRLite model...")

    model_lite = GazeTRLite(
        face_backbone='resnet18',
        eye_backbone='resnet18',
        embed_dim=512,
        num_transformer_layers=4,
    )

    params_lite = model_lite.count_parameters()
    print(f"\nGazeTRLite parameters:")
    for k, v in params_lite.items():
        print(f"  {k}: {v/1e6:.2f}M")

    gaze_lite = model_lite(face, left, right)
    print(f"\nOutput shape: {gaze_lite.shape}")

    loss_lite = loss_fn(gaze_lite, target)
    print(f"Angular loss: {loss_lite.item():.4f} rad = {loss_lite.item() * 180 / 3.14159:.2f}°")
