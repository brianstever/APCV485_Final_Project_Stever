"""
CNN + Squeeze-and-Excitation detector for deepfake audio.

Architecture:
  - Four convolutional blocks (32 → 64 → 128 → 256), each with two 3×3
    conv layers, batch norm, ReLU, and 2×2 max-pooling.
  - A Squeeze-and-Excitation module after the final block recalibrates
    channel responses (reduction ratio r = 16).
  - Global average pooling → 512-unit FC → dropout(0.5) → 1-unit output.

The forward pass returns raw logits; apply sigmoid externally when
you need probabilities.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """Two 3×3 conv layers + BN + ReLU, followed by 2×2 max-pool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepfakeCNNSE(nn.Module):
    """Spectrogram-based CNN with SE-attention for deepfake audio detection."""

    def __init__(self, se_reduction: int = 16, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.se = SEBlock(256, reduction=se_reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x  # raw logit
