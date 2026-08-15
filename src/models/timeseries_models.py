"""State-of-the-art 1D Neural Network architectures for multivariate gas sensor time-series.

Includes:
- InceptionTime: Multi-scale 1D convolutions with bottleneck layers and residual connections
- ResNet1D: Deep 1D Residual Network with skip connections
- AttentiveLSTM: Bidirectional LSTM with temporal attention pooling
"""

from __future__ import annotations

import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """1D convolution with explicit 'same' padding across arbitrary kernel sizes."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = False) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)


class InceptionBlock1D(nn.Module):
    """Core 1D Inception Block with bottleneck projection and multi-scale kernel branches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        kernel_sizes: Sequence[int] = (9, 19, 39),
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1

        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_channels

        self.convs = nn.ModuleList([
            Conv1dSamePadding(conv_in, out_channels, kernel_size=k, bias=False)
            for k in kernel_sizes
        ])

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = Conv1dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)

        total_out = out_channels * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, seq_len)
        z = self.bottleneck(x)
        conv_outs = [conv(z) for conv in self.convs]
        pool_out = self.maxpool_conv(self.maxpool(x))
        concat = torch.cat(conv_outs + [pool_out], dim=1)
        return self.relu(self.bn(concat))


class InceptionModule1D(nn.Module):
    """Inception block combined with a residual shortcut connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        kernel_sizes: Sequence[int] = (9, 19, 39),
        bottleneck_channels: int = 32,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.inception = InceptionBlock1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
        )
        total_out = out_channels * (len(kernel_sizes) + 1)

        if use_residual:
            self.shortcut = (
                nn.Sequential(
                    Conv1dSamePadding(in_channels, total_out, kernel_size=1, bias=False),
                    nn.BatchNorm1d(total_out),
                )
                if in_channels != total_out
                else nn.BatchNorm1d(total_out)
            )
        else:
            self.shortcut = None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inception(x)
        if self.use_residual and self.shortcut is not None:
            res = self.shortcut(x)
            out = self.relu(out + res)
        return out


class InceptionTime(nn.Module):
    """InceptionTime neural network for multivariate electronic nose classification."""

    def __init__(
        self,
        n_sensors: int,
        n_classes: int,
        num_blocks: int = 3,
        out_channels: int = 32,
        kernel_sizes: Sequence[int] = (9, 19, 39),
        bottleneck_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        layers: list[nn.Module] = []
        in_ch = n_sensors
        block_out_dim = out_channels * (len(kernel_sizes) + 1)

        for _ in range(num_blocks):
            layers.append(
                InceptionModule1D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    use_residual=True,
                )
            )
            in_ch = block_out_dim

        self.backbone = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(block_out_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, n_sensors, seq_len) or (batch_size, seq_len, n_sensors)
        if x.dim() == 3 and x.shape[1] != self.n_sensors and x.shape[2] == self.n_sensors:
            x = x.transpose(1, 2)

        feat = self.backbone(x)
        pooled = self.gap(feat).squeeze(-1)
        out = self.fc(self.dropout(pooled))
        return out


class ResNetBlock1D(nn.Module):
    """1D Residual block with 3 conv layers and a shortcut projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = Conv1dSamePadding(in_channels, out_channels, kernel_size=7)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = Conv1dSamePadding(out_channels, out_channels, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = Conv1dSamePadding(out_channels, out_channels, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = (
            nn.Sequential(
                Conv1dSamePadding(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + res)


class ResNet1D(nn.Module):
    """Deep 1D ResNet for time-series gas sensor classification."""

    def __init__(self, n_sensors: int, n_classes: int, layer_sizes: Sequence[int] = (64, 128, 128)) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        blocks: list[nn.Module] = []
        in_ch = n_sensors
        for out_ch in layer_sizes:
            blocks.append(ResNetBlock1D(in_ch, out_ch))
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(layer_sizes[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != self.n_sensors and x.shape[2] == self.n_sensors:
            x = x.transpose(1, 2)
        feat = self.backbone(x)
        pooled = self.gap(feat).squeeze(-1)
        return self.fc(pooled)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism over sequence time-steps."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq_len, hidden_dim)
        scores = self.weight(h)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * h, dim=1)  # (batch, hidden_dim)
        return context


class AttentiveLSTM(nn.Module):
    """Bidirectional LSTM with temporal attention pooling."""

    def __init__(
        self,
        n_sensors: int,
        n_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If shape is (batch, n_sensors, seq_len), transpose to (batch, seq_len, n_sensors)
        if x.dim() == 3 and x.shape[1] == self.n_sensors and x.shape[2] != self.n_sensors:
            x = x.transpose(1, 2)
        h, _ = self.lstm(x)
        context = self.attention(h)
        return self.fc(context)
