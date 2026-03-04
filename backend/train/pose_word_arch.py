from __future__ import annotations

import torch
from torch import nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PoseWordClassifier(nn.Module):
    """Baseline pose-word classifier: 1D-CNN front-end + BiGRU temporal encoder."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        conv_channels: int = 192,
        gru_hidden: int = 192,
        gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.conv_channels = int(conv_channels)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)
        self.dropout = float(dropout)

        self.conv = nn.Sequential(
            ConvBlock1D(self.input_dim, self.conv_channels, kernel_size=5, dropout=self.dropout),
            ConvBlock1D(self.conv_channels, self.conv_channels, kernel_size=3, dropout=self.dropout),
        )
        self.gru = nn.GRU(
            input_size=self.conv_channels,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.gru_hidden * 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.gru_hidden * 2, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        if x.dim() != 3:
            raise ValueError(f"expected [B,T,F], got {tuple(x.shape)}")
        feat = x.transpose(1, 2)
        feat = self.conv(feat)
        feat = feat.transpose(1, 2)
        seq, _ = self.gru(feat)
        pooled = seq.mean(dim=1)
        return self.head(pooled)


__all__ = ["PoseWordClassifier"]
