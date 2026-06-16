"""Models for Fashion-MNIST OOD experiment: Random Forest baseline and a small CNN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Small VGG-ish CNN for 28x28x3 input.

    Two conv blocks, global average pooling, then a small classifier.
    The penultimate 128-d activation is used as the feature for t-SNE.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                    # 1x1
        )
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)        # (B, 128)
        z = F.relu(self.fc1(z))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        return self.fc2(self.dropout(z))
