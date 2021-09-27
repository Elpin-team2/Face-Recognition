from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
from typing import List, Any, Tuple, Optional


class GoogLeNet(nn.Module):
    def __init__(
        self,
        emb_dim: int = 512,
        aux_logits: bool = True,
    ) -> None:
        super(GoogLeNet, self).__init__()

        conv_block = BasicConv2d
        inception_block = Inception
        inception_aux_block = InceptionAux

        self.aux_logits = aux_logits

        self.conv1 = conv_block(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, emb_dim)
            self.aux2 = inception_aux_block(528, emb_dim)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, emb_dim)
        
    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 1 x 256 x 256
        x = self.conv1(x)
        # N x 64 x 128 x 128
        x = self.maxpool1(x)
        # N x 64 x 64 x 64
        x = self.conv2(x)
        # N x 64 x 64 x 64
        x = self.conv3(x)
        # N x 192 x 64 x 64
        x = self.maxpool2(x)

        # N x 192 x 32 x 32
        x = self.inception3a(x)
        # N x 256 x 32 x 32
        x = self.inception3b(x)
        # N x 480 x 32 x 32
        x = self.maxpool3(x)
        # N x 480 x 16 x 16
        x = self.inception4a(x)
        # N x 512 x 16 x 16
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 16 x 16
        x = self.inception4c(x)
        # N x 512 x 16 x 16
        x = self.inception4d(x)
        # N x 528 x 16 x 16
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 16 x 16
        x = self.maxpool4(x)
        # N x 832 x 8 x 8
        x = self.inception5a(x)
        # N x 832 x 8 x 8
        x = self.inception5b(x)
        # N x 1024 x 8 x 8

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x emb_dim
        return x, aux2, aux1

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)   