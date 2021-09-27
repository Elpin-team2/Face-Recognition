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


class Faces(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.to_pil = transforms.ToPILImage()
        self.images = df.iloc[:, 0]
        self.labels = df.iloc[:, 1]
        self.index = df.index.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item].reshape(256, 256, 1)
        anchor_label = self.labels[item]

        # get positive image
        positive_list = self.index[self.index != item][self.labels[self.index!=item]==anchor_label]
        positive_item = random.choice(positive_list) 
        positive_img = self.images[positive_item].reshape(256, 256, 1)

        # get negative image
        nonpositive_list = self.index[self.index != item][self.labels[self.index!=item]!=anchor_label]
        nonpositive_item = random.choice(nonpositive_list)
        negative_label = anchor_label + 1 if anchor_label%2 == 0 else anchor_label - 1
        negative_list = self.index[self.index != item][self.labels[self.index!=item]==negative_label]
        negative_item = random.choice(np.append(negative_list, nonpositive_item))
        negative_img = self.images[negative_item].reshape(256, 256, 1)

        # transform
        if self.transform:
            anchor_img = self.transform(self.to_pil(anchor_img))
            positive_img = self.transform(self.to_pil(positive_img))
            negative_img = self.transform(self.to_pil(negative_img))

        return anchor_img, positive_img, negative_img, anchor_label