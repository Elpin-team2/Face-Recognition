from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

from model import GoogLeNet


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


class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1-x2).pow(2).sum(1) 
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


def train():
    epochs = 50
    emd_dim = 512

    faces_loader = load_data()

    ## Create instance
    model = GoogLeNet(emd_dim)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # criterion = TripletLoss()
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
        
    model.train()
    for epoch in range(epochs):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(faces_loader):
            
            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
                
            running_loss.append(loss.detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))

    # save model
    torch.save(model, "model.pt")

def load_data():
    # faces_df 불러오기
    faces_df = pd.read_pickle("faces_df.pkl")

    faces_ds = Faces(faces_df,
                 transform=transforms.Compose([transforms.ToTensor()]))

    batch_size = 8
    faces_loader = DataLoader(faces_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    return faces_loader

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

if __name__ == "__main__":
    train()