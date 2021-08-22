import os
import numpy as np
import pandas as pd

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor

import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

import spygr
import spygr_dataloader

def main():
    transforms_train = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transforms_val = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transforms_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = "D:/dataset"

    train_set = spygr_dataloader(root_dir, "fine", "train", transforms_train)
    valid_set = spygr_dataloader(root_dir, "fine", "val", transforms_train)
    test_set = spygr_dataloader(root_dir, "fine", "test", transforms_test)

    train_batch_size = 256
    valid_batch_size = 64
    test_batch_size = 32

    train_loader = DataLoader(train_set, train_batch_size, num_workers=1)
    valid_loader = DataLoader(valid_set, valid_batch_size, num_workers=1)
    test_loader = DataLoader(test_set, test_batch_size, num_workers=1)

    device = torch.device("cuda")

    model = spygr(device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MultiLabelSoftMarginLoss()

    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)


if __name__ == "__main__":
    main()