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
from torch.backends import cudnn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import tqdm

from spygr import *
from spygr_dataloader import *

from torch.backends import cudnn

device = torch.device("cuda")

def evaluate_model(val_dataloader: CityScapesData, model: SpyGR, criterion):

    eval_iou_score = 0
    eval_loss = 0

    for sample_set in tqdm(val_dataloader):
        with torch.no_grad():
            eval_images = sample_set["image"].to(device)
            eval_labels = sample_set["lable"].to(device)

            eval_outputs = model(eval_images)
            model.eval()
            eval_loss = criterion(eval_outputs, eval_labels)
            eval_outputs = F.softmax(eval_outputs, dim=1)
            eval_outputs = torch.argmax(eval_outputs, dim=1)
            eval_outputs = eval_outputs.contiguous().view(-1)
            eval_labels = eval_labels.contiguous().view(-1)

            iou_per_class = []
            for num_class in range(len(val_dataloader.class_names)):
                true_class = (eval_outputs == num_class)
                true_label = (eval_labels == num_class)
                if true_label.long().sum().item() == 0:
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(
                        true_class, true_label).sum().float().item()
                    union = torch.logical_and(
                        true_class, true_label).sum().float().item()

                    iou = (intersect + 1e-10) / (union + 1e-10)
                    iou_per_class.append(iou)
            eval_iou_score += np.nanmean(iou_per_class)
            eval_loss += eval_loss

    return eval_loss, eval_iou_score


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

    root_dir = "D:/dataset" ### Directory of dataset

    train_set = CityScapesData(root_dir, "fine", "train", transforms_train, (768,768))
    valid_set = CityScapesData(root_dir, "fine", "val", transforms_val, (768,768))
    test_set = CityScapesData(root_dir, "fine", "test", transforms_test, (768,768))

    train_batch_size = 16  # Control by YAML
    valid_batch_size = 64  # Control by YAML
    test_batch_size = 32  # Control by YAML

    train_loader = DataLoader(train_set, train_batch_size, num_workers=1)
    valid_loader = DataLoader(valid_set, valid_batch_size, num_workers=1)
    test_loader = DataLoader(test_set, test_batch_size, num_workers=1)

    # device = torch.device("cuda")

    model = SpyGR(device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Control by YAML
    criterion = nn.CrossEntropyLoss(ignore_index=train_set.ignore_label).cuda()

    num_epochs = 80  # Control by YAML
    model.train()

    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape)
                            for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    cudnn.benchmark = True

    global_step = 0
    steps_per_epoch = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print("[epoch][s/s_per_e/global_step]: [{}/{}][{}/{}/{}], loss: {:.12f}".format(
                epoch+1, num_epochs, i+1, steps_per_epoch, global_step+1, loss))

            checkpoint = {"global_step": global_step,
                          "model": model.state_dict(),
                          "optimizer": optimizer.state_dict()}

            torch.save(checkpoint, os.path.join(
                "D:/model", "spygr", "-{:07d}.pth".format(global_step)))

        eval_loss, eval_iou_score = evaluate_model(valid_loader, model, criterion)
        print("Epoch: {}/{} | validation average loss: {:.5f} | evaluation mIoU: {:.5f}".format(epoch, num_epochs, eval_loss/len(valid_loader, eval_iou_score/len(valid_loader))))
        model.train()

        global_step += 1

if __name__ == "__main__":
    main()
