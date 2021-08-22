import os
import sys
import csv
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torchvision import transforms

class SpygrModule(nn.Module):
    def __init__(self, x, device):
        super().__init__()
        self.x = x
        self.x.to(device)
        self.M = 64
        self.device = device
        self.phi_conv = nn.Conv2d(x.shape[1], self.M, kernel_size=3, stride=1, padding=1, bias=True)
        self.glob_pool = nn.AvgPool2d((x.shape[2], x.shape[3]))
        self.glob_conv = nn.Conv2d(x.shape[1], self.M, kernel_size=1, stride=1, padding=0, bias=False)
        self.graph_weight = nn.Conv2d(self.x.shape[1], self.x.shape[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.ReLu = nn.ReLU()

    def forward(self):
        x_phi_conv = self.phi_conv(self.x)
        x_phi = x_phi_conv.view([x_phi_conv.shape[0], -1, self.M])
        x_phi = self.ReLu(x_phi)
        x_phi_T = x_phi_conv.view([x_phi_conv.shape[0], self.M, -1])
        x_phi_T = self.ReLu(x_phi_T)

        x_glob_pool = self.glob_pool(self.x)
        x_glob_conv = self.glob_conv(x_glob_pool)
        x_glob_diag = torch.zeros(x_glob_conv.shape[0], x_glob_conv.shape[1], x_glob_conv.shape[1]).to(self.device)

        for i in range(x_glob_conv.shape[0]):
            x_glob_diag[i, :, :] = torch.diag(x_glob_conv[i, :, :, :].reshape(1, x_glob_conv.shape[1]))

        A_tilde = torch.matmul(torch.matmul(x_phi, x_glob_diag), x_phi_T)
        # print(A_tilde)
        D_tilde = torch.zeros_like(A_tilde).to(self.device)
        temp_sum = torch.sum(A_tilde, 2)
        for i in range(D_tilde.shape[0]):
            D_tilde[i, :, :] = torch.diag(temp_sum[i, :])
        # D_tilde = torch.diag(torch.sum(A_tilde, 2))

        # D_inv = torch.inverse(torch.sqrt(D_tilde))
        # D_inv = torch.linalg.inv(torch.sqrt(D_tilde))
        D_inv = D_tilde - A_tilde ## To Fix

        I = torch.eye(D_inv.shape[1]).to(self.device)
        I = I.repeat(D_inv.shape[0], 1, 1)

        L_tilde = I - torch.matmul(torch.matmul(D_inv, A_tilde), D_inv)

        output = torch.matmul(L_tilde, self.x.reshape(self.x.shape[0], -1, self.x.shape[1]))
        # print(output)
        output = output.reshape(self.x.shape[0], self.x.shape[1], self.x.shape[2], self.x.shape[3])
        output = self.graph_weight(output)
        output = self.ReLu(output)

        return output

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super().__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out

class SpyGRSS(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.pretrained_resnet = models.resnet50(pretrained=True).to(self.device)

        for param in self.pretrained_resnet.parameters():
            param.requires_grad_(False)

        ## Layer ##
        self.reduce_dim = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False)

        self.down_samp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pretrained_resnet.conv1(x)
        x = self.pretrained_resnet.bn1(x)
        x = self.pretrained_resnet.relu(x)
        x = self.pretrained_resnet.maxpool(x)
        x = self.pretrained_resnet.layer1(x)
        x = self.pretrained_resnet.layer2(x)
        x = self.pretrained_resnet.layer3(x)
        x = self.pretrained_resnet.layer4(x)
        x = self.reduce_dim(x)

        GR_1 = SpygrModule(x, self.device).to(self.device)

        x = self.down_samp(x)

        GR_2 = SpygrModule(x, self.device).to(self.device)

        x = self.down_samp(x)

        GR_3 = SpygrModule(x, self.device).to(self.device)

        x_gr_1 = GR_1.forward()
        x_gr_2 = GR_2.forward()
        x_gr_3 = GR_3.forward()
        output = x_gr_2 + F.interpolate(x_gr_3, scale_factor=2, mode="nearest")
        output = x_gr_1 + F.interpolate(output, scale_factor=2, mode="nearest")

        final_upsampling = upconv(512, 3, ratio=16).to(self.device)
        output = final_upsampling.forward(output)

        return output

def main():
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    print(torch.cuda.is_available())
    print(torch.device('cuda:0'))

    temp = Image.open("D:/dataset/gtFine_trainvaltest/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png").convert("RGB")
    temp_img = np.array(temp)
    temp_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    temp_img = temp_tf(temp)
    temp_img = torch.stack([temp_img, temp_img]).to(device)

    # temp_model = SpyGRSS(device)
    temp_model = SpyGRSS(device)
    temp_model.to(device)
    temp_model.cuda()
    output = temp_model.forward(temp_img)
    # print(output.shape)
    temp_tff = T.ToPILImage()
    output = temp_tff(output[0, :, :, :])
    output.show()

    # temp_test = torch.randn(3, 3, 3)
    # temp_test.to(device)
    # print("success")



if __name__ == "__main__":
    main()