import numpy as np
import pandas as pd

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import torchvision.models as models

from PIL import Image
import torchvision.transforms as T

class GRModule(nn.Module):
    def __init__(self, x, device):
        super().__init__()
        self.x = x
        self.device = device
        self.M = 64
        self.phi_conv = nn.Conv2d(self.x.shape[1], self.M, kernel_size=3, stride=1, padding=1, bias=True)
        self.glob_pool = nn.AvgPool2d(self.x.shape[2], self.x.shape[3])
        self.glob_conv = nn.Conv2d(self.x.shape[1], self.M, kernel_size=1, stride=1, padding=0, bias=False)
        self.graph_weight = nn.Conv2d(self.x.shape[1], self.x.shape[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self):
        x_phi_conv = self.phi_conv(self.x)
        x_phi = x_phi_conv.view([x_phi_conv.shape[0], -1, self.M])
        x_phi = self.relu(x_phi)
        x_phi_T = x_phi_conv.view([x_phi_conv.shape[0], self.M, -1])
        x_phi_T = self.relu(x_phi_T)

        x_glob_pool = self.glob_pool(self.x)
        x_glob_conv = self.glob_conv(x_glob_pool)
        x_glob_diag = torch.zeros(x_glob_conv.shape[0], x_glob_conv.shape[1], x_glob_conv.shape[1]).to(self.device)
        
        for i in range(x_glob_conv.shape[0]):
            x_glob_diag[i, :, :] = torch.diag(x_glob_conv[i, :, :, :].reshape(1, x_glob_conv.shape[1]))
        
        A_tilde = torch.matmul(torch.matmul(x_phi, x_glob_diag), x_phi_T)
        # D_tilde = torch.zeros_like(A_tilde).to(self.device)
        D_sqrt_inv = torch.zeros_like(A_tilde).to(self.device)
        
        # temp_sum = torch.sum(A_tilde, 2)
        # for i in range(D_tilde.shape[0]):
        #     D_tilde[i, :, :] = torch.diag(temp_sum[i, :])
        
        diag_sum = torch.sum(A_tilde, 2)
        
        for i in range(diag_sum.shape[0]):
            diag_sqrt = 1.0 / torch.sqrt(diag_sum[i, :])
            diag_sqrt[torch.isnan(diag_sqrt)] = 0
            diag_sqrt[torch.isinf(diag_sqrt)] = 0
            D_sqrt_inv[i, :, :] = torch.diag(diag_sqrt)
        print("NaN Check(D_sqrt_inv): ", torch.isnan(D_sqrt_inv).any())
        I = torch.eye(D_sqrt_inv.shape[1]).to(self.device)
        I = I.repeat(D_sqrt_inv.shape[0], 1, 1)

        L_tilde = I - torch.matmul(torch.matmul(D_sqrt_inv, A_tilde), D_sqrt_inv)
        print("NaN Check(L_tilde): ", torch.isnan(L_tilde).any())
        out = torch.matmul(L_tilde, self.x.reshape(self.x.shape[0], -1, self.x.shape[1]))
        print("NaN Check(out): ", torch.isnan(out).any())
        out = out.reshape(self.x.shape[0], self.x.shape[1], self.x.shape[2], self.x.shape[3])
        out = self.graph_weight(out)
        out = self.relu(out)
        
        return out
    
class upconv(nn.Module):
    def __init__(self, in_channels, out_chennels, ratio):
        super().__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_chennels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.ratio, mode="nearest")
        out = self.conv(up_x)
        out = self.elu(out)

        return out
    
class SpyGR(nn.Module):
    def __init__(self, device, num_class=19):
        super().__init__()
        self.device = device
        self.num_class = num_class
        self.pretrained_resnet = models.resnet50(pretrained=True).to(self.device)

        for param in self.pretrained_resnet.parameters():
            param.requires_grad_(False)
        
        self.reduce_dim = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.down_samp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.classification = nn.Conv2d(512, self.num_class, kernel_size=1, padding=0)
        self.relu = nn.ReLU()

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
        x = self.relu(x)
        GR_1 = GRModule(x, self.device).to(self.device)
        x = self.down_samp(x)
        GR_2 = GRModule(x, self.device).to(self.device)
        x = self.down_samp(x)
        GR_3 = GRModule(x, self.device).to(self.device)

        x_gr_1 = GR_1.forward()
        x_gr_1 = self.relu(x_gr_1)
        x_gr_2 = GR_2.forward()
        x_gr_2 = self.relu(x_gr_2)
        x_gr_3 = GR_3.forward()
        x_gr_3 = self.relu(x_gr_3)

        out = x_gr_2 + F.interpolate(x_gr_3, scale_factor=2, mode="nearest")
        out = x_gr_1 + F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.classification(out)
        final_upsampling = upconv(self.num_class, self.num_class, ratio=32).to(self.device)
        out = final_upsampling.forward(out)
        
        return out

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    temp = Image.open("D:/dataset/gtFine_trainvaltest/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png").convert("RGB")
    temp_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    temp_img = temp_tf(temp)
    temp_img = torch.stack([temp_img, temp_img]).to(device)

    temp_model = SpyGR(device)
    temp_model.to(device)
    output = temp_model.forward(temp_img)
    print("output size: ", output.shape)

    temp_tff = T.ToPILImage()
    output = temp_tff(output[0, :, :, :])
    output.show()