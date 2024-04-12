import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision
from .odconnv import ODConv2d

def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(BasicBlock, self).__init__()
        self.conv1 = odconv3x3(inplanes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = odconv3x3(planes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # img = cv2.imread("2015_00006.jpg")
    img = cv2.imread("./2015_00015.jpg")
    img = cv2.resize(img,(512,512))
    
    # lowres = cv2.resize(img,(256,256))
    img = (np.asarray(img))
    img = torch.from_numpy(img).float().permute(2,0,1)
    img = img.cuda().unsqueeze(0)
    image_multiplied = img.repeat(8, 1, 1, 1)
    # img = torch.Tensor(8, 3, 400, 600).cuda()
    net = BasicBlock(3,3).cuda()
    # net.init_weights()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    Hierarchical_Representation = net(image_multiplied)
    torchvision.utils.save_image(Hierarchical_Representation, "Hierarchical_Repre.png")
    # torchvision.utils.save_image(high, "high.png")