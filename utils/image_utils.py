#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def warpped_depth(depth):
    # 1. 当渲染的深度值<10时，则缩放到[0, 1]；>=10时，则按2-10/深度值非线性压缩深度
    # 2. 再除以2，进一步压缩到[0,1]
    return torch.where(depth < 10.0, depth / 10.0, 2.0 - 10 / depth) / 2.0

def unwarpped_depth(depth):
    return torch.where(depth < 0.5, 2 * depth * 10.0, 10 / (1.0 - depth) / 2)
