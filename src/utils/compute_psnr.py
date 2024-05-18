# import sys
# import logging.config
# import os
from datetime import datetime

# import subprocess
# import random


import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import json
import skimage.measure
# from piq import ssim

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms, utils
import piq
# from torchvision.transforms.functional import rgb_to_grayscale
from model.isabel import *


from model.generator import Generator
# from model.discriminator import Discriminator
from model.vgg19 import VGG19

from utils.logger_utils import setup_logger
from utils.config_setter import load_config

def vparams2azel(vparams):
	azel = []
	for vp in vparams:
		angle_rad = np.arctan2(vp[1], vp[0])
		theta_value = np.rad2deg(angle_rad)
		phi_value = vp[2] * 90.0
		azel.append([phi_value, theta_value])
	return azel

# select device
device = torch.device("cuda:0")

chkpt = input("Enter the path to the model: ")
# chkpt =  "../outputs/Isabel_mse_active_random/vgg_resume1500ep/model/chk_1499.pth.tar"
lr = 1e-3
betas=(0.9, 0.999)

# set random seed
np.random.seed(1)
torch.manual_seed(1)

# dataset creation
train_dataset = IsabelDataset(
    root="../data/Isabel_pressure_volume_images/train/",
	param_file = "isabel_pr_viewparams_train.csv",
    train=True,
    test=False,
    transform=transforms.Compose([Normalize(), ToTensor()]))

test_dataset = IsabelDataset(
    root="../data/Isabel_pressure_volume_images/test/",
	param_file = "isabel_pr_viewparams_test.csv",
    train=False,
    test=True,
    transform=transforms.Compose([Normalize(), ToTensor()]))

kwargs = {"num_workers": 4, "pin_memory": True}
train_loader = DataLoader(train_dataset, batch_size=100,
                        shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=100,
                        shuffle=True, **kwargs)
# model
g_model = Generator(dvp=3,
					dvpe=512,
					ch=64)

g_model.to(device)

# optimizer
g_optimizer = optim.Adam(g_model.parameters(), lr=lr,
						betas=betas)

checkpoint = torch.load(chkpt)
g_model.load_state_dict(checkpoint["g_model_state_dict"])
g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
# batch_train_losses = checkpoint["batch_train_losses"]
# train_losses = checkpoint["train_losses"]
# test_losses = checkpoint["test_losses"]


g_model.eval()

# LPIPS metric
lpips_metric = piq.LPIPS(reduction='mean',
                         mean = [0., 0., 0.],
                         std = [1., 1., 1.]).to(device) # Setting mean and std to 0 and 1 respectively as dont need to renormalize images using imagenet statistics

ssim_loss = 0.
psnr_loss = 0.
lpips_loss = 0.

with torch.no_grad():
    for i, sample in enumerate(test_loader):
        image = sample["image"].to(device)
        image2 = (image+ 1.) * .5
        vparams = sample["vparams"].to(device)
        fake_image = g_model(vparams)
        fake_image2 = (fake_image+ 1.) * .5
        # test_loss += mse_criterion(image, fake_image).item() * len(image)
        ssim_loss += piq.ssim(image2, fake_image2, data_range=1.).item() * len(image)/len(test_loader.dataset)
        psnr_loss += piq.psnr(image2, fake_image2, data_range=1., reduction='mean').item() * len(image)/len(test_loader.dataset)
        # lpips_loss += piq.LPIPS(image, fake_image, reduction='mean').item() * len(image)/len(test_loader.dataset)
        lpips_loss += lpips_metric(image, fake_image).item() * len(image)/len(test_loader.dataset)
        # print (f"\tTest set SSIM loss for Batch {i+1}/{len(test_loader)} is {ssim_loss:.4f}")
        # print (f"\tTest set PSNR loss for Batch {i+1}/{len(test_loader)} is {psnr_loss:.4f}")
        # print (f"\tTest set LPIPS loss for Batch {i+1}/{len(test_loader)} is {lpips_loss:.4f}")

        # run_logger.info("\tTest set loss for epoch %d, Batch %d/%d is %.4f", epoch, i+1, len(test_loader), test_loss)
avg_ssim_loss = ssim_loss
avg_psnr_loss = psnr_loss
avg_lpips_loss = lpips_loss

print(f"\t====> Average Test set SSIM: \t\t{avg_ssim_loss:.4f}")
print(f"\t====> Average Test set PSNR: \t\t{avg_psnr_loss:.4f}")
print(f"\t====> Average Test set LPIPS: \t\t{avg_lpips_loss:.4f}")
