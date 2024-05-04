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
from piq import ssim

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms, utils
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
num_new_samples = 10
# query_strategy = "complexity"
# query_strategy = "Random"
# query_strategy = "MSELoss" 
# query_strategy = "VGG" 
# query_strategy = "rand_MSELoss" 
query_strategy = "diversity"


chkpt = "../data/chk_99.pth.tar"
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
train_loader = DataLoader(train_dataset, batch_size=150,
                        shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=150,
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
train_losses = checkpoint["train_losses"]
test_losses = checkpoint["test_losses"]


g_model.eval()

all_vparams = []
uncertainties = []
uncertain_indices = []



if query_strategy == "Random":
	phi_values = np.random.uniform(-90, 90, num_new_samples) #phi -90,90 elevation
	theta_values = np.random.uniform(0,360, num_new_samples) #theta 0 - 360 azimuth
	# Create pairs of phi and theta
	vparams_selected = np.dstack([phi_values, theta_values])[0]
	vparams_selected = vparams_selected.tolist()
else:
	with torch.no_grad():
		ssim_values = []
		for i, sample in enumerate(train_loader):
			image = sample["image"].to(device)
			vparams = sample["vparams"].to(device)
			all_vparams.extend(vparams.tolist())

			# Get model predictions
			output = g_model(vparams)

			if query_strategy == "MSELoss":
				uncertainty = nn.MSELoss(reduction='none')(image, output)
				uncertainty1 = uncertainty.sum(dim=(1, 2, 3)) # sum over height,width and channels
				num_samples = num_new_samples
			elif query_strategy == "VGG":
				norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
				norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
				vgg = VGG19('relu1_2').eval()
				# if args.data_parallel and torch.cuda.device_count() > 1:
				# 	vgg = nn.DataParallel(vgg)
				vgg.to(device)
				# normalize
				image = ((image + 1.) * .5 - norm_mean) / norm_std
				output = ((output + 1.) * .5 - norm_mean) / norm_std
				features = vgg(image)
				output_features = vgg(output)
				uncertainty = nn.MSELoss(reduction='none')(features, output_features)        
				uncertainty1 = uncertainty.sum(dim=(1, 2, 3))
				num_samples = num_new_samples
			elif query_strategy =="rand_MSELoss":
				uncertainty = nn.MSELoss(reduction='none')(image, output)
				uncertainty1 = uncertainty.sum(dim=(1, 2, 3))
				num_samples = round(num_new_samples/2)
			elif query_strategy =="rand_VGG":
				norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
				norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
				vgg = VGG19('relu1_2').eval()
				# if args.data_parallel and torch.cuda.device_count() > 1:
				# 	vgg = nn.DataParallel(vgg)
				vgg.to(device)
				# normalize
				image = ((image + 1.) * .5 - norm_mean) / norm_std
				output = ((output + 1.) * .5 - norm_mean) / norm_std
				features = vgg(image)
				output_features = vgg(output)
				uncertainty = nn.MSELoss(reduction='none')(features, output_features)        
				uncertainty1 = uncertainty.sum(dim=(1, 2, 3))
				num_samples = round(num_new_samples/2)
			elif query_strategy =="complexity":
				grayscale_images = transforms.functional.rgb_to_grayscale(image)
				np_array = grayscale_images.cpu().numpy()
				entropies = []
				for j in range(image.shape[0]):
					entropy = skimage.measure.shannon_entropy(np_array[j, 0]) 
					entropies.append(entropy)
				uncertainty1 = np.array(entropies) 
				num_samples = num_new_samples
			elif query_strategy =="rand_complexity":
				grayscale_images = transforms.functional.rgb_to_grayscale(image)
				np_array = grayscale_images.cpu().numpy()
				entropies = []
				for j in range(image.shape[0]):
					entropy = skimage.measure.shannon_entropy(np_array[j, 0]) 
					entropies.append(entropy)
				uncertainty1 = np.array(entropies) 
				num_samples = round(num_new_samples/2)
			elif query_strategy =="diversity":
				for j in range(image.shape[0]):
					for k in range(j+1, image.shape[0]):
						ssim_output = ssim(image[j:j+1], image[k:k+1], data_range=1.0)
						ssim_values.append(ssim_output.item())


				# grayscale_images = transforms.functional.rgb_to_grayscale(image)
				# np_array = grayscale_images.cpu().numpy()
				# entropies = []
				# for i in range(image.shape[0]):
				# 	entropy = skimage.measure.shannon_entropy(np_array[i, 0]) 
				# 	entropies.append(entropy)
				# uncertainty1 = np.array(entropies) 
				# num_samples = round(num_new_samples/2)
			uncertainties.extend(uncertainty1.tolist())
	# Get indices of samples with highest uncertainty
	uncertain_indices = np.argsort(uncertainties)
	ui_selected = uncertain_indices[-num_samples:]
	vparams_selected = [all_vparams[i] for i in ui_selected]
	vparams_selected = vparams2azel(vparams_selected)
	if len(vparams_selected) < 	num_new_samples:
		phi_values = np.random.uniform(-90, 90, num_new_samples-len(vparams_selected)) #phi -90,90 elevation
		theta_values = np.random.uniform(0,360, num_new_samples-len(vparams_selected)) #theta 0 - 360 azimuth
		# Create pairs of phi and theta
		vparams_gen = np.dstack([phi_values, theta_values])[0]
		# vparams_gen = vparams_gen.tolist()
		vparams_selected.extend(vparams_gen.tolist())
	print("SSIM values among all images:", ssim_values)