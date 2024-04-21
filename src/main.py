import sys
import logging.config
import os
from datetime import datetime

import subprocess
import random


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms, utils

from model.isabel import *


from model.generator import Generator
from model.discriminator import Discriminator
from model.vgg19 import VGG19

from utils.logger_utils import setup_logger
from utils.config_setter import load_config

# from data_processing.processor import process_data
# from model_training.trainer import train

# CONFIG_DIR = "../config"
LOG_DIR = "../logs"
main_log_file = 'main_logger.log'
run_log_file = 'run_logger.log'

def weights_init(m):
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		nn.init.orthogonal_(m.weight)
		if m.bias is not None:
			nn.init.zeros_(m.bias)

def add_sn(m):
	for name, c in m.named_children():
		m.add_module(name, add_sn(c))
	if isinstance(m, (nn.Linear, nn.Conv2d)):
		return nn.utils.spectral_norm(m, eps=1e-4)
	else:
		return m

def genRandList(start, end, num):
    res = []
    for j in range(num): 
        res.append(random.uniform(start, end)) 
    return res 

def vparams2azel(vparams):
	azel = []
	for vp in vparams:
		angle_rad = np.arctan2(vp[1], vp[0])
		theta_value = np.rad2deg(angle_rad)
		phi_value = vp[2] * 90.0
		azel.append([phi_value, theta_value])

	return azel

def select_uncertain_samples(args, model, train_loader):
	model.eval()
	all_vparams = []
	uncertainties = []
	uncertain_indices = []

	# select device
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:0" if args.cuda else "cpu")

	with torch.no_grad():
		for i, sample in enumerate(train_loader):
			image = sample["image"].to(device)
			vparams = sample["vparams"].to(device)
			all_vparams.extend(vparams.tolist())
			# Get model predictions
			output = model(vparams)
			uncertainty = calculate_uncertainty(args, output, image, vgg = None)
			uncertainties.extend(uncertainty.tolist())

	# Get indices of samples with highest uncertainty
	uncertain_indices = np.argsort(uncertainties)
	ui_selected = uncertain_indices[-args.num_new_samples:]
	vparams_selected = [all_vparams[i] for i in ui_selected]

	return ui_selected, vparams_selected

def calculate_uncertainty(args, output, image, vgg=None):
	# select device
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:0" if args.cuda else "cpu")
	losstype = args.query_strategy
	if losstype == "MSELoss":
		uncertainty = nn.MSELoss(reduction='none')(image, output)
		uncertainty1 = uncertainty.sum(dim=(1, 2, 3)) # sum over height,width and channels
	elif losstype == "VGG":
		norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
		norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
		if vgg is None:
			vgg = VGG19('relu1_2').eval()
			if args.data_parallel and torch.cuda.device_count() > 1:
				vgg = nn.DataParallel(vgg)
			vgg.to(device)
		# normalize
		image = ((image + 1.) * .5 - norm_mean) / norm_std
		output = ((output + 1.) * .5 - norm_mean) / norm_std
		features = vgg(image)
		output_features = vgg(output)
		uncertainty = nn.MSELoss(reduction='none')(features, output_features)        
		uncertainty1 = uncertainty.sum(dim=(1, 2, 3))
	elif losstype =="complexity":
		# https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html#calculate-entropy-of-text
		# # https://unimatrixz.com/blog/latent-space-image-quality-with-entropy/#python-libraries-for-image-entropy-calculation
		# image = cv2.imread(image_path)
		# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# _bins = 128
		# hist, _ = np.histogram(gray_image.ravel(), bins=_bins, range=(0, _bins))
		# prob_dist = hist / hist.sum()
		# image_entropy = entropy(prob_dist, base=2)
		# print(f"Image Entropy {image_entropy}")
		pass
	return uncertainty1


# if __name__ == "__main__":    
# setup_logging()
main_logger, main_file_handler = setup_logger(LOG_DIR, main_log_file, mod_name = __name__)
main_logger.info('Starting the application...')
args = load_config()

# select device
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
args.root_out_dir = os.path.join(args.root_out_dir, timestamp)
os.makedirs(args.root_out_dir, exist_ok=True)

# set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# dataset creation
train_dataset = IsabelDataset(
    root=args.root_dir_train,
	param_file = args.param_file_train,
    train=True,
    test=False,
    transform=transforms.Compose([Normalize(), ToTensor()]))
main_logger.info('Train dataset created.')

test_dataset = IsabelDataset(
    root=args.root_dir_test,
	param_file = args.param_file_test,
    train=False,
    test=True,
    transform=transforms.Compose([Normalize(), ToTensor()]))
main_logger.info('Test dataset created.')

kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=True, **kwargs)
main_logger.info('Train & Test dataloader created.')

# model
g_model = Generator(dvp=args.dvp,
					dvpe=args.dvpe,
					ch=args.ch)
g_model.apply(weights_init)

if args.data_parallel and torch.cuda.device_count() > 1:
	g_model = nn.DataParallel(g_model)
g_model.to(device)
main_logger.info('Generator initialised.')

# setup discriminator model if using GAN loss
if args.use_gan_loss:
	d_model = Discriminator(dvp=args.dvp,
							dvpe=args.dvpe,
							ch=args.ch)
	d_model.apply(weights_init)
	if args.sn:
		d_model = add_sn(d_model)

	if args.data_parallel and torch.cuda.device_count() > 1:
		d_model = nn.DataParallel(d_model)
	d_model.to(device)
	main_logger.info('Discriminator initialised.')

# loss
if args.use_vgg_loss:
	norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
	norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
	vgg = VGG19(layer = 'relu1_2').eval()
	if args.data_parallel and torch.cuda.device_count() > 1:
		vgg = nn.DataParallel(vgg)
	vgg.to(device)
	main_logger.info('VGG model initialised.')

# mse_criterion = nn.MSELoss()
mse_criterion = nn.MSELoss(reduction='mean')
train_losses, test_losses = [], []
d_losses, g_losses = [], []
main_logger.info('MSE loss initialised.')

# optimizer
g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
						betas=(args.beta1, args.beta2))
main_logger.info('Optimizer for Generator model initialised.')

if args.use_gan_loss:
	d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr,
							betas=(args.beta1, args.beta2))
	main_logger.info('Optimizer for Discriminator model initialised.')

# load checkpoint
if args.resume:
	checkpointFile = os.path.join(args.root_out_dir, args.chkpt)
	if os.path.isfile(checkpointFile):
		main_logger.info('Loading checkpoint:  %s', checkpointFile)
		checkpoint = torch.load(checkpointFile)
		# # To load checkpoint from a model trained on multiple GPUs
		# for key in checkpoint.keys():
		# 	# Check if the key corresponds to a model state dictionary
		# 	if key.endswith('model_state_dict'):
		# 		state_dict = checkpoint[key]
		# 		state_dict = {k.partition('module.')[2] if k.startswith('module.') else k: v for k, v in state_dict.items()}
		# 		# Update the checkpoint dictionary with the transformed state dictionary
		# 		checkpoint[key] = state_dict
		args.start_epoch = checkpoint["epoch"]
		g_model.load_state_dict(checkpoint["g_model_state_dict"])
		g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
		if args.use_gan_loss:
			d_model.load_state_dict(checkpoint["d_model_state_dict"])
			d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
			d_losses = checkpoint["d_losses"]
			g_losses = checkpoint["g_losses"]
		train_losses = checkpoint["train_losses"]
		test_losses = checkpoint["test_losses"]
		main_logger.info('Loaded epoch %s from checkpoint %s.', checkpoint["epoch"], checkpointFile)

# Active learning loop
# setup logger for the run
run_logger, runlog_file_handler = setup_logger(LOG_DIR, run_log_file, mod_name = 'run_logger')

for epoch in tqdm(range(args.start_epoch, args.epochs)):
	main_logger.info('Entered training loop at epoch : %s.', epoch)
	g_model.train()
	if args.use_gan_loss:
		d_model.train()
	
	train_loss = 0.0
	epoch_losses = [] #why?


	for i, sample in enumerate(train_loader):
		loss = 0.0
		
		image = sample["image"].to(device)
		vparams = sample["vparams"].to(device)
		g_optimizer.zero_grad()
		fake_image = g_model(vparams)

		# gan loss
		if args.use_gan_loss:
			# update discriminator
			d_optimizer.zero_grad()
			decision = d_model(vparams, image)
			d_loss_real = torch.mean(F.relu(1. - decision))  # using hinge loss
			fake_decision = d_model(vparams, fake_image.detach())
			d_loss_fake = torch.mean(F.relu(1. + fake_decision))  # using hinge loss
			d_loss = d_loss_real + d_loss_fake
			d_loss.backward()
			d_optimizer.step()

			# loss of generator
			g_optimizer.zero_grad()
			fake_decision = d_model(vparams, fake_image)
			g_loss = args.gan_loss_weight * torch.mean(fake_decision)  # using hinge loss
			loss += g_loss

		# mse loss
		if args.use_mse_loss:
			mse_loss = mse_criterion(image, fake_image)
			loss += mse_loss

		# perceptual loss
		if args.use_vgg_loss:
			# normalize
			image1 = ((image + 1.) * .5 - norm_mean) / norm_std
			fake_image1 = ((fake_image + 1.) * .5 - norm_mean) / norm_std
			features = vgg(image1)
			fake_features = vgg(fake_image1)
			perc_loss = args.vgg_loss_weight * mse_criterion(features, fake_features)
			loss += perc_loss

		loss.backward()
		g_optimizer.step()
		train_loss += loss.item() * args.batch_size 

		if (epoch%10 == 0 or epoch == args.epochs-1) and i == 0: #save comparison every 10th epoch & last epoch & first batch
			n = min(args.batch_size, 8)
			comparison = torch.cat(
				[image[:n], fake_image.view(args.batch_size, 3, 128, 128)[:n]])
			
			comparison_dir = os.path.join(args.root_out_dir, "images")
			os.makedirs(comparison_dir, exist_ok=True)
			fname = os.path.join(comparison_dir, 'train_' + str(epoch) + '_batch_' + str(i) + ".png")
			save_image(((comparison.cpu() + 1.) * .5),
						fname, nrow=n)

		# log training status
		if i % args.log_every == 0:
			print(f"Train Epoch: {epoch} [Batch {i}/{len(train_loader)} ({100. * i / len(train_loader):.2f}%)]\tLoss: {(loss.item()):.6f}")
			if args.use_gan_loss:
				print(f"DLoss: {d_loss.item():.6f}, GLoss: {g_loss.item():.6f}")
				d_losses.append(d_loss.item())
				g_losses.append(g_loss.item())
			train_losses.append(loss.item())
			
	print(f"====> Epoch: {epoch} Average loss: {(sum(train_losses[-len(train_loader):])/len(train_loader)):.4f}")
	run_logger.info('Epoch: %s', epoch)
    ##########################################
    ## Active learning section
    ##########################################
	if (not args.no_active_learning) and (len(train_dataset) < args.sampling_budget):
		# select uncertain samples
		print("Selecting uncertain samples")	
		gen_uncertain_indices, gen_vparams = select_uncertain_samples(args, g_model, train_loader)
		run_logger.info('Selected uncertain samples: %s', gen_uncertain_indices)
		run_logger.info('Selected uncertain viewparams: %s', vparams2azel(gen_vparams))
		run_logger.info('Selected len of uncertain samples: %s', len(gen_uncertain_indices))
		run_logger.info('Selected len of uncertain viewparams: %s', len(gen_vparams))

		params = vparams2azel(gen_vparams)
		pvpythonpath = "../../ParaView-5.12.0-egl-MPI-Linux-Python3.10-x86_64/bin/pvbatch"

		phi_new_batch = []
		theta_new_batch = []  
		for vp in params:
			[phi_value, theta_value] = vp
			phi_value = phi_value+np.random.uniform(-5, 5)
			theta_value = theta_value+np.random.uniform(-5, 5)
			phi_new_batch.append(phi_value)
			theta_new_batch.append(theta_value)
			run_logger.info('Generating image for phi: %s, theta: %s', phi_value, theta_value)
			# subprocess.run([pvpythonpath, gen_img_script_path, '--inFile', args.raw_inp_file, '--varName', args.varname, '--phi_val', str(phi_value), '--theta_val', str(theta_value), '--outPath', args.root_dir_train])
		phi_theta_pairs = json.dumps(list(zip(phi_new_batch, theta_new_batch)))
		# phi_theta_pairs = np.column_stack((phi_new_batch, theta_new_batch))
		# phi_theta_pairs = json.dumps(phi_theta_pairs)
		subprocess.run([pvpythonpath, args.data_gen_script, '--inFile', args.raw_inp_file, '--varName', args.varname, '--view_params', phi_theta_pairs, '--outPath', args.root_dir_train])

		# Add newly generated data to the train dataset
		new_data = pd.DataFrame({'phi': phi_new_batch, 'theta': theta_new_batch})
		train_dataset.add_samples(new_data)

		# Update the train dataloader
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
								shuffle=True, **kwargs)

		run_logger.info('Train dataset length: %s', len(train_dataset))
		print(f"Length of Train Dataset: {len(train_dataset)}")

		### generate scatterplot of new modified param space
		if (epoch % 10 == 0  or epoch == args.epochs-1):
			indata_train = np.loadtxt(os.path.join(train_dataset.root, train_dataset.param_file),delimiter=',')
			plt.figure(figsize=(10,10))
			plt.scatter(indata_train[:,0], indata_train[:,1],c='r',s=1)
			splot_dir = os.path.join(args.root_out_dir, 'scatter_plots')
			os.makedirs(splot_dir, exist_ok=True)
			splot_fname = os.path.join(splot_dir, 'plot_epoch_' + str(epoch) + '_batch_' + str(i) + ".png")
			plt.savefig(splot_fname)

	# Testing loss on test data set for each epoch
	g_model.eval()
	if args.use_gan_loss:
		d_model.eval()
	test_loss = 0.
	with torch.no_grad():
		for i, sample in enumerate(test_loader):
			image = sample["image"].to(device)
			vparams = sample["vparams"].to(device)
			fake_image = g_model(vparams)
			# test_loss += mse_criterion(image, fake_image).item() * len(image)
			test_loss += mse_criterion(image, fake_image).item()

			if (epoch%10 == 0 or epoch == args.epochs-1) and i == 0: #save comparison every 10th epoch & last epoch & first batch
				n = min(args.batch_size, 8)
				comparison = torch.cat(
					[image[:n], fake_image.view(args.batch_size, 3, 128, 128)[:n]])
				
				comparison_dir = os.path.join(args.root_out_dir, "images")
				os.makedirs(comparison_dir, exist_ok=True)
				fname = os.path.join(comparison_dir, 'test_' + str(epoch) + '_batch_' + str(i) + ".png")
				save_image(((comparison.cpu() + 1.) * .5),
							fname, nrow=n)
			test_losses.append(test_loss)
			print (f"Test set loss for epoch {epoch}, batch {i} is {test_loss}")
	avg = sum(test_losses[-len(test_loader):])/len(test_loader)
	print(f"====> Epoch: {epoch} Average for Test set loss: {avg:.4f}\n\n")
	# saving...
	if ((epoch+1) % args.check_every) == 0:
		print("=> saving checkpoint at epoch {}".format(epoch+1))
		model_dir = os.path.join(args.root_out_dir, "model")
		os.makedirs(model_dir, exist_ok=True)
		chkname = os.path.join(model_dir, 'chk_' + str(epoch) + ".pth.tar")
		mname = os.path.join(model_dir, 'model_' + str(epoch) + ".pth")
		if args.use_gan_loss:
			torch.save({"epoch": epoch + 1,
						"g_model_state_dict": g_model.state_dict(),
						"g_optimizer_state_dict": g_optimizer.state_dict(),
						"d_model_state_dict": d_model.state_dict(),
						"d_optimizer_state_dict": d_optimizer.state_dict(),
						"d_losses": d_losses,
						"g_losses": g_losses,
						"train_losses": train_losses,
						"test_losses": test_losses},
						chkname)
		else:
			torch.save({"epoch": epoch + 1,
						"g_model_state_dict": g_model.state_dict(),
						"g_optimizer_state_dict": g_optimizer.state_dict(),
						"train_losses": train_losses,
						"test_losses": test_losses},
						chkname)

		torch.save(g_model.state_dict(), mname)
		main_logger.info('Model & Checkpoint saved at epoch %s', epoch)

		
		runlog_file_handler.close()

	main_logger.info('Exiting the application...')
	main_file_handler.close()