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
import skimage.measure
import piq


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
# LOG_DIR = "../logs"
LOG_DIR = "logs"

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

	if args.query_strategy == "Random":
		phi_values = np.random.uniform(-90, 90, args.num_new_samples) #phi -90,90 elevation
		theta_values = np.random.uniform(0,360, args.num_new_samples) #theta 0 - 360 azimuth
		# Create pairs of phi and theta
		vparams_selected = np.dstack([phi_values, theta_values])[0]
		vparams_selected = vparams_selected.tolist()
	else:
		with torch.no_grad():
			for i, sample in enumerate(train_loader):
				image = sample["image"].to(device)
				vparams = sample["vparams"].to(device)
				all_vparams.extend(vparams.tolist())

				# Get model predictions
				output = g_model(vparams)

				if args.query_strategy == "MSELoss":
					uncertainty = nn.MSELoss(reduction='none')(image, output)
					uncertainty1 = uncertainty.sum(dim=(1, 2, 3)) # sum over height,width and channels
					num_samples = args.num_new_samples
				elif args.query_strategy == "VGG":
					norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
					norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
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
					num_samples = args.num_new_samples
				elif args.query_strategy =="rand_MSELoss":
					uncertainty = nn.MSELoss(reduction='none')(image, output)
					uncertainty1 = uncertainty.sum(dim=(1, 2, 3))
					num_samples = round(args.num_new_samples/2)
				elif args.query_strategy =="rand_VGG":
					norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
					norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
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
					num_samples = round(args.num_new_samples/2)
				elif args.query_strategy =="complexity":
					grayscale_images = transforms.functional.rgb_to_grayscale(image)
					np_array = grayscale_images.cpu().numpy()
					entropies = []
					for i in range(image.shape[0]):
						entropy = skimage.measure.shannon_entropy(np_array[i, 0]) 
						entropies.append(entropy)
					uncertainty1 = np.array(entropies) 
					num_samples = args.num_new_samples
				elif args.query_strategy =="rand_complexity":
					grayscale_images = transforms.functional.rgb_to_grayscale(image)
					np_array = grayscale_images.cpu().numpy()
					entropies = []
					for i in range(image.shape[0]):
						entropy = skimage.measure.shannon_entropy(np_array[i, 0]) 
						entropies.append(entropy)
					uncertainty1 = np.array(entropies) 
					num_samples = round(args.num_new_samples/2)
				# elif args.query_strategy =="diversity":
				# 	grayscale_images = transforms.functional.rgb_to_grayscale(image)
				# 	np_array = grayscale_images.cpu().numpy()
				# 	entropies = []
				# 	for i in range(image.shape[0]):
				# 		entropy = skimage.measure.shannon_entropy(np_array[i, 0]) 
				# 		entropies.append(entropy)
				# 	uncertainty1 = np.array(entropies) 
				# 	num_samples = round(args.num_new_samples/2)
				uncertainties.extend(uncertainty1.tolist())
		# Get indices of samples with highest uncertainty
		uncertain_indices = np.argsort(uncertainties)
		ui_selected = uncertain_indices[-num_samples:]
		vparams_selected = [all_vparams[i] for i in ui_selected]
		vparams_selected = vparams2azel(vparams_selected)
		if len(vparams_selected) < 	args.num_new_samples:
			phi_values = np.random.uniform(-90, 90, args.num_new_samples-len(vparams_selected)) #phi -90,90 elevation
			theta_values = np.random.uniform(0,360, args.num_new_samples-len(vparams_selected)) #theta 0 - 360 azimuth
			# Create pairs of phi and theta
			vparams_gen = np.dstack([phi_values, theta_values])[0]
			# vparams_gen = vparams_gen.tolist()
			vparams_selected.extend(vparams_gen.tolist())

		return vparams_selected

# def calculate_uncertainty(args, output, image, vgg=None):
# 	# select device
# 	args.cuda = not args.no_cuda and torch.cuda.is_available()
# 	device = torch.device("cuda:0" if args.cuda else "cpu")
# 	losstype = args.query_strategy
# 	if losstype == "MSELoss":
# 		uncertainty = nn.MSELoss(reduction='none')(image, output)
# 		uncertainty1 = uncertainty.sum(dim=(1, 2, 3)) # sum over height,width and channels
# 	elif losstype == "VGG":
# 		norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
# 		norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
# 		if vgg is None:
# 			vgg = VGG19('relu1_2').eval()
# 			if args.data_parallel and torch.cuda.device_count() > 1:
# 				vgg = nn.DataParallel(vgg)
# 			vgg.to(device)
# 		# normalize
# 		image = ((image + 1.) * .5 - norm_mean) / norm_std
# 		output = ((output + 1.) * .5 - norm_mean) / norm_std
# 		features = vgg(image)
# 		output_features = vgg(output)
# 		uncertainty = nn.MSELoss(reduction='none')(features, output_features)        
# 		uncertainty1 = uncertainty.sum(dim=(1, 2, 3))
# 	elif losstype =="rand_MSELoss":
# 		uncertainty = nn.MSELoss(reduction='none')(image, output)
# 		uncertainty1 = uncertainty.sum(dim=(1, 2, 3))
# 	elif losstype =="complexity":
# 		# https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html#calculate-entropy-of-text
# 		# # https://unimatrixz.com/blog/latent-space-image-quality-with-entropy/#python-libraries-for-image-entropy-calculation
# 		# image = cv2.imread(image_path)
# 		# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 		# _bins = 128
# 		# hist, _ = np.histogram(gray_image.ravel(), bins=_bins, range=(0, _bins))
# 		# prob_dist = hist / hist.sum()
# 		# image_entropy = entropy(prob_dist, base=2)
# 		# print(f"Image Entropy {image_entropy}")
# 		pass
# 	return uncertainty1


# if __name__ == "__main__":    
# setup_logging()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
args = load_config(timestamp)
# LOG_DIR = os.path.join(LOG_DIR, timestamp)
args.root_out_dir = os.path.join(args.root_out_dir, timestamp)
os.makedirs(args.root_out_dir, exist_ok=True)
LOG_DIR = os.path.join(args.root_out_dir, LOG_DIR)
main_logger, main_file_handler = setup_logger(LOG_DIR, main_log_file, mod_name = __name__)
main_logger.info('Starting the application...')


# select device
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")




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
batch_train_losses, train_losses, batch_test_losses, batch_ssim, batch_psnr, test_losses, test_ssim, test_psnr = [], [], [], [], [], [], [], []
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
	# checkpointFile = os.path.join(args.root_out_dir, args.chkpt)
	if os.path.isfile(args.chkpt):
		main_logger.info('Loading checkpoint:  %s', args.chkpt)
		checkpoint = torch.load(args.chkpt)
		args.start_epoch = checkpoint["epoch"]
		g_model.load_state_dict(checkpoint["g_model_state_dict"], strict=False)
		g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
		if args.use_gan_loss:
			d_model.load_state_dict(checkpoint["d_model_state_dict"], strict=False)
			d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
			d_losses = checkpoint["d_losses"]
			g_losses = checkpoint["g_losses"]
		batch_train_losses = checkpoint["batch_train_losses"]
		batch_test_losses = checkpoint["batch_test_losses"]
		batch_ssim = checkpoint["batch_ssim"]
		batch_psnr = checkpoint["batch_psnr"]
		train_losses = checkpoint["train_losses"]
		test_losses = checkpoint["test_losses"]
		test_ssim = checkpoint["test_ssim"]
		test_psnr = checkpoint["test_psnr"]
		main_logger.info('Loaded epoch %s from checkpoint %s.', checkpoint["epoch"], args.chkpt)

# Active learning loop
# setup logger for the run
run_logger, runlog_file_handler = setup_logger(LOG_DIR, run_log_file, mod_name = 'run_logger')

for epoch in tqdm(range(args.start_epoch, args.epochs)):
	main_logger.info('Entered training loop at epoch : %s.', epoch)
	g_model.train()
	if args.use_gan_loss:
		d_model.train()
	
	train_loss = 0.0

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
			loss += g_loss * len(image)/len(train_loader.dataset)

		# mse loss
		if args.use_mse_loss:
			mse_loss = args.mse_loss_weight * mse_criterion(image, fake_image)
			loss += mse_loss * len(image)/len(train_loader.dataset)

		# perceptual loss
		if args.use_vgg_loss:
			# normalize
			image1 = ((image + 1.) * .5 - norm_mean) / norm_std
			fake_image1 = ((fake_image + 1.) * .5 - norm_mean) / norm_std
			features = vgg(image1)
			fake_features = vgg(fake_image1)
			perc_loss = args.vgg_loss_weight * mse_criterion(features, fake_features)
			loss += perc_loss * len(image)/len(train_loader.dataset)

		loss.backward()
		g_optimizer.step()
		# train_loss += loss.item() * len(image)

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
		# print(f"Train Epoch: {epoch} [Batch {i+1}/{len(train_loader)} ({100. * (i+1) / len(train_loader):.2f}%)]\tLoss: {(loss.item()):.4f}")
		run_logger.info('Train Epoch: %s [Batch %d/%d (%.2f%%)]\t\tLoss: %.4f', epoch, i+1, len(train_loader), 100. * (i+1) / len(train_loader), loss.item())
		if args.use_gan_loss:
			# print(f"DLoss: {d_loss.item():.6f}, GLoss: {g_loss.item():.4f}")
			run_logger.info("DLoss: %.6f, GLoss: %.4f", d_loss.item(), g_loss.item())
			d_losses.append(d_loss.item())
			g_losses.append(g_loss.item())
		batch_train_losses.append(loss.item())
	
	epoch_loss = sum(batch_train_losses[-len(train_loader):])
	print(f"\n====> Epoch: {epoch} Train loss: \t\t\t{epoch_loss:.4f}")
	run_logger.info('====> Epoch: %d Train loss: \t\t\t\t\t%.4f\n', epoch, epoch_loss)
	train_losses.append(epoch_loss)
    ##########################################
    ## Active learning section
    ##########################################
	if (not args.no_active_learning) and (len(train_dataset) < args.sampling_budget):
		# select uncertain samples
		print("Selecting uncertain samples")	
		gen_vparams = select_uncertain_samples(args, g_model, train_loader)
		# run_logger.info('Selected uncertain samples: %s', gen_uncertain_indices)
		run_logger.info('Selected uncertain viewparams: %s', gen_vparams)
		# run_logger.info('Selected len of uncertain samples: %s', len(gen_uncertain_indices))
		# run_logger.info('Selected len of uncertain viewparams: %s', len(gen_vparams))

		
		pvpythonpath = "../../ParaView-5.12.0-egl-MPI-Linux-Python3.10-x86_64/bin/pvbatch"

		phi_new_batch = []
		theta_new_batch = []  
		for vp in gen_vparams:
			[phi_value, theta_value] = vp
			phi_value = phi_value+np.random.uniform(-5, 5)
			theta_value = theta_value+np.random.uniform(-5, 5)
			# # Ensure phi is within -90 to 90
			# phi_value = np.clip(phi_value, -90, 90)
			# # Ensure theta is within 0 to 360
			# theta_value = np.clip(theta_value, 0, 360)

			# Wrap phi values within -90 to 90 degrees
			phi_value = ((phi_value + 180) % 360) - 180
			# Wrap theta values within 0 to 360 degrees
			theta_value = theta_value % 360

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

		run_logger.info('Length of Train Dataset: %s', len(train_dataset))
		print(f"Length of Train Dataset: {len(train_dataset)}")

		### generate scatterplot of new modified param space
		# if ((epoch+1) % 10 == 0 or len(train_dataset) >= 10000):
		if ((epoch+1) % 10 == 0):
			indata_train = np.loadtxt(os.path.join(train_dataset.root, train_dataset.param_file),delimiter=',')
			plt.figure(figsize=(10,10))
			plt.scatter(indata_train[:,0], indata_train[:,1],c='r',s=1)
			splot_dir = os.path.join(args.root_out_dir, 'scatter_plots')
			os.makedirs(splot_dir, exist_ok=True)
			splot_fname = os.path.join(splot_dir, 'plot_epoch_' + str(epoch) + ".png")
			plt.savefig(splot_fname)

	# Testing loss on test data set for each epoch
	g_model.eval()
	if args.use_gan_loss:
		d_model.eval()
	


	with torch.no_grad():
		for i, sample in enumerate(test_loader):
			image = sample["image"].to(device)
			vparams = sample["vparams"].to(device)
			fake_image = g_model(vparams)
			test_loss = 0.
			test_loss += mse_criterion(image, fake_image) * len(image)/len(test_loader.dataset)

			# ssim_loss = 0.
			# psnr_loss = 0.
			image2 = (image+ 1.) * .5
			fake_image2 = (fake_image+ 1.) * .5
			# ssim_loss = piq.ssim(image2, fake_image2, data_range=1.)
			# psnr_loss = piq.psnr(image2, fake_image2, data_range=1., reduction='mean')

			ssim_loss = piq.ssim(image2, fake_image2, data_range=1.) * len(image)/len(test_loader.dataset)
			psnr_loss = piq.psnr(image2, fake_image2, data_range=1., reduction='mean') * len(image)/len(test_loader.dataset)

			# perceptual loss
			if args.use_vgg_loss:
				# normalize
				image1 = ((image + 1.) * .5 - norm_mean) / norm_std
				fake_image1 = ((fake_image + 1.) * .5 - norm_mean) / norm_std
				features = vgg(image1)
				fake_features = vgg(fake_image1)
				perc_loss = args.vgg_loss_weight * mse_criterion(features, fake_features)
				test_loss += perc_loss * len(image)/len(test_loader.dataset)

			if (epoch%10 == 0 or epoch == args.epochs-1) and i == 0: #save comparison every 10th epoch & last epoch & first batch
				n = min(len(image), 8)
				comparison = torch.cat(
					[image[:n], fake_image.view(len(image), 3, 128, 128)[:n]])
				
				comparison_dir = os.path.join(args.root_out_dir, "images")
				os.makedirs(comparison_dir, exist_ok=True)
				fname = os.path.join(comparison_dir, 'test_' + str(epoch) + '_batch_' + str(i) + ".png")
				save_image(((comparison.cpu() + 1.) * .5),
							fname, nrow=n)
			batch_test_losses.append(test_loss.item())
			batch_ssim.append(ssim_loss.item())
			batch_psnr.append(psnr_loss.item())
			# print (f"\tTest set loss for epoch {epoch}, Batch {i+1}/{len(test_loader)} is {test_loss.item():.4f}")
			run_logger.info("\tTest set loss for epoch %d, Batch %d/%d is %.4f", epoch, i+1, len(test_loader), test_loss.item())
	# epoch_ssim = sum(batch_ssim[-len(test_loader):])/len(test_loader)
	# epoch_psnr = sum(batch_psnr[-len(test_loader):])/len(test_loader)
	epoch_ssim = sum(batch_ssim[-len(test_loader):])
	epoch_psnr = sum(batch_psnr[-len(test_loader):])
	epoch_test_loss = sum(batch_test_losses[-len(test_loader):])
	print(f"\t====> Epoch: {epoch} Test loss: \t\t{epoch_test_loss:.4f}")
	run_logger.info("\t====> Epoch: %d Test loss: \t\t\t\t%.4f", epoch, epoch_test_loss)
	run_logger.info("\t====> Epoch: %d SSIM: \t\t\t\t%.4f", epoch, epoch_ssim)
	run_logger.info("\t====> Epoch: %d PSNR: \t\t\t\t%.4f", epoch, epoch_psnr)
	test_losses.append(epoch_test_loss)
	test_ssim.append(epoch_ssim)
	test_psnr.append(epoch_psnr)
	# saving...
	if (((epoch+1) % args.check_every == 0) or (epoch == args.epochs-1)) :
		print("=> saving checkpoint at epoch {}".format(epoch+1))
		run_logger.info("=> saving checkpoint at epoch %d", epoch + 1)
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
						"batch_train_losses": batch_train_losses, 
						"batch_test_losses": batch_test_losses,
						"batch_ssim": batch_ssim,
						"batch_psnr": batch_psnr,
						"train_losses": train_losses,
						"test_losses": test_losses,
						"test_ssim": test_ssim,
						"test_psnr": test_psnr},
						chkname)
		else:
			torch.save({"epoch": epoch + 1,
						"g_model_state_dict": g_model.state_dict(),
						"g_optimizer_state_dict": g_optimizer.state_dict(),
						"batch_train_losses": batch_train_losses,
						"batch_test_losses": batch_test_losses,
						"batch_ssim": batch_ssim,
						"batch_psnr": batch_psnr,
						"train_losses": train_losses,
						"test_losses": test_losses,
						"test_ssim": test_ssim,
						"test_psnr": test_psnr},
						chkname)

		torch.save(g_model.state_dict(), mname)
		main_logger.info('Model & Checkpoint saved at epoch %s', epoch)

	if (epoch == args.epochs-1):
		epochs = list(range(1, epoch + 2))
		plt.figure(figsize=(10, 6))
		plt.plot(epochs, train_losses, label="Train Loss")
		plt.plot(epochs, test_losses, label="Test Loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		# plt.title("Training and Testing Losses")
		plt.legend()
		plt.grid(True)

		loss_plots_dir = os.path.join(args.root_out_dir, "loss_plots")
		os.makedirs(loss_plots_dir, exist_ok=True)
		fname = os.path.join(loss_plots_dir, 'loss_plot_epoch_'+ str(epoch) + ".png")

		# plt.show() 
		plt.savefig(fname)
		main_logger.info('Loss plot saved at epoch %s', epoch)
		print (f"SSIM: {test_ssim[-1]}, PSNR: {test_psnr[-1]}")

runlog_file_handler.close()
main_logger.info('Exiting the application...')
main_file_handler.close()
