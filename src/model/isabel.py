# Copyright 2019 The IEVA-DGM Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Isabel dataset

from __future__ import absolute_import, division, print_function

import os

import pandas as pd
import numpy as np
from skimage import io, transform

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

#phi -90,90 elevation
#theta 0 - 360 azimuth

class IsabelDataset(Dataset):
	def __init__(self, root, param_file, train=False, test=False, transform=None):
		self.root = root
		self.train = train
		self.test = test
		self.param_file = param_file
		self.transform = transform
		colnames=['phi', 'theta']

		self.data = pd.read_csv(os.path.join(root, param_file),
								sep=",", names=colnames, header=None)
		# if self.train:
		# 	self.data = pd.read_csv(os.path.join(root, param_file),
		# 							sep=",", names=colnames, header=None)
		# elif self.test:
		# 	self.data = pd.read_csv(os.path.join(root, param_file),
		# 							sep=",", names=colnames, header=None)
		self.filenames = self.data['phi'].apply(lambda x: f'{x:.4f}') + '_' + self.data['theta'].apply(lambda x: f'{x:.4f}') + '.png'

	def __len__(self):
			return len(self.data)

	def __getitem__(self, index):
		if type(index) == torch.Tensor:
			index = index.item()

		params = self.data.iloc[index]
		img_name = os.path.join(self.root, self.filenames[index])
		vparams = np.zeros(3, dtype=np.float32)
		# vparams[0] = np.cos(np.deg2rad(params.iloc[1]))
		vparams[0] = np.cos(np.deg2rad(params['theta'])) 
		vparams[1] = np.sin(np.deg2rad(params['theta']))
		vparams[2] = params['phi'] / 90.

		image = io.imread(img_name)[:, :, 0:3]
		sample = {"image": image, "vparams": vparams}

		if self.transform:
			sample = self.transform(sample)
		
		return sample

	def add_samples(self, new_data):
		# Assuming new_data is a DataFrame similar to the structure of self.data
		# e.g. new_data = pd.DataFrame({0: [2,4,6], 1: [10,100,1000]})
		self.data = pd.concat([self.data, new_data], ignore_index=True)
		self.filenames = pd.concat([self.filenames, (new_data['phi'].apply(lambda x: f'{x:.4f}') + '_' + new_data['theta'].apply(lambda x: f'{x:.4f}') + '.png')], ignore_index=True)

		# Append new_data to the param_file
		with open(os.path.join(self.root, self.param_file), 'a') as f:
			new_data.to_csv(f, header=False, index=False)


# utility functions
def imshow(image):
	plt.imshow(image.numpy().transpose((1, 2, 0)))

# data transformation
class Resize(object):
	def __init__(self, size):
		assert isinstance(size, (int, tuple))
		self.size = size

	def __call__(self, sample):
		image = sample["image"]
		vparams = sample["vparams"]

		h, w = image.shape[:2]
		if isinstance(self.size, int):
			if h > w:
				new_h, new_w = self.size * h / w, self.size
			else:
				new_h, new_w = self.size, self.size * w / h
		else:
			new_h, new_w = self.size

		new_h, new_w = int(new_h), int(new_w)

		image = transform.resize(
				image, (new_h, new_w), order=1, mode="reflect",
				preserve_range=True, anti_aliasing=True).astype(np.float32)

		return {"image": image, "vparams": vparams}

class Normalize(object):
	def __call__(self, sample):
		image = sample["image"]
		vparams = sample["vparams"]

		image = (image.astype(np.float32) - 127.5) / 127.5

		return {"image": image, "vparams": vparams}

class ToTensor(object):
	def __call__(self, sample):
		image = sample["image"]
		vparams = sample["vparams"]

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return {"image": torch.from_numpy(image),
				"vparams": torch.from_numpy(vparams)}

# # dataset verification
# from torchvision import transforms, utils
# Idataset = IsabelDataset(
#     root="../data/Isabel_pressure_volume_images/train/",
# 	param_file = "isabel_pr_viewparams_train.csv",
#     train=True,
#     test=False,
#     transform=transforms.Compose([Resize(64), Normalize(), ToTensor()]))

# loader = DataLoader(Idataset, batch_size=5, shuffle=True, num_workers=4)

# dataiter = iter(loader)
# samples = next(dataiter)

# print(samples["image"].shape)

# fig = plt.figure()
# imshow(utils.make_grid(((samples["image"] + 1.) * .5)))
# plt.show()
	
# # Add new samples
# new_data = pd.DataFrame({'phi': [2,4,6], 'theta': [10,100,1000]})
# Idataset.add_samples(new_data)
# np.savetxt('./isabel_pr_viewparams.csv', \
#             np.column_stack(([2,4,6], [10,100,1000])), delimiter=',')
# len(Idataset.filenames)
# Idataset.filenames[514]

# from torch.utils.data import Subset
# subset_indices = range(0,500)
# subset_dataset = Subset(Idataset, subset_indices)
# for i, sample in enumerate(subset_dataset):
#   print(i, sample["image"].shape, sample["vparams"].shape)    