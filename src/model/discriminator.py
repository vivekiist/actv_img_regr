# Discriminator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resblock import FirstBlockDiscriminator, BasicBlockDiscriminator

class Discriminator(nn.Module):
	def __init__(self, dvp=3, dvpe=512, ch=64):
		super(Discriminator, self).__init__()
		self.dvp = dvp
		self.dvpe = dvpe
		self.ch = ch

		# view parameters subnet
		self.vparams_subnet = nn.Sequential(
			nn.Linear(dvp, dvpe), nn.ReLU(),
			nn.Linear(dvpe, dvpe), nn.ReLU(),
			nn.Linear(dvpe, ch * 16),
			nn.ReLU()
		)

		# image classification subnet
		self.img_subnet = nn.Sequential(
			FirstBlockDiscriminator(3, ch, kernel_size=3, stride=1, padding=1), #in 128X128 with ch =3 out 64x64 with ch = 64
			BasicBlockDiscriminator(ch, ch * 2, kernel_size=3, stride=1, padding=1), #in 64x64 ch=64, out 32x32 ch=128
			BasicBlockDiscriminator(ch * 2, ch * 4, kernel_size=3, stride=1, padding=1), #in 32x32  ch=128, out 16x16  ch=256
			BasicBlockDiscriminator(ch * 4, ch * 8, kernel_size=3, stride=1, padding=1), #in 16x16 ch=256, out 8x8  ch=512
			BasicBlockDiscriminator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1), #in 8x8  ch=512, out 4x4  ch=512
			BasicBlockDiscriminator(ch * 8, ch * 16, kernel_size=3, stride=1, padding=1, downsample=False), #in 4x4  ch=512, out 4x4  ch=1024
			BasicBlockDiscriminator(ch * 16, ch * 16, kernel_size=3, stride=1, padding=1, downsample=False), #in 4x4  ch=1024, out 4x4  ch=1024
			nn.ReLU()
		)

		# output subnets
		self.out_subnet = nn.Sequential(nn.Linear(ch * 16, 1))

	def forward(self, vp, x):
		# print("vp shape before vparams_subnet:", vp.shape)
		vp = self.vparams_subnet(vp)
		# print("vp shape after vparams_subnet:", vp.shape)
		
		# print("x shape before img_subnet:", x.shape)
		x = self.img_subnet(x)
		# print("x shape after img_subnet:", x.shape)
		
		x = torch.sum(x, (2, 3)) #global sum pooling
		# print("x shape after global sum pooling:", x.shape)

		out = self.out_subnet(x)
		# print("out shape after out_subnet:", out.shape)
		
		out += torch.sum(vp * x, 1, keepdim=True) #use projection to incorporate the conditional information into the discriminator of GANs
		# print("out shape after adding vp * x:", out.shape)

		return out

# if __name__ == "__main__":
# 	d_model = Discriminator(dvp=3, dvpe=512, ch=64)
# 	dummy_vp = torch.randn(1, 3)
# 	dummy_x = torch.randn(1, 3, 128, 128)
# 	d_model(dummy_vp, dummy_x)