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
      FirstBlockDiscriminator(3, ch, kernel_size=3, stride=1, padding=1), #in 128X128 with ch =3 out 64x64 with ch = 1
      BasicBlockDiscriminator(ch, ch * 2, kernel_size=3, stride=1, padding=1), #in 64x64 out 32x32
      BasicBlockDiscriminator(ch * 2, ch * 4, kernel_size=3, stride=1, padding=1), #in 32x32 out 16x16
      BasicBlockDiscriminator(ch * 4, ch * 8, kernel_size=3, stride=1, padding=1), #in 16x16 out 8x8
      BasicBlockDiscriminator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1), #in 8x8 out 4x4
      BasicBlockDiscriminator(ch * 8, ch * 16, kernel_size=3, stride=1, padding=1, downsample=False), #in 4x4 out 4x4
      nn.ReLU()
    )

    # output subnets
    self.out_subnet = nn.Sequential(nn.Linear(ch * 16, 1))

  def forward(self, vp, x):
    vp = self.vparams_subnet(vp)
    x = self.img_subnet(x)
    x = torch.sum(x, (2, 3)) #global sum pooling

    out = self.out_subnet(x)
    out += torch.sum(vp * x, 1, keepdim=True) #use projection to incorporate the conditional information into the discriminator of GANs

    return out

# if __name__ == "__main__":
#     d_model = Discriminator(dvp=3, dvpe=512, ch=1)
#     dummy_vp = torch.randn(1, 3)
#     dummy_x = torch.randn(1, 3, 128, 128)
#     d_model(dummy_vp, dummy_x)