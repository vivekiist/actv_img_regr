# Generator architecture

import torch.nn as nn
import torch.nn.functional as F

from model.resblock import BasicBlockGenerator
class Generator(nn.Module):
  def __init__(self, dvp=3, dvpe=512, ch=64):
    super(Generator, self).__init__()

    self.dvp = dvp
    self.dvpe = dvpe
    self.ch = ch

    # view parameters subnet
    self.vparams_subnet = nn.Sequential(
      nn.Linear(dvp, dvpe), nn.ReLU(),
      nn.Linear(dvpe, dvpe), nn.ReLU(),
      nn.Linear(dvpe, ch * 16 * 4 * 4, bias=False)
    )

    # image generation subnet
    self.img_subnet = nn.Sequential(
      # BasicBlockGenerator(ch * 16, ch * 16, kernel_size=3, stride=1, padding=1), # in = 2x2, out = 4x4 
      BasicBlockGenerator(ch * 16, ch * 8, kernel_size=3, stride=1, padding=1), # in = 4x4, out = 8x8 
      BasicBlockGenerator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1), # in = 8x8, out = 16x16, adds one upsample step 
      BasicBlockGenerator(ch * 8, ch * 4, kernel_size=3, stride=1, padding=1), # in = 16x16, out = 32x32
      BasicBlockGenerator(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1), # in = 32x32, out = 64x64
      BasicBlockGenerator(ch * 2, ch, kernel_size=3, stride=1, padding=1), # in = 64x64, out = 128x128
      nn.BatchNorm2d(ch),
      nn.ReLU(),
      nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1),
      nn.Tanh()
    )

  def forward(self, vp):
    vp = self.vparams_subnet(vp)

    x = vp.view(vp.size(0), self.ch * 16, 4, 4) # batch size, channels, height, width
    x = self.img_subnet(x)

    return x

# import torch
# if __name__ == "__main__":
#     g_model = Generator(dvp=3, dvpe=512, ch=1)
#     dummy_vp = torch.randn(1, 3)
#     g_model(dummy_vp)