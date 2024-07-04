import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import skimage
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk
from PIL import Image
from torchvision import transforms

root='../data/Isabel_pressure_volume_images/hm/'
param_file = 'isabel_pr_viewparams_hm2.csv'

colnames=['phi', 'theta']
data = pd.read_csv(os.path.join(root, param_file), sep=",", names=colnames, header=None)
filenames = data['phi'].apply(lambda x: f'{x:.4f}') + '_' + data['theta'].apply(lambda x: f'{x:.4f}') + '.png'

snn_entropies = []
pt_gray_entropies = []
morph_entropies = []
pt_clr_entropies = []
bw_entropies = []
for fn in filenames:
    img = Image.open(os.path.join(root, fn))
    gray_pt = transforms.functional.rgb_to_grayscale(img)
    snn_entropy = skimage.measure.shannon_entropy(gray_pt)



    gray_image = color.rgb2gray(img)
    gray_image = img_as_ubyte(gray_image)
    entropy_image = rank.entropy(gray_image, disk(1))
    mean_entropy = np.mean(entropy_image)
    # print(f'Mean Entropy: {mean_entropy}')
    pt_gray_entropies.append(gray_pt.entropy())
    morph_entropies.append(mean_entropy)
    pt_clr_entropies.append(img.entropy())
    snn_entropies.append(snn_entropy)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(entropy_image, cmap='viridis')
    # plt.colorbar(label='Entropy')
    # plt.title('Entropy Image')
    # plt.show()
data['pt_gray_entropies'] = pd.Series(pt_gray_entropies)
data['pt_clr_entropies'] = pd.Series(pt_clr_entropies)
data['morph_entropy'] = pd.Series(morph_entropies)
data['snn_entropy'] = pd.Series(snn_entropies)
data.to_csv('data_all.csv', index=False)