import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk

root='../data/Isabel_pressure_volume_images/hm/'
param_file = 'isabel_pr_viewparams_hm2.csv'

colnames=['phi', 'theta']
data = pd.read_csv(os.path.join(root, param_file), sep=",", names=colnames, header=None)
filenames = data['phi'].apply(lambda x: f'{x:.4f}') + '_' + data['theta'].apply(lambda x: f'{x:.4f}') + '.png'


entropies = []
for fn in filenames:
    image = io.imread(os.path.join(root, fn))
    gray_image = color.rgb2gray(image)
    gray_image = img_as_ubyte(gray_image)
    entropy_image = rank.entropy(gray_image, disk(1))
    mean_entropy = np.mean(entropy_image)
    print(f'Mean Entropy: {mean_entropy}')
    entropies.append(mean_entropy)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(entropy_image, cmap='viridis')
    # plt.colorbar(label='Entropy')
    # plt.title('Entropy Image')
    # plt.show()
data['entropy'] = pd.Series(entropies)
data.to_csv('data_mph.csv', index=False)