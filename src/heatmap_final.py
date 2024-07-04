import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp2d

# Set default font sizes
plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'axes.titlesize': 32})
plt.rcParams.update({'axes.labelsize': 32})
plt.rcParams.update({'xtick.labelsize': 28})
plt.rcParams.update({'ytick.labelsize': 28})
plt.rcParams.update({'legend.fontsize': 28})
plt.rcParams.update({'figure.titlesize': 34})

root='../data/vortex_volume_images/hm/'
param_file = 'vortex_viewparams_hm.csv'
file = '../data/vortex_volume_images/7/train/vortex_viewparams_train.csv'


num_samples = 100
phi_values = np.linspace(-90, 90, num_samples) #phi -90,90 elevation
theta_values = np.linspace(0, 360, num_samples) #theta 0 - 360 azimuth
X, Y = np.meshgrid(phi_values, theta_values)
XY_pairs = list(zip(X.ravel(), Y.ravel()))

colnames=['phi', 'theta']
data = pd.read_csv(os.path.join(root, param_file), sep=",", names=colnames, header=None)
entropies = []

for i in range(len(XY_pairs)):
    filename = os.path.join(root, f"{XY_pairs[i][0]:.4f}_{XY_pairs[i][1]:.4f}.png")
    img = Image.open(filename)
    entropies.append(img.entropy())	


data['entropy'] = pd.Series(entropies)
Z = np.array(entropies).reshape((num_samples, num_samples))

f = interp2d(phi_values, theta_values, Z, kind='cubic')
xnew = np.linspace(-90, 90, 1000)
ynew = np.linspace(0, 360, 1000)
Znew = f(xnew, ynew)

plt.figure(figsize=(12, 14))
plt.imshow(Znew, aspect='auto', cmap='viridis', extent=[xnew.min(), xnew.max(), ynew.min(), ynew.max()])

# Set ticks for x and y axes
plt.xticks(ticks=np.arange(-90, 91, 30), labels=np.arange(-90, 91, 30), rotation=45)
plt.yticks(ticks=np.arange(0, 361, 60), labels=np.arange(0, 361, 60))

# Add a colorbar
cbar = plt.colorbar(label='Entropy', location='top', orientation='horizontal', pad=0.05)
cbar.set_label('Entropy', labelpad=7, fontsize=30)
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.xaxis.set_label_position('top')


indata_train = np.loadtxt(file,delimiter=',')
phi_value = indata_train[:,0]
theta_value = indata_train[:,1] % 360
phi_value = ((phi_value + 90) % 180) - 90


plt.scatter(phi_value, theta_value,c='r',s=3, label='Training Data')


# plt.colorbar(label='Z value')
plt.xlabel(r'Elevation ($\phi$)')
plt.ylabel(r'Azimuth ($\theta$)')
# plt.title('Entropy Heatmap with Training Data Overlay')

# Add legend
# plt.legend(loc='lower right', bbox_to_anchor=(1, -0.15))

plt.savefig('../outputs/Vortex/entropy_train_plots/7.png', dpi = 300)
# plt.show()
