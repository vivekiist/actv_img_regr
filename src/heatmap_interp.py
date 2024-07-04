# from scipy.interpolate import interp2d
from scipy.interpolate import SmoothBivariateSpline, LinearNDInterpolator
import skimage
from torchvision import transforms
from torch.utils.data import DataLoader
from model.isabel import *

import seaborn as sns
import matplotlib.pyplot as plt

#phi -90,90 elevation
#theta 0 - 360 azimuth

kwargs = {"num_workers": 8, "pin_memory": True}
train_dataset = IsabelDataset(
				root='/data1/vivekg/actv_img_regr/data/Isabel_pressure_volume_images/test',
				param_file = 'Isabel_pressure_viewparams_test.csv',
				train=True,
				test=False,
				transform=transforms.Compose([Normalize(), ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size=400,shuffle=True, **kwargs)
device = torch.device("cuda:0")

entropies = []
for i, sample in enumerate(train_loader):
	image = sample["image"].to(device)
	vparams = sample["vparams"].to(device)
	grayscale_images = transforms.functional.rgb_to_grayscale(image)
	np_array = grayscale_images.cpu().numpy()
	for i in range(image.shape[0]):
		entropy = skimage.measure.shannon_entropy(np_array[i, 0]) 
		entropies.append(entropy)

train_dataset.data['entropy'] = pd.Series(entropies)

# Extract view parameters and entropy
elevation = train_dataset.data['phi'].values
azimuth = train_dataset.data['theta'].values
entropy = train_dataset.data['entropy'].values

xy = np.c_[elevation, azimuth]   # or just list(zip(x, y))
lut2 = LinearNDInterpolator(xy, entropy)

X = np.linspace(min(elevation), max(elevation))
Y = np.linspace(min(azimuth), max(azimuth))
X, Y = np.meshgrid(X, Y)
Z = lut2(X, Y)

fig = plt.figure(figsize=(10, 10))
plt.imshow(Z, aspect='auto', cmap='viridis', extent=[X.min(), X.max(), Y.min(), Y.max()])

# # Setting the ticks for x and y axes
# plt.xticks(ticks=x[::20], labels=np.round(x[::20], 2), rotation=45)
# plt.yticks(ticks=y[::20], labels=np.round(y[::20], 2))

plt.colorbar(label='Z value')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Interpolated Heatmap with Matplotlib')

# plt.savefig('interpolated_heatmap.png')
# plt.show()