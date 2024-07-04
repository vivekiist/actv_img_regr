# from scipy.interpolate import interp2d
# from scipy.interpolate import SmoothBivariateSpline, LinearNDInterpolator
import skimage
from torchvision import transforms
from torch.utils.data import DataLoader
from model.isabel import *

# import seaborn as sns
import matplotlib.pyplot as plt

#phi -90,90 elevation
#theta 0 - 360 azimuth

kwargs = {"num_workers": 8, "pin_memory": True}
train_dataset = IsabelDataset(
				root='../data/Isabel_pressure_volume_images/hm/',
				param_file = 'isabel_pr_viewparams_hm2.csv',
				train=True,
				test=False,
				transform=transforms.Compose([ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size=1,shuffle=True, **kwargs)
device = torch.device("cuda:0")

entropies = []
for i, sample in enumerate(train_loader):
	image = sample["image"].to(device)
	vparams = sample["vparams"].to(device)

	# image = (image * 127.5) + 127.5
	grayscale_images = transforms.functional.rgb_to_grayscale(image)
	np_array = grayscale_images.cpu().numpy()
	for i in range(image.shape[0]):
		entropy = skimage.measure.shannon_entropy(np_array[i, 0]) 
		entropies.append(entropy)

train_dataset.data['entropy'] = pd.Series(entropies)

# Extract view parameters and entropy
x = train_dataset.data['phi'].values
y = train_dataset.data['theta'].values

Z = train_dataset.data['entropy'].values

# f = interp2d(x, y, Z, kind='cubic')



# plt.figure(figsize=(10, 10))
# plt.imshow(Z, aspect='auto', cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()])

# # Setting the ticks for x and y axes
# plt.xticks(ticks=x[::20], labels=np.round(x[::20], 2), rotation=45)
# plt.yticks(ticks=y[::20], labels=np.round(y[::20], 2))

# plt.colorbar(label='Z value')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.title('Interpolated Heatmap with Matplotlib')

# plt.savefig('interpolated_heatmap1.png')


# plt.show()