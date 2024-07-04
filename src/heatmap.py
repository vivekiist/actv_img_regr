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

# # Plotting the heatmap
# plt.figure(figsize=(10, 8))
# # heatmap_data = train_dataset.data.pivot("theta", "phi", "entropy")
# heatmap_data = train_dataset.data.pivot(index='theta', columns='phi', values='entropy')

# sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Entropy'})

# # Overlay points using plt.scatter
# phi_values = train_dataset.data['phi'].values
# theta_values = train_dataset.data['theta'].values
# entropy_values = train_dataset.data['entropy'].values

# # Normalize entropy values for the colormap
# norm = plt.Normalize(vmin=min(entropy_values), vmax=max(entropy_values))
# cmap = plt.get_cmap('viridis')

# # Create scatter points
# for phi, theta, entropy in zip(phi_values, theta_values, entropy_values):
#     plt.scatter(phi, theta, color=cmap(norm(entropy)), s=100, edgecolor='k')  # s is the size of the points



# plt.xlabel('Phi')
# plt.ylabel('Theta')
# plt.title('Entropy Heatmap of Images')
# # plt.show()
# plt.savefig('entropy_heatmap.png')


# Extract view parameters and entropy
elevation = train_dataset.data['phi'].values
azimuth = train_dataset.data['theta'].values
entropy = train_dataset.data['entropy'].values

# Create a 2D grid
unique_azimuth = np.sort(np.unique(azimuth))
unique_elevation = np.sort(np.unique(elevation))
entropy_grid = np.zeros((len(unique_elevation), len(unique_azimuth)))

for i, (az, el, ent) in enumerate(zip(azimuth, elevation, entropy)):
    az_index = np.where(unique_azimuth == az)[0][0]
    el_index = np.where(unique_elevation == el)[0][0]
    entropy_grid[el_index, az_index] = ent

plt.figure(figsize=(12, 8))
sns.heatmap(entropy_grid, cmap='viridis', xticklabels=np.arange(-90, 90, 10), yticklabels=np.arange(0, 360, 10))
plt.xlabel('Azimuth')
plt.ylabel('Elevation')
plt.title('Entropy Heatmap of View Parameters')
plt.show()