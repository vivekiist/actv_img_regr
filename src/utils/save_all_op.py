import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.utils import save_image
from model.isabel import *
from model.generator import Generator
from torchvision import transforms, utils

# chkpt =  "/users/mtech/vivekg/actv_img_regr/outputs/Isabel_mse_active_random/batch-size-sel/b100/model/chk_1699.pth.tar"

device = torch.device("cuda:0")
chkpt = input("Enter the path to the model: ")
checkpoint = torch.load(chkpt)

test_dataset = IsabelDataset(
    root="../data/Isabel_pressure_volume_images/test/",
	param_file = "isabel_pr_viewparams_test.csv",
    train=False,
    test=True,
    transform=transforms.Compose([Normalize(), ToTensor()]))

kwargs = {"num_workers": 4, "pin_memory": True}

test_loader = DataLoader(test_dataset, batch_size=50,
                        shuffle=True, **kwargs)
# model
g_model = Generator(dvp=3, dvpe=512, ch=64)
g_model.to(device)
g_model.load_state_dict(checkpoint["g_model_state_dict"], strict=False)

g_model.eval()
comparison_dir = os.path.join(os.getcwd(), "test_set_comparison")
os.makedirs(comparison_dir, exist_ok=True)

for i, sample in enumerate(test_loader):
    image = sample["image"].to(device)
    vparams = sample["vparams"].to(device)
    fake_image = g_model(vparams)

    batch_size = image.size(0)
    num_rows = batch_size // 10

    # Initialize an empty tensor to hold the rearranged images
    rearranged_images = torch.empty(0, *image.shape[1:], device=device)

    for j in range(num_rows):
        real_images = image[j*10:(j+1)*10]
        fake_images = fake_image[j*10:(j+1)*10]
        rearranged_images = torch.cat((rearranged_images, real_images, fake_images), dim=0)

    # Define filename and save image
    fname = os.path.join(comparison_dir, f'test_batch_{i}.png')
    save_image(((rearranged_images.cpu() + 1.) * 0.5), fname, nrow=10)


# for i, sample in enumerate(test_loader):
# 	image = sample["image"].to(device)
# 	vparams = sample["vparams"].to(device)
# 	fake_image = g_model(vparams)
# 	comparison = torch.stack([image, fake_image], dim=1).view(-1, *image.shape[1:]) #pairs adjacent images
# 	# comparison = torch.cat([image, fake_image], dim=0)
# 	# comparison = torch.cat([image, fake_image.view(len(image), 3, 128, 128)])
# 	comparison_dir = os.path.join(os.getcwd(), "test_set_comparison")
# 	os.makedirs(comparison_dir, exist_ok=True)
# 	fname = os.path.join(comparison_dir, 'test_' + 'batch_' + str(i) + ".png")
# 	save_image(((comparison.cpu() + 1.) * .5), fname, nrow=10)

print("Image saving complete.")
