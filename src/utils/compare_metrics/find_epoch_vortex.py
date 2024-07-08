import torch
import os
import sys
import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'axes.titlesize': 12})
# plt.rcParams.update({'axes.labelsize': 12})
# plt.rcParams.update({'xtick.labelsize': 8})
# plt.rcParams.update({'ytick.labelsize': 8})
# plt.rcParams.update({'legend.fontsize': 8})
# plt.rcParams.update({'figure.titlesize': 14})

plt.rcParams.update({
    'font.size': 22,       # Global font size
    'axes.titlesize': 24,  # Title font size
    'axes.labelsize': 22,  # X and Y label font size
    'xtick.labelsize': 20, # X tick label font size
    'ytick.labelsize': 20, # Y tick label font size
    'legend.fontsize': 20,  # Legend font size
    'figure.titlesize': 26, # Figure title font size
})

loss_plots_dir = '../../../outputs/Vortex/comparison'
chkpt_random = '../../../outputs/Vortex/rand_resume1500ep/model/chk_1499.pth.tar'
chkpt_mse = '../../../outputs/Vortex/mse_resume1500ep/model/chk_1499.pth.tar'
chkpt_rmse = '../../../outputs/Vortex/rand_mse_resume1500ep/model/chk_1499.pth.tar'
chkpt_vgg = '../../../outputs/Vortex/vgg_resume1500ep/model/chk_1499.pth.tar'
chkpt_rvgg = '../../../outputs/Vortex/rand_vgg_resume1500ep/model/chk_1499.pth.tar'
chkpt_comp = '../../../outputs/Vortex/complexity_resume1500ep/model/chk_1499.pth.tar'
chkpt_rcomp = '../../../outputs/Vortex/rand_complexity_1500ep/model/chk_1499.pth.tar'

def load_checkpoint(filepath):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        return checkpoint
    else:
        print(f"=> no checkpoint found at '{filepath}'")
        sys.exit(1)

rcheckpoint = load_checkpoint(chkpt_random)
checkpoint_mse = load_checkpoint(chkpt_mse)
checkpoint_rmse = load_checkpoint(chkpt_rmse)
checkpoint_vgg = load_checkpoint(chkpt_vgg)
checkpoint_rvgg = load_checkpoint(chkpt_rvgg)
checkpoint_comp = load_checkpoint(chkpt_comp)
checkpoint_rcomp = load_checkpoint(chkpt_rcomp)

rtrain_losses = rcheckpoint["train_losses"]
rtest_losses = rcheckpoint["test_losses"]
rtest_ssim = rcheckpoint["test_ssim"]
rtest_psnr = rcheckpoint["test_psnr"]
rtest_lpips = rcheckpoint["test_lpips"]

train_losses_mse = checkpoint_mse["train_losses"]
test_losses_mse = checkpoint_mse["test_losses"]
test_ssim_mse = checkpoint_mse["test_ssim"]
test_psnr_mse = checkpoint_mse["test_psnr"]
test_lpips_mse = checkpoint_mse["test_lpips"]

train_losses_rmse = checkpoint_rmse["train_losses"]
test_losses_rmse = checkpoint_rmse["test_losses"]
test_ssim_rmse = checkpoint_rmse["test_ssim"]
test_psnr_rmse = checkpoint_rmse["test_psnr"]
test_lpips_rmse = checkpoint_rmse["test_lpips"]

train_losses_vgg = checkpoint_vgg["train_losses"]
test_losses_vgg = checkpoint_vgg["test_losses"]
test_ssim_vgg = checkpoint_vgg["test_ssim"]
test_psnr_vgg = checkpoint_vgg["test_psnr"]
test_lpips_vgg = checkpoint_vgg["test_lpips"]

train_losses_rvgg = checkpoint_rvgg["train_losses"]
test_losses_rvgg = checkpoint_rvgg["test_losses"]
test_ssim_rvgg = checkpoint_rvgg["test_ssim"]
test_psnr_rvgg = checkpoint_rvgg["test_psnr"]
test_lpips_rvgg = checkpoint_rvgg["test_lpips"]

train_losses_comp = checkpoint_comp["train_losses"]
test_losses_comp = checkpoint_comp["test_losses"]
test_ssim_comp = checkpoint_comp["test_ssim"]
test_psnr_comp = checkpoint_comp["test_psnr"]
test_lpips_comp = checkpoint_comp["test_lpips"]

train_losses_rcomp = checkpoint_rcomp["train_losses"]
test_losses_rcomp = checkpoint_rcomp["test_losses"]
test_ssim_rcomp = checkpoint_rcomp["test_ssim"]
test_psnr_rcomp = checkpoint_rcomp["test_psnr"]
test_lpips_rcomp = checkpoint_rcomp["test_lpips"]

epochs = list(range(101, 1500 + 1))

f_loss = plt.figure(figsize=(12, 8))
ax_loss = f_loss.add_subplot(111)
# ax_loss.plot(epochs, test_ssim_mse[100:], label="Test SSIM for MSE", color="blue")
ax_loss.plot(epochs, test_ssim_rmse[100:], label="Test SSIM for Random + MSE", color="orange")
# ax_loss.plot(epochs, test_ssim_vgg[100:], label="Test SSIM for VGG", color="green")
ax_loss.plot(epochs, test_ssim_rvgg[100:], label="Test SSIM for Random + VGG", color="red")
# ax_loss.plot(epochs, test_ssim_comp[100:], label="Test SSIM for Complexity", color="purple")
# ax_loss.plot(epochs, test_ssim_rcomp[100:], label="Test SSIM for Random + Complexity", color="brown")
ax_loss.plot(epochs, rtest_ssim[100:], label="Test SSIM for Random", color="purple")

ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("SSIM")
# ax_loss.title("Training and Testing Losses")
ax_loss.legend()
ax_loss.grid(True)

os.makedirs(loss_plots_dir, exist_ok=True)
fname = os.path.join(loss_plots_dir, 'SSIM_plot1.png')

# f_loss.show()
f_loss.savefig(fname, dpi = 300)


# =====================================================Test Loss=======================
f_loss = plt.figure(figsize=(12, 8))
ax_loss = f_loss.add_subplot(111)
# ax_loss.plot(epochs, test_ssim_mse[100:], label="Test SSIM for MSE", color="blue")
ax_loss.plot(epochs, test_losses_rmse[100:], label="Test loss for Random + MSE", color="orange")
# ax_loss.plot(epochs, test_ssim_vgg[100:], label="Test SSIM for VGG", color="green")
# ax_loss.plot(epochs, test_ssim_rvgg[100:], label="Test SSIM for Random + VGG", color="red")
# ax_loss.plot(epochs, test_ssim_comp[100:], label="Test SSIM for Complexity", color="purple")
# ax_loss.plot(epochs, test_ssim_rcomp[100:], label="Test SSIM for Random + Complexity", color="brown")
ax_loss.plot(epochs, rtest_losses[100:], label="Test loss for Random", color="purple")

ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("Test Loss")
# ax_loss.title("Training and Testing Losses")
ax_loss.legend()
ax_loss.grid(True)

os.makedirs(loss_plots_dir, exist_ok=True)
fname = os.path.join(loss_plots_dir, 'Test_loss_plot1.png')

# f_loss.show()
f_loss.savefig(fname, dpi = 300)

# =====================================================PSNR=======================
f_loss = plt.figure(figsize=(12, 8))
ax_loss = f_loss.add_subplot(111)
# ax_loss.plot(epochs, test_ssim_mse[100:], label="Test SSIM for MSE", color="blue")
ax_loss.plot(epochs, test_psnr_rmse[100:], label="Test PSNR for Random + MSE", color="orange")
# ax_loss.plot(epochs, test_ssim_vgg[100:], label="Test SSIM for VGG", color="green")
ax_loss.plot(epochs, test_psnr_rvgg[100:], label="Test PSNR for Random + VGG", color="red")
# ax_loss.plot(epochs, test_ssim_comp[100:], label="Test SSIM for Complexity", color="purple")
# ax_loss.plot(epochs, test_ssim_rcomp[100:], label="Test SSIM for Random + Complexity", color="brown")
ax_loss.plot(epochs, rtest_psnr[100:], label="Test PSNR for Random", color="purple")

ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("PSNR")
# ax_loss.title("Training and Testing Losses")
ax_loss.legend()
ax_loss.grid(True)

os.makedirs(loss_plots_dir, exist_ok=True)
fname = os.path.join(loss_plots_dir, 'PSNR_plot1.png')

# f_loss.show()
f_loss.savefig(fname, dpi = 300)

# =====================================================LPIPS=======================
f_loss = plt.figure(figsize=(12, 8))
ax_loss = f_loss.add_subplot(111)
# ax_loss.plot(epochs, test_ssim_mse[100:], label="Test SSIM for MSE", color="blue")
ax_loss.plot(epochs, test_lpips_rmse[100:], label="Test LPIPS for Random + MSE", color="orange")
# ax_loss.plot(epochs, test_ssim_vgg[100:], label="Test SSIM for VGG", color="green")
ax_loss.plot(epochs, test_lpips_rvgg[100:], label="Test LPIPS for Random + VGG", color="red")
# ax_loss.plot(epochs, test_ssim_comp[100:], label="Test SSIM for Complexity", color="purple")
# ax_loss.plot(epochs, test_ssim_rcomp[100:], label="Test SSIM for Random + Complexity", color="brown")
ax_loss.plot(epochs, rtest_lpips[100:], label="Test LPIPS for Random", color="purple")

ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("LPIPS")
# ax_loss.title("Training and Testing Losses")
ax_loss.legend()
ax_loss.grid(True)

os.makedirs(loss_plots_dir, exist_ok=True)
fname = os.path.join(loss_plots_dir, 'LPIPS_plot1.png')

# f_loss.show()
f_loss.savefig(fname, dpi = 300)