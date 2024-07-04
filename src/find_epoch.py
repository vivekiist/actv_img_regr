import torch
import os, sys
import matplotlib.pyplot as plt

loss_plots_dir = '../outputs/Vortex/comparison'
chkpt_random = '../outputs/Vortex/rand_resume1500ep/model/chk_1499.pth.tar'
chkpt_mse = '../outputs/Vortex/mse_resume1500ep/model/chk_1499.pth.tar'
chkpt_rmse = '../outputs/Vortex/rand_mse_resume1500ep/model/chk_1499.pth.tar'
chkpt_vgg = '../outputs/Vortex/vgg_resume1500ep/model/chk_1499.pth.tar'
chkpt_rvgg = '../outputs/Vortex/rand_vgg_resume1500ep/model/chk_1499.pth.tar'
chkpt_comp = '../outputs/Vortex/complexity_resume1500ep/model/chk_1499.pth.tar'
chkpt_rcomp = '../outputs/Vortex/rand_complexity_1500ep/model/chk_1499.pth.tar'


if os.path.isfile(chkpt_random):
	rcheckpoint = torch.load(chkpt_random)
	rtrain_losses = rcheckpoint["train_losses"]
	rtest_losses = rcheckpoint["test_losses"]
	rtest_ssim = rcheckpoint["test_ssim"]
	rtest_psnr = rcheckpoint["test_psnr"]
	rtest_lpips = rcheckpoint["test_lpips"]
else:
	print("=> no checkpoint found at '{}'".format(chkpt_random))
	sys.exit(1)

if os.path.isfile(chkpt_mse):
	checkpoint = torch.load(chkpt_mse)
	train_losses_mse = checkpoint["train_losses"]
	test_losses_mse = checkpoint["test_losses"]
	test_ssim_mse = checkpoint["test_ssim"]
	test_psnr_mse = checkpoint["test_psnr"]
	test_lpips_mse = checkpoint["test_lpips"]
else:
	print("=> no checkpoint found at '{}'".format(chkpt_mse))
	sys.exit(1)
	
if os.path.isfile(chkpt_rmse):
    checkpoint = torch.load(chkpt_rmse)
    train_losses_rmse = checkpoint["train_losses"]
    test_losses_rmse = checkpoint["test_losses"]
    test_ssim_rmse = checkpoint["test_ssim"]
    test_psnr_rmse = checkpoint["test_psnr"]
    test_lpips_rmse = checkpoint["test_lpips"]  
else:    
    print("=> no checkpoint found at '{}'".format(chkpt_rmse))
    sys.exit(1)
	
if os.path.isfile(chkpt_vgg):
    checkpoint = torch.load(chkpt_vgg)
    train_losses_vgg = checkpoint["train_losses"]
    test_losses_vgg = checkpoint["test_losses"]
    test_ssim_vgg = checkpoint["test_ssim"]
    test_psnr_vgg = checkpoint["test_psnr"]
    test_lpips_vgg = checkpoint["test_lpips"]
else:    
    print("=> no checkpoint found at '{}'".format(chkpt_vgg))
    sys.exit(1)

if os.path.isfile(chkpt_rvgg):
    checkpoint = torch.load(chkpt_rvgg)
    train_losses_rvgg = checkpoint["train_losses"]
    test_losses_rvgg = checkpoint["test_losses"]
    test_ssim_rvgg = checkpoint["test_ssim"]
    test_psnr_rvgg = checkpoint["test_psnr"]
    test_lpips_rvgg = checkpoint["test_lpips"]
else:   
    print("=> no checkpoint found at '{}'".format(chkpt_rvgg))
    sys.exit(1)

if os.path.isfile(chkpt_comp):  
    checkpoint = torch.load(chkpt_comp)
    train_losses_comp = checkpoint["train_losses"]
    test_losses_comp = checkpoint["test_losses"]
    test_ssim_comp = checkpoint["test_ssim"]
    test_psnr_comp = checkpoint["test_psnr"]
    test_lpips_comp = checkpoint["test_lpips"]
else:
    print("=> no checkpoint found at '{}'".format(chkpt_comp))
    sys.exit(1)

if os.path.isfile(chkpt_rcomp):
    checkpoint = torch.load(chkpt_rcomp)
    train_losses_rcomp = checkpoint["train_losses"]
    test_losses_rcomp = checkpoint["test_losses"]
    test_ssim_rcomp = checkpoint["test_ssim"]
    test_psnr_rcomp = checkpoint["test_psnr"]
    test_lpips_rcomp = checkpoint["test_lpips"]
else:
    print("=> no checkpoint found at '{}'".format(chkpt_rcomp))
    sys.exit(1)




epochs = list(range(1, 1500 + 1))

f_loss = plt.figure(figsize=(10, 6))
ax_loss = f_loss.add_subplot(111)
ax_loss.plot(epochs, test_ssim_mse, label="Test SSIM MSE")
ax_loss.plot(epochs, test_ssim_rmse, label="Test SSIM RMSE")
ax_loss.plot(epochs, test_ssim_vgg, label="Test SSIM VGG")
ax_loss.plot(epochs, test_ssim_rvgg, label="Test SSIM Random VGG")
ax_loss.plot(epochs, test_ssim_comp, label="Test SSIM Complexity")
ax_loss.plot(epochs, test_ssim_rcomp, label="Test SSIM Random Complexity")
ax_loss.plot(epochs, rtest_ssim, label="Test SSIM Random")

ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("SSIM")
# ax_loss.title("Training and Testing Losses")
ax_loss.legend()
ax_loss.grid(True)

os.makedirs(loss_plots_dir, exist_ok=True)
fname = os.path.join(loss_plots_dir, 'SSIM_plot.png')

# plt.show() 
f_loss.savefig(fname)