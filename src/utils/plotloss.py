import torch
import matplotlib.pyplot as plt
import numpy as np

model_path = input("Enter the path to the model: ")
checkpoint = torch.load(model_path)

epoch = checkpoint["epoch"]
train_losses = checkpoint["train_losses"]
test_losses = checkpoint["test_losses"]
batch_train_losses = checkpoint["batch_train_losses"]
batch_test_losses = checkpoint["batch_test_losses"]
test_ssim = checkpoint["test_ssim"]
test_psnr = checkpoint["test_psnr"]

epochs = list(range(1, epoch +1))

# Plot the training and testing losses
f_loss = plt.figure(figsize=(10, 6))
ax_loss = f_loss.add_subplot(111)
ax_loss.plot(epochs, train_losses, label="Train Loss")
ax_loss.plot(epochs, test_losses, label="Test Loss")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("Loss")
# ax_loss.title("Training and Testing Losses")
ax_loss.legend()
ax_loss.grid(True)

# loss_plots_dir = os.path.join(args.root_out_dir, "loss_plots")
# os.makedirs(loss_plots_dir, exist_ok=True)
# fname = os.path.join(loss_plots_dir, 'loss_plot_epoch_'+ str(epoch) + ".png")

# plt.show() 
f_loss.savefig('loss_plot_epoch_'+ str(epoch) + ".png")
# main_logger.info('Loss plot saved at epoch %s', epoch)

# Save a copy of Loss plot with lowest training and test loss and epoch annotated
Lowest_trainloss_index = train_losses.index(min(train_losses))
Lowest_testloss_index = test_losses.index(min(test_losses))
Lowest_trainloss_epoch = epochs[Lowest_trainloss_index]
Lowest_testloss_epoch = epochs[Lowest_testloss_index]
Lowest_trainloss_value = train_losses[Lowest_trainloss_index]
Lowest_testloss_value = test_losses[Lowest_testloss_index]

# Adjust the xytext position based on the Lowest_trainloss_epoch and Lowest_trainloss_value
Lowest_trainloss_xytext_x = Lowest_trainloss_epoch - 200 if Lowest_trainloss_epoch > 500 else Lowest_trainloss_epoch
Lowest_trainloss_xytext_y = Lowest_trainloss_value + 0.5 if Lowest_trainloss_value < 1 else Lowest_trainloss_value - 0.05

# Adjust the xytext position based on the Lowest_trainloss_epoch and Lowest_trainloss_value
Lowest_testloss_xytext_x = Lowest_testloss_epoch - 200 if Lowest_testloss_epoch > 500 else Lowest_testloss_epoch + 50
Lowest_testloss_xytext_y = Lowest_testloss_value + 1.2 if Lowest_testloss_value < 1 else Lowest_testloss_value - 0.05

ax_loss.annotate(f"Min Train loss: {Lowest_trainloss_value:.4f}\nEpoch: {Lowest_trainloss_epoch}", 
                xy=(Lowest_trainloss_epoch, Lowest_trainloss_value), 
                xytext=(Lowest_trainloss_xytext_x, Lowest_trainloss_xytext_y),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                bbox=dict(boxstyle="round", fc="w"))
ax_loss.annotate(f"Min Test loss: {Lowest_testloss_value:.4f}\nEpoch: {Lowest_testloss_epoch}", 
                xy=(Lowest_testloss_epoch, Lowest_testloss_value), 
                xytext=(Lowest_testloss_xytext_x, Lowest_testloss_xytext_y),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                bbox=dict(boxstyle="round", fc="w"))
# fnamea = os.path.join(loss_plots_dir, 'loss_plot_ann_epoch_'+ str(epoch) + ".png")
# f_loss.show()
f_loss.savefig('loss_plot_ann_epoch_'+ str(epoch) + ".png")
# main_logger.info('Annotated Loss  plot saved at epoch %s', epoch)


f_ssim = plt.figure(figsize=(10, 6))
ax_ssim = f_ssim.add_subplot(111)

ax_ssim.plot(epochs, test_ssim, label="Test set SIM")
ax_ssim.set_xlabel("Epochs")
ax_ssim.set_ylabel("SSIM")
ax_ssim.legend()
ax_ssim.grid(True)
f_ssim.savefig('ssim_plot_epoch_'+ str(epoch) + ".png")

# Save a copy of SSIM plot with highest SSIM value and epoch annotated
highest_ssim_index = test_ssim.index(max(test_ssim))
highest_ssim_epoch = epochs[highest_ssim_index]
highest_ssim_value = test_ssim[highest_ssim_index]

# Adjust the xytext position based on the highest_ssim_epoch and highest_ssim_value
xytext_x = highest_ssim_epoch - 120 if highest_ssim_epoch > 800 else highest_ssim_epoch + 50
xytext_y = highest_ssim_value - 0.10 if highest_ssim_value > 0.6 else highest_ssim_value + 0.05

ax_ssim.annotate(f"Highest SSIM: {highest_ssim_value:.4f}\nEpoch: {highest_ssim_epoch}", 
                xy=(highest_ssim_epoch, highest_ssim_value), 
                xytext=(xytext_x, xytext_y),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                bbox=dict(boxstyle="round", fc="w"))
# ssima_fname = os.path.join(loss_plots_dir, 'ssim_plot_ann_epoch_'+ str(epoch) + ".png")
# f_ssim.show()
f_ssim.savefig('ssim_plot_ann_epoch_'+ str(epoch) + ".png")



f_psnr = plt.figure(figsize=(10, 6))
ax_psnr = f_psnr.add_subplot(111)
ax_psnr.plot(epochs, test_psnr, label="Test set PSNR")
ax_psnr.set_xlabel("Epochs")
ax_psnr.set_ylabel("PSNR")
ax_psnr.legend()
ax_psnr.grid(True)

# psnr_fname = os.path.join(loss_plots_dir, 'psnr_plot_epoch_'+ str(epoch) + ".png")
f_psnr.savefig('psnr_plot_epoch_'+ str(epoch) + '.png')

# Save a copy of PSNR plot with highest PSNR value and epoch annotated
highest_psnr_index = test_psnr.index(max(test_psnr))
highest_psnr_epoch = epochs[highest_psnr_index]
highest_psnr_value = test_psnr[highest_psnr_index]

# Adjust the xytext position based on the highest_ssim_epoch and highest_ssim_value
xytext_x = highest_psnr_epoch - 120 if highest_psnr_epoch > 800 else highest_psnr_epoch + 50
xytext_y = highest_psnr_value - 5.5 if highest_psnr_value > 28 else highest_psnr_value + 1

ax_psnr.annotate(f"Max PSNR: {highest_psnr_value:.4f}\nEpoch: {highest_psnr_epoch}", 
                xy=(highest_psnr_epoch, highest_psnr_value), 
                xytext=(xytext_x, xytext_y),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                bbox=dict(boxstyle="round", fc="w"))
# psnra_fname = os.path.join(loss_plots_dir, 'psnr_plot_ann_epoch_'+ str(epoch) + ".png")
# f_psnr.show()
f_psnr.savefig('psnr_plot_ann_epoch_'+ str(epoch) + ".png")
