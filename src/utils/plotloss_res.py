import torch
import matplotlib.pyplot as plt
import numpy as np


checkpoint = torch.load("../../outputs/Isabel_mse_active_random/base200ep/model/chk_199.pth.tar")

epoch = checkpoint["epoch"]
train_losses = checkpoint["train_losses"]
test_losses = checkpoint["test_losses"]
batch_train_losses = checkpoint["batch_train_losses"]
batch_test_losses = checkpoint["batch_test_losses"]

# avg_train_losses = [sum(train_losses[i:i+10]) / 10 for i in range(0, len(train_losses), 10)]


# epoch_losses = []
# current_epoch_losses = []
# train_len = 900
# for i, loss in enumerate(train_losses):
#     batches = 3
#     # train_len = min(train_len + 100, 10000)
#     train_len = min(1000 + i*100, 10000)
#     num_batches = round(np.ceil(train_len / 400))

#     current_epoch_losses.append(loss)
#     if len(current_epoch_losses) == num_batches:
#         epoch_avg_loss = sum(current_epoch_losses) / num_batches
#         epoch_losses.append(epoch_avg_loss)
#         current_epoch_losses = []

epochs = list(range(1, epoch + 2))

# Plot the training and testing losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.title("Training and Testing Losses")
plt.legend()
plt.grid(True)

# plt.show() 
plt.savefig("loss_plot.png")