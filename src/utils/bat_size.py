import torch
import matplotlib.pyplot as plt
import numpy as np
import math


checkpoint = torch.load("./mse/model/chk_599.pth.tar")

epoch = checkpoint["epoch"]

train_losses = checkpoint["train_losses"]
test_losses = checkpoint["test_losses"]

epoch_batch_sizes = [3]*300

dataset_length = 1000
max_dataset_length = 10000
batch_size = 400
log_every = int(input("Enter the value for log_every: "))


for i in range(epoch):
    dataset_length = min(dataset_length + i*100, max_dataset_length)
    num_batches = math.ceil(dataset_length / batch_size)
    epoch_batch_sizes.append(num_batches)

# Compute average train loss for each epoch
# epoch_train_losses = []
# start_idx = 100
# for batch_size in epoch_batch_sizes:
#     end_idx = start_idx + batch_size
#     epoch_train_losses.append(sum(train_losses[start_idx:end_idx]) / batch_size)
#     start_idx = end_idx