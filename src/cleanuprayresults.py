import os
import shutil
import glob

# Base directory where the train_model_dc344_* folders are located
base_dir = '/data1/vivekg/ray_results/train_model_vortex_2024-06-04_01-13-47'

# Pattern to match train_model_dc344_* directories
folder_pattern = os.path.join(base_dir, 'train_model_vortex_97de9_000*')

# Get the list of train_model_dc344_* directories
train_model_dirs = glob.glob(folder_pattern)

# Iterate through each matched directory
for train_model_dir in train_model_dirs:
    # Get the list of checkpoint_* directories within the current train_model directory
    checkpoint_dirs = [d for d in os.listdir(train_model_dir) if d.startswith('checkpoint_')]
    
    # Sort the checkpoint directories to identify the last one
    checkpoint_dirs.sort()

    # Delete all checkpoints except the last one
    for checkpoint in checkpoint_dirs[:-1]:
        checkpoint_path = os.path.join(train_model_dir, checkpoint)
        print(f'Deleting {checkpoint_path}')
        shutil.rmtree(checkpoint_path)
    
    if checkpoint_dirs:
        print(f'Kept the last checkpoint: {checkpoint_dirs[-1]} in {train_model_dir}')

print('Completed cleanup of all checkpoint directories.')
