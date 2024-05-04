import os
import glob
import pandas as pd

# Open the CSV file
df = pd.read_csv('./test/isabel_pr_viewparams_test.csv', header=None, names=['phi_val', 'theta_val'])

# Combine column values into filenames
filenames = [f"{phi:.4f}_{theta:.4f}.png" for phi, theta in zip(df['phi_val'], df['theta_val'])]

# Get the directory containing the CSV file
directory = os.path.dirname(os.path.abspath('./test/isabel_pr_viewparams_test.csv'))

# List all files in the directory
all_files = os.listdir(directory)

# Filter out files that match the expected filename format
expected_files = set(filenames)
unexpected_files = [f for f in all_files if os.path.splitext(f)[1].lower() == '.png' and f not in expected_files]

# Print the unexpected files
if unexpected_files:
    print("Unexpected files found:")
    for file in unexpected_files:
        print(file)
else:
    print("No unexpected files found.")


train_dir = './train'
test_dir = './test'

train_filenames = glob.glob(f"{train_dir}/*.png")

# Get the list of filenames in the test directory
test_filenames = glob.glob(f"{test_dir}/*.png")

# Check for common filenames
common_filenames = set(train_filenames) & set(test_filenames)

if common_filenames:
    print("Data leakage detected! The following filenames are present in both train and test directories:")
    for filename in common_filenames:
        print(filename)
else:
    print("No data leakage detected between train and test directories.")