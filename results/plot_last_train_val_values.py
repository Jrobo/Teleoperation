import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import os

# Define the directory where your files are located
directory = "/home/jamil/PyRep/projects/results/"

# Define the pattern to match files ending with "last100trainLoss.npy"
pattern = directory + "*last100valLoss.npy"# may change with train

# Use glob to find all files matching the pattern
files = glob.glob(pattern)

# Create a new directory to save the files
destination_directory = "/home/jamil/PyRep/projects/results/all_last100_loss_data/"
os.makedirs(destination_directory, exist_ok=True)

# Print the number of files found
print("Number of files:", len(files))

# Print the names of the files
for file in files:
    print(file)


# Copy the files to the new directory
for file in files:
    shutil.copy(file, destination_directory)

print(f"{len(files)} files copied to {destination_directory}")
