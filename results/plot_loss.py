


import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where your files are located
directory = "/home/jamil/PyRep/projects/results/all_last100_loss_data/"

# Initialize dictionaries to store train and validation losses
train_losses = {}
val_losses = {}

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        filepath = os.path.join(directory, filename)
        # Load the data from the file
        data = np.load(filepath)
        # Extract the file name without extension
        file_key = os.path.splitext(filename)[0]
        # Check if the file name ends with "trainLoss" or "valLoss"
        if file_key.endswith("trainLoss"):
            # Remove "trainLoss" suffix and store in train_losses dictionary
            train_losses[file_key[:-9]] = np.mean(data)  # Calculate the average value
        elif file_key.endswith("valLoss"):
            # Remove "valLoss" suffix and store in val_losses dictionary
            val_losses[file_key[:-7]] = np.mean(data)  # Calculate the average value

# Plot train losses
plt.scatter(train_losses.keys(), train_losses.values(), label='Train Loss', color='blue')
# Plot validation losses
plt.scatter(val_losses.keys(), val_losses.values(), label='Validation Loss', color='red')

plt.xlabel('Sigma')
plt.ylabel('Loss')
plt.title('Train and Validation Losses')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
