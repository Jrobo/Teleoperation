import os
import re
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Define the directory where your files are located
directory = "/home/jamil/PyRep/projects/results/lastvalues100/"

# Initialize lists to store data
file_names = []
average_values = []
sigma_values = []
train_values = []
val_values = []

# Define the regex patterns to extract values
sigma_pattern = r"sigma_([\d.]+)"
train_pattern = r"train_([\d.]+)"
val_pattern = r"val_([\d.]+)"

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".npy") and "epochs_30000" in filename:
        file_names.append(filename)
        filepath = os.path.join(directory, filename)
        # Load the data from the file
        data = np.load(filepath)
        # Calculate the average value of the data
        average = np.mean(data)
        average_values.append(average)
        
        # Extract sigma value
        sigma_match = re.search(sigma_pattern, filename)
        if sigma_match:
            sigma_value = float(sigma_match.group(1))
            sigma_values.append(sigma_value)
        else:
            sigma_values.append(None)
        
        # Extract train value
        train_match = re.search(train_pattern, filename)
        if train_match:
            train_value = float(train_match.group(1))
            train_values.append(train_value)
        else:
            train_values.append(None)
        
        # Extract val value
        val_match = re.search(val_pattern, filename)
        if val_match:
            val_value = float(val_match.group(1))
            val_values.append(val_value)
        else:
            val_values.append(None)

# Create a DataFrame to store the data
data_df = pd.DataFrame({
    'File Name': file_names,
    'Average Value': average_values,
    'Sigma Value': sigma_values,
    'Train Value': train_values,
    'Val Value': val_values
})


# Print the DataFrame in tabulate format
print(tabulate(data_df, headers='keys', tablefmt='grid', showindex=False))

# Filter data for files ending with "last100valLoss" and "last100trainLoss"
val_loss_data = data_df[data_df['File Name'].str.endswith('last100valLoss.npy')]
train_loss_data = data_df[data_df['File Name'].str.endswith('last100trainLoss.npy')]

# Plot the values
plt.plot(train_loss_data['Sigma Value'], train_loss_data['Average Value'], 'ro', label='Train Loss')
plt.plot(val_loss_data['Sigma Value'], val_loss_data['Average Value'], 'b+', label='Val Loss')

plt.xlabel('Sigma Value')
plt.xscale('log')
plt.ylabel('Loss Value')
plt.title('Loss Value vs Sigma Value')
plt.legend()
plt.grid(True)
# Print the sorted values in tabulate format
print("Sorted Val Loss Data:")
print(tabulate(val_loss_data.sort_values(by='Average Value'), headers='keys', tablefmt='grid', showindex=False))
print("\nSorted Train Loss Data:")
print(tabulate(train_loss_data.sort_values(by='Average Value'), headers='keys', tablefmt='grid', showindex=False))

plt.tight_layout()
plt.show()