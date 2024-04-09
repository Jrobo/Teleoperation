import os
import re
import numpy as np
import pandas as pd
from tabulate import tabulate

# Define the directory where your files are located
directory = "/home/jamil/PyRep/projects/results/all_last100_loss_data/"

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
    if filename.endswith(".npy"):
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
            if train_value == 30000:  # Filter for train value of 30000
                train_values.append(train_value)
            else:
                train_values.append(None)
        else:
            train_values.append(None)
        
        # Extract val value
        val_match = re.search(val_pattern, filename)
        if val_match:
            val_value = float(val_match.group(1))
            if val_value == 30000:  # Filter for val value of 30000
                val_values.append(val_value)
            else:
                val_values.append(None)
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

# Drop rows with NaN values
data_df = data_df.dropna()

# Print the DataFrame in tabulate format
print(tabulate(data_df, headers='keys', tablefmt='grid'))
