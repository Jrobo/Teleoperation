# import numpy as np

# # Load the first dataset
# npy_filename_1 = 'dataset/joint_data_with_identifiers.npz'
# data_1 = np.load(npy_filename_1)

# # Extract joint_positions from the first dataset
# joint_positions_1 = data_1['joint_positions']
# print("Shape of joint_positions from joint_data_with_identifiers.npz:", joint_positions_1.shape)

# # Flatten the joint_positions array from the first dataset to a single one-dimensional array
# joint_positions_flat_1 = joint_positions_1.flatten()
# print("Size of joint_positions_flat from joint_data_with_identifiers.npz:", joint_positions_flat_1.size)

# # Find the minimum and maximum values in the flattened array from the first dataset
# min_joint_position_1 = np.min(joint_positions_flat_1)
# max_joint_position_1 = np.max(joint_positions_flat_1)

# # Print the results for the first dataset
# print("Minimum joint position value from joint_data_with_identifiers.npz:", min_joint_position_1)
# print("Maximum joint position value from joint_data_with_identifiers.npz:", max_joint_position_1)

# # Load the second dataset
# npy_filename_2 = 'dataset/uniform_dataset.npz'
# data_2 = np.load(npy_filename_2)

# # Extract joint_positions from the second dataset
# joint_positions_2 = data_2['joint_positions']
# print("Shape of joint_positions from uniform_dataset.npz", joint_positions_2.shape)

# # Flatten the joint_positions array from the second dataset to a single one-dimensional array
# joint_positions_flat_2 = joint_positions_2.flatten()
# print("Size of joint_positions_flat from uniform_dataset.npz:", joint_positions_flat_2.size)

# # Find the minimum and maximum values in the flattened array from the second dataset
# min_joint_position_2 = np.min(joint_positions_flat_2)
# max_joint_position_2 = np.max(joint_positions_flat_2)

# # Print the results for the second dataset
# print("Minimum joint position value from uniform_dataset.npz:", min_joint_position_2)
# print("Maximum joint position value from uniform_dataset.npz:", max_joint_position_2)

import numpy as np

# Load the dataset
npy_filename = 'dataset/joint_data_with_identifiers.npz'
data = np.load(npy_filename)

# Extract joint_positions from the loaded data
joint_positions = data['joint_positions']
print("Shape of joint_positions:", joint_positions.shape)

# Flatten the joint_positions array to a single one-dimensional array
joint_positions_flat = joint_positions.flatten()

print("Size of joint_positions_flat:", joint_positions_flat.size)

# Find the minimum and maximum values in the flattened array
min_joint_position = np.min(joint_positions_flat)
max_joint_position = np.max(joint_positions_flat)

# Print the results
print("Minimum joint position value:", min_joint_position)
print("Maximum joint position value:", max_joint_position)
