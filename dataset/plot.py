import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Load the dataset
data = np.load('uniform_dataset.npz')

# Extract individual arrays from the loaded data
joint_positions = data['joint_positions']
joint_velocities = data['joint_velocities']
identifiers = data['identifiers']

# Find the maximum value reached
max_value_reached = np.max(identifiers)

# Print shape and dtype of each array
print("Joint Positions:")
print("  Shape:", joint_positions.shape)
print("  Dtype:", joint_positions.dtype)
print("\nJoint Velocities:")
print("  Shape:", joint_velocities.shape)
print("  Dtype:", joint_velocities.dtype)
print("\nIdentifiers:")
print("  Shape:", identifiers.shape)
print("  Dtype:", identifiers.dtype)
print("Maximum value reached in identifiers:", max_value_reached)

# Plot all the data
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot joint positions
for i in range(joint_positions.shape[1]):
    axs[0].plot(joint_positions[:, i], label=f'Joint {i+1}')
axs[0].set_title('Joint Positions')
axs[0].set_xlabel('Sample Index')
axs[0].set_ylabel('Position')
axs[0].legend()

# Plot joint velocities
for i in range(joint_velocities.shape[1]):
    axs[1].plot(joint_velocities[:, i], label=f'Joint {i+1}')
axs[1].set_title('Joint Velocities')
axs[1].set_xlabel('Sample Index')
axs[1].set_ylabel('Velocity')
axs[1].legend()

# Plot identifiers
axs[2].plot(identifiers)
axs[2].set_title('Identifiers')
axs[2].set_xlabel('Sample Index')
axs[2].set_ylabel('Identifier')

plt.tight_layout()
plt.show()
