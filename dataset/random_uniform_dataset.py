import numpy as np
import matplotlib.pyplot as plt

# Load the NPZ file
data = np.load('joint_data_with_identifiers.npz')

# Extract arrays
identifiers = data['identifiers']

# Find unique identifiers
unique_identifiers = np.unique(identifiers)

# Dictionary to store sizes
sizes = {}

# Iterate over unique identifiers
for identifier in unique_identifiers:
    # Find indices where identifier appears
    indices = np.where(identifiers == identifier)[0]
    # Get size of data for this identifier
    size = len(indices)
    print("Length of indices", size)
    # Store size in the dictionary
    sizes[identifier] = size

# Size of the random uniform disibution
num_samples_per_identifier = 100
num_identifiers = len(unique_identifiers)
print("num identifiers", num_identifiers)

# Initialize arrays to store random data
joint_velocities = np.empty((0, num_samples_per_identifier))
joint_positions = np.empty((0, num_samples_per_identifier))
new_identifiers = np.empty((0, num_samples_per_identifier))

# Generate random uniform distributions for each identifier
for identifier, size in sizes.items():
    # Generate random uniform distributions for joint velocities and positions
    random_joint_velocities = np.random.uniform(size=(1, num_samples_per_identifier))
    random_joint_positions = np.random.uniform(size=(1, num_samples_per_identifier))

    # Replicate identifiers for the generated samples
    identifier_array = np.full((1, num_samples_per_identifier), identifier)

    # Append generated data to the arrays
    joint_velocities = np.vstack((joint_velocities, random_joint_velocities))
    joint_positions = np.vstack((joint_positions, random_joint_positions))
    identifiers = np.vstack((new_identifiers, identifier_array))
    print(joint_velocities.shape)

# Save the generated dataset
np.savez('dataset/generated_dataset.npz', joint_velocities=joint_velocities, joint_positions=joint_positions, identifiers=identifiers)
print("Dataset generated and saved")


# Plotting
plt.figure(figsize=(12, 8))

# Plot joint velocities
plt.subplot(2, 1, 1)
for i in range(len(identifiers)):
    plt.scatter(np.arange(len(joint_velocities[i])), joint_velocities[i], label=f'Identifier {identifiers[i]}', alpha=0.5)
plt.title('Joint Velocities')
plt.xlabel('Sample Index')
plt.ylabel('Velocity')
plt.legend()

# Plot joint positions
plt.subplot(2, 1, 2)
for i in range(len(identifiers)):
    plt.scatter(np.arange(len(joint_positions[i])), joint_positions[i], label=f'Identifier {identifiers[i]}', alpha=0.5)
plt.title('Joint Positions')
plt.xlabel('Sample Index')
plt.ylabel('Position')
plt.legend()

plt.tight_layout()
plt.show()
