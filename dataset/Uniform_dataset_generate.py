import torch
import numpy as np
import matplotlib.pyplot as plt

class RobotTrajectory():
    def __init__(self, num_samples, num_joints, trajectory_id):
        self.num_samples = num_samples
        self.num_joints = num_joints
        self.trajectory_id = trajectory_id

        # Define the range for joint values
        self.joint_range = (-np.pi, np.pi)

        # Generate random samples for joint values and velocities
        self.joint_values = torch.FloatTensor(num_samples, num_joints).uniform_(*self.joint_range)
        self.velocities = torch.FloatTensor(num_samples, num_joints).uniform_(-1, 1) * np.pi  # Adjusted range for velocities

    def get_trajectory_data(self):
        return self.joint_values.numpy(), self.velocities.numpy(), self.trajectory_id

# Number of samples and joints
num_samples = 5
num_joints = 7
num_trajectories = 20

# trajectories
joint_positions_list = []
joint_velocities_list = []
identifiers_list = []

# Generate and store data for each trajectory
for trajectory_id in range(num_trajectories):
    trajectory = RobotTrajectory(num_samples, num_joints, trajectory_id)
    joint_positions, joint_velocities, identifier = trajectory.get_trajectory_data()
    joint_positions_list.append(joint_positions)
    joint_velocities_list.append(joint_velocities)
    identifiers_list.extend([identifier] * num_samples)

# Concatenate data for all trajectories
all_joint_positions = np.concatenate(joint_positions_list)
all_joint_velocities = np.concatenate(joint_velocities_list)
all_identifiers = np.array(identifiers_list)


# Create a dictionary to store the data
data_dict = {
    'Joint Positions': all_joint_positions,
    'Joint Velocities': all_joint_velocities,
    'Identifiers': all_identifiers
}


# Save the data to a .npz file
np.savez('dataset/uniform_dataset.npz', joint_positions=all_joint_positions, joint_velocities=all_joint_velocities, identifiers=all_identifiers)

