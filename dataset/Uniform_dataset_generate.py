import torch
import numpy as np
import matplotlib.pyplot as plt

class RobotTrajectory():
    def __init__(self, num_samples, num_joints, trajectory_id, min_joint_values, max_joint_values):
        self.num_samples = num_samples
        self.num_joints = num_joints
        self.trajectory_id = trajectory_id

        # Range for joint values for each joint
        self.min_joint_values = min_joint_values
        self.max_joint_values = max_joint_values

        # Random samples for joint values and velocities
        self.joint_values = torch.FloatTensor(num_samples, num_joints)
        for i in range(num_joints):
            self.joint_values[:, i].uniform_(self.min_joint_values[i], self.max_joint_values[i])
        self.velocities = torch.FloatTensor(num_samples, num_joints).uniform_(-1, 1) * np.pi  # velocities

    def get_trajectory_data(self):
        return self.joint_values.numpy(), self.velocities.numpy(), self.trajectory_id

# Number of samples and joints
num_samples = 109
num_joints = 7
num_trajectories = 20

# Min and Max joint position values for each joint
max_joint_values = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
min_joint_values = [-2.8973, -1.7628, -2.973, -3.0718, -2.8973, -0.0175, -2.8973]


# Trajectories
joint_positions_list = []
joint_velocities_list = []
identifiers_list = []

# Generate and store data for each trajectory
for trajectory_id in range(num_trajectories):
    trajectory = RobotTrajectory(num_samples, num_joints, trajectory_id, min_joint_values, max_joint_values)
    joint_positions, joint_velocities, identifier = trajectory.get_trajectory_data()
    joint_positions_list.append(joint_positions)
    joint_velocities_list.append(joint_velocities)
    identifiers_list.extend([identifier] * num_samples)

# Concatenate data for all trajectories
all_joint_positions = np.concatenate(joint_positions_list)
all_joint_velocities = np.concatenate(joint_velocities_list)
all_identifiers = np.array(identifiers_list)

print("Shape of all_joint_positions:", all_joint_positions.shape)
print("Shape of all_joint_velocities:", all_joint_velocities.shape)
print("Size of identifiers_list:", len(identifiers_list))

# Create a dictionary to store the data
data_dict = {
    'Joint Positions': all_joint_positions,
    'Joint Velocities': all_joint_velocities,
    'Identifiers': all_identifiers
}

# Save the data to a .npz file
np.savez('dataset/uniform_dataset.npz', joint_positions=all_joint_positions, joint_velocities=all_joint_velocities, identifiers=all_identifiers)



# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# class RobotTrajectory():
#     def __init__(self, num_samples, num_joints, trajectory_id):
#         self.num_samples = num_samples
#         self.num_joints = num_joints
#         self.trajectory_id = trajectory_id

#         # Range for joint values
#         self.joint_range = (-np.pi, np.pi)

#         # Random samples for joint values and velocities
#         self.joint_values = torch.FloatTensor(num_samples, num_joints).uniform_(*self.joint_range)
#         self.velocities = torch.FloatTensor(num_samples, num_joints).uniform_(-1, 1) * np.pi  #velocities

#     def get_trajectory_data(self):
#         return self.joint_values.numpy(), self.velocities.numpy(), self.trajectory_id

# # Number of samples and joints
# num_samples = 5
# num_joints = 7
# num_trajectories = 20

# # Trajectories
# joint_positions_list = []
# joint_velocities_list = []
# identifiers_list = []

# # Generate and store data for each trajectory
# for trajectory_id in range(num_trajectories):
#     trajectory = RobotTrajectory(num_samples, num_joints, trajectory_id)
#     print('Trajectory',trajectory)
#     print(trajectory_id)
#     joint_positions, joint_velocities, identifier = trajectory.get_trajectory_data()
#     print(joint_positions)
#     joint_positions_list.append(joint_positions)
#     print(joint_positions_list)
#     joint_velocities_list.append(joint_velocities)
#     identifiers_list.extend([identifier] * num_samples)

# # Concatenate data for all trajectories
# all_joint_positions = np.concatenate(joint_positions_list)
# all_joint_velocities = np.concatenate(joint_velocities_list)
# all_identifiers = np.array(identifiers_list)


# # Create a dictionary to store the data
# data_dict = {
#     'Joint Positions': all_joint_positions,
#     'Joint Velocities': all_joint_velocities,
#     'Identifiers': all_identifiers
# }


# # Save the data to a .npz file
# np.savez('dataset/uniform_dataset.npz', joint_positions=all_joint_positions, joint_velocities=all_joint_velocities, identifiers=all_identifiers)

