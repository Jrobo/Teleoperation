import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(path, train_ratio=0.8, val_ratio=0.2, shuffle_demos=False):
    """Load data from the specified path and prepare it for training, validation, and testing."""
    data = np.load(path, allow_pickle=True)
    # Extract manipulator state, joint velocity, and identifiers from the loaded data
    manipulator_state = data['joint_positions']
    joint_velocity = data['joint_velocities']
    identifiers = data['identifiers']

    # Convert to PyTorch tensors
    manipulator_state_tensor = torch.tensor(manipulator_state, dtype=torch.float32)
    joint_velocity_tensor = torch.tensor(joint_velocity, dtype=torch.float32)

    # Combine manipulator state and joint velocity into a dataset
    dataset = TensorDataset(manipulator_state_tensor, joint_velocity_tensor)

    demo_indices = list(set(identifiers))
    # Shuffle demonstrations if True
    demo_indices = np.unique(identifiers)
    if shuffle_demos:
        np.random.shuffle(demo_indices)

    # Calculate the number of demonstrations for training and validation
    num_demos = len(demo_indices)
    num_train_demos = int(train_ratio * num_demos)
    num_val_demos = int(val_ratio * num_demos)

    # Divide the demonstration indices into training, validation, and testing sets
    train_indices = demo_indices[:num_train_demos]
    val_indices = demo_indices[num_train_demos:num_train_demos + num_val_demos]
    test_indices = demo_indices[num_train_demos + num_val_demos:]

# # Iterate over each index and corresponding demonstration identifier in the 'identifiers' list.
# # Create a list of indices (train_indices_list) where the corresponding demonstration identifier 
# # matches those intended for training (present in the 'train_indices' list).
#     train_indices_list = [
#         i # Index of the current demonstration identifier
#         for i, demo_id in enumerate(identifiers)  # Iterate over indices and identifiers
#         if demo_id in train_indices   # Filter: Check if the demonstration identifier is in 'train_indices'
# ]

    train_dataset = torch.utils.data.Subset(dataset, [i for i, demo_id in enumerate(identifiers) if demo_id in train_indices])
    val_dataset = torch.utils.data.Subset(dataset, [i for i, demo_id in enumerate(identifiers) if demo_id in val_indices])
    test_dataset = torch.utils.data.Subset(dataset, [i for i, demo_id in enumerate(identifiers) if demo_id in test_indices])
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=4):
    """"Create data loaders for the provided datasets with batch size=----"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


# def load_data(path, train_ratio=0.7, val_ratio=0.15):
#     data = np.load(path, allow_pickle=True)
#     manipulator_state = data[:, :7]
#     true_velocity = data[:, 7:]
#     manipulator_state_tensor = torch.tensor(manipulator_state, dtype=torch.float32)
#     joint_velocity_tensor = torch.tensor(true_velocity, dtype=torch.float32)
#     dataset = TensorDataset(manipulator_state_tensor, joint_velocity_tensor)
#     total_size = len(dataset)
#     #print("Sizeof_Dataset_", total_size)
#     train_size = int(train_ratio * total_size)
#     val_size = int(val_ratio * total_size)
#     test_size = total_size - train_size - val_size
#     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
#     return train_dataset, val_dataset, test_dataset
