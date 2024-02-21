import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_data(path, train_ratio=0.7, val_ratio=0.15):
    data = np.load(path, allow_pickle=True)
    manipulator_state = data[:, :7]
    true_velocity = data[:, 7:]
    manipulator_state_tensor = torch.tensor(manipulator_state, dtype=torch.float32)
    joint_velocity_tensor = torch.tensor(true_velocity, dtype=torch.float32)
    dataset = TensorDataset(manipulator_state_tensor, joint_velocity_tensor)
    total_size = len(dataset)
    print("Sizeof_Dataset_", total_size)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=20):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader
