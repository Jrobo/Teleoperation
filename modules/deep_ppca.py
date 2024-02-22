import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Model
class DeepPPCA(nn.Module):
    def __init__(self, sigma=0.01):
        super(DeepPPCA, self).__init__()
        self.sigma = sigma
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 49)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        H = self.fc3(x)
        H = H.view(-1, 7, 7)
        print("size of H",H.shape)
        return H
        
    # Log-likelihood function
    def log_likelihood(self, data):
        H = self(data)
        covariance_matrix = H @ H.transpose(1,2) + self.sigma**2 * torch.eye(data.size(1))
        mvn = torch.distributions.MultivariateNormal(torch.zeros(data.size()), covariance_matrix)
        return mvn.log_prob(data)


    def get_transformation(self, data):
            """
            Data is a 7-dimensional column vector
            """
            H = self(data)
            covariance_matrix = H @ H.transpose(1,2) + self.sigma**2 * torch.eye(data.size(1))
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix.squeeze())
            idx = eigenvalues.argsort().flip([0])
            eigenvalues = eigenvalues[idx]
            eigenvalues_sign = torch.sign(eigenvalues[idx])
            print(torch.sign(eigenvectors[0, idx]), eigenvectors[:, idx])
            eigenvectors = eigenvectors[:,idx] * torch.sign(eigenvectors[0, idx])
            return eigenvectors

    def get_mode_transformation(self, data, mode):
        """
        mode switching:
        Mode0:col 1 & 2 ; Mode1:col 3 & 4 ; Mode2:col 5 & 6 ; Mode3:col 6 & 7
        """
        eigenvectors = self.get_transformation(data)
        if mode < 3:
            return eigenvectors[:, mode*2:mode*2+2]
        else:
            return eigenvectors[:, -2:]

    def predict_velocities(self, robot_state, joystick, mode=0):
            robot_state_torch = torch.tensor(robot_state, dtype=torch.float32).unsqueeze(0)
            joystick_torch = torch.tensor(joystick, dtype=torch.float32)
            H_pred=self.get_mode_transformation(robot_state_torch, mode)
            print("H prediction before multiplying joystick values",H_pred)
            ret = (H_pred @ joystick_torch).squeeze()           
            return ret
        
