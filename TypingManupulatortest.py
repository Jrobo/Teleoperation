import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define HThetaNetwork to find the H matrix
class HThetaNetwork(nn.Module):
    def __init__(self):
        super(HThetaNetwork, self).__init__()
        self.fc1 = nn.Linear(7, 128)  # Input
        self.fc2 = nn.Linear(128, 128)  # Hidden layer 1
        self.fc3 = nn.Linear(128, 14)  # Output

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Hidden layer 1
        x = F.relu(self.fc2(x))  # Hidden layer 2
        H = self.fc3(x)  # Output
        H = H.view(-1, 7, 2)
        return H

# Define FOmegaNetwork model
class FOmegaNetwork(nn.Module):
    def __init__(self):
        super(FOmegaNetwork, self).__init__()
        self.fc1 = nn.Linear(14, 128)  # Input
        self.fc2 = nn.Linear(128, 128)  # Hidden layer 1/2
        self.fc3 = nn.Linear(128, 2)  # Output

    def forward(self, x, v):
        z = torch.cat((x, v), dim=1)
        z = F.relu(self.fc1(z))  # Hidden layer 1
        z = F.relu(self.fc2(z))  # Hidden layer 2
        a = torch.tanh(self.fc3(z))  # Output with tanh activation
        return a

# Evaluate model function
def evaluate_model(h_theta_net, f_omega_net, dataloader, criterion):
    h_theta_net.eval()
    f_omega_net.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            manipulator_state, true_velocities = batch
            predicted_H = h_theta_net(manipulator_state)
            inferred_a = f_omega_net(manipulator_state, true_velocities)
            predicted_velocity = torch.matmul(predicted_H, inferred_a.unsqueeze(-1)).squeeze(-1)
            loss = criterion(predicted_velocity, true_velocities)
            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches
    return average_loss

# Load dataset
data = np.load('/home/jamil/PyRep/projects/all_demos_joint_data.npy', allow_pickle=True)

# Data split
#joystick_input = data[:, :7]  # Joystick inputs
manipulator_state = data[:, :7]  # Joint states
true_velocity = data[:, 7:]  # True velocities

# Torch tensors
#joystick_input_tensor = torch.tensor(joystick_input, dtype=torch.float32)
manipulator_state_tensor = torch.tensor(manipulator_state, dtype=torch.float32)
joint_velocity_tensor = torch.tensor(true_velocity, dtype=torch.float32)

# Tensor dataset
dataset = TensorDataset(manipulator_state_tensor, joint_velocity_tensor)

# Train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Initialize networks
h_theta_net = HThetaNetwork()
f_omega_net = FOmegaNetwork()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(h_theta_net.parameters()) + list(f_omega_net.parameters()), lr=0.001)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    for batch in train_loader:
        manipulator_state, true_velocity = batch
        predicted_H = h_theta_net(manipulator_state)
        inferred_a = f_omega_net(manipulator_state, true_velocity)
        predicted_velocity = torch.bmm(predicted_H, inferred_a.unsqueeze(-1)).squeeze(-1)
        loss = criterion(predicted_velocity, true_velocity)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    val_loss = evaluate_model(h_theta_net, f_omega_net, val_loader, criterion)

    # Record losses
    train_losses.append(loss.item())
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

# Evaluate the model on the test set
test_loss = evaluate_model(h_theta_net, f_omega_net, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')

print("Training complete")

# Save the models
torch.save(h_theta_net.state_dict(), 'h_theta_net.pth')
torch.save(f_omega_net.state_dict(), 'f_omega_net.pth')
print("Models saved.")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
