import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
import torch.nn.functional as F

#defining according to Michel's preposed model


# Model to find the H matrix
class HThetaNetwork(nn.Module):
    def __init__(self):
        super(HThetaNetwork, self).__init__()
        self.fc1 = nn.Linear(7, 128)  # I/P # Relu, tanh, ....
        self.fc2 = nn.Linear(128, 128) # H1
        self.fc3 = nn.Linear(128, 14) # O/P

    def forward(self, x):
        z1 = F.relu(self.fc1(x))   # H1
        y1 = F.relu(self.fc2(z1))   # H2
        H = self.fc3(y1)           # O/P
        H = H.view(-1, 7, 2)      
        return H

# Model 
class FOmegaNetwork(nn.Module):
    def __init__(self):
        super(FOmegaNetwork, self).__init__()
        self.fc1 = nn.Linear(14, 128) # I/P
        self.fc2 = nn.Linear(128, 128) # H1/H2
        self.fc3 = nn.Linear(128, 2)  # O/P 
        
    def forward(self, x, v):
        print(x.shape)
        z = torch.cat((x, v), dim=1)
        print(z.shape)
        # x = torch.cat((x, v), dim=1)  # x= Joint state and v=true velocity(Modified on 26 january)
        z = F.relu(self.fc1(z))       # H1
        z = F.relu(self.fc2(z))       # H2
        a = self.fc3(z)               # O/P 
        a = torch.tanh(a)             # tanh 
        print("shape of a inside the fomega class",a.shape)
        return a
'''
#Model to find the H matrix
class HThetaNetwork(nn.Module):
    def __init__(self):
        super(HThetaNetwork, self).__init__()
        self.fc = nn.Linear(7, 14)  # flattened(7x2)
        
    def forward(self, x):
        H = self.fc(x)
        H = H.view(-1, 7, 2)  #Reshape(7x2)
        return H
#model to find a vector( i need to replace later in depoyment phase)
#need to think how to deploy after seeting the all jont conditions
class FOmegaNetwork(nn.Module):
    def __init__(self):
        super(FOmegaNetwork, self).__init__()
        self.fc = nn.Linear(14, 2)  # According to Michels model is state and velocity 
        
    def forward(self, x, v):
        x = torch.cat((x, v), dim=1)  # state and true velocity 
        a = self.fc(x)
        return a
'''
'build this evauation because it # No need to track gradients in validation as well as training'

def evaluate_model(h_theta_net, f_omega_net, dataloader, criterion):
    h_theta_net.eval()  # Set to evaluation mode
    f_omega_net.eval()  #also same 
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  
        for batch in dataloader:
            manipulator_state, true_velocities = batch
            # Predict the H matrix using HThetaNet
            #represents a a 2 Rd (a column vector). The matrix H 2 Rm⇥d is
            #qˆ˙ 2 Rm and the user’s low dimensional action is
            #linear subspace of high-dimensional manipulation commands 
            # controllable by low-dimensional actions.
            predicted_H = h_theta_net(manipulator_state)
            # Predict the user’s low dimensional action v a vector using FOmegaNet
            inferred_a = f_omega_net(manipulator_state, true_velocities)
            
            # predicted high dimensional robotic velocity command
            predicted_velocity = torch.matmul(predicted_H, inferred_a.unsqueeze(-1)).squeeze(-1)
            
            # Calculate the loss for this batch
            loss = criterion(predicted_velocity, true_velocities)
            total_loss += loss.item()  # .item() to get the value as a Python float
            num_batches += 1
            
    # the average loss 
    average_loss = total_loss / num_batches
    return average_loss

# available training tuples {(observation,q_dot)}#dataset loading
data = np.load('/home/jamil/PyRep/projects/all_demos_joint_data.npy', allow_pickle=True)

# Data Split
#joystick_input = data[:, :7]  # joystick inputs
manipulator_state = data[:, :7]  # joint states
true_velocity = data[:, 7:]  #  true velocities

#torch tensors
#joystick_input_tensor = torch.tensor(joystick_input, dtype=torch.float32)
manipulator_state_tensor = torch.tensor(manipulator_state, dtype=torch.float32)
joint_velocity_tensor = torch.tensor(true_velocity, dtype=torch.float32)

# Tensor Dataset
dataset = TensorDataset(manipulator_state_tensor, joint_velocity_tensor)

# training, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])



# Initialize the networks
h_theta_net = HThetaNetwork()
f_omega_net = FOmegaNetwork()

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(h_theta_net.parameters()) + list(f_omega_net.parameters()), lr=0.001)

train_losses = []
val_losses = []
test_losses = []

# DataLoaders 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    for batch in train_loader:
        manipulator_state, true_velocity = batch

        # Predict transformation matrix and infer joystick inputs
        predicted_H = h_theta_net(manipulator_state)
        print("Predicted_H shape",predicted_H.shape)
        inferred_a = f_omega_net(manipulator_state, true_velocity)
        print("inferred-a size", inferred_a.shape)
        # predicted velocity
        predicted_velocity = torch.bmm(predicted_H, inferred_a.unsqueeze(-1)).squeeze(-1)
        print("predicted Velocity",predicted_velocity.shape)
        # Calculate loss
        loss = criterion(predicted_velocity, true_velocity)

        # Backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    val_loss = evaluate_model(h_theta_net, f_omega_net, val_loader, criterion)

    # losses
    train_losses.append(loss.item())
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

# evaluate the model 
test_loss = evaluate_model(h_theta_net, f_omega_net, test_loader, criterion)
test_losses.append(test_loss)
print(f'Test Loss: {test_loss:.4f}')

print("Training complete")



# # save the models
# torch.save(h_theta_net.state_dict(), 'h_theta_net.pth')
# torch.save(f_omega_net.state_dict(), 'f_omega_net.pth')
# print("Models saved.")
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
# plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
# #plt.plot(range(1, num_epochs + 1), [test_losses[0]] * num_epochs, label='Test')
# plt.title('Training, Validation and Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

'''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.load('/home/jamil/PyRep/all_demos_joint_data.npy', allow_pickle=True)

# Split the dataa
inputs = data[:, :7]  # input features (joint positions)
outputs = data[:, 7:]  # outputs to predict (joint velocities)

# Convert numpy arrays to torch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

# Create a TensorDataset
dataset = TensorDataset(inputs_tensor, outputs_tensor)

# Split the dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network architecture
class ManipulatorNet(nn.Module):
    def __init__(self):
        super(ManipulatorNet, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the network
net = ManipulatorNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Function to compute the loss on a given dataset loader
def compute_loss(loader):
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

# Training and validation loop
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    net.train()
    for states, velocities in train_loader:
        optimizer.zero_grad()
        outputs = net(states)
        loss = criterion(outputs, velocities)
        loss.backward()
        optimizer.step()
    
    net.eval()
    train_loss = compute_loss(train_loader)
    val_loss = compute_loss(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

print("Training complete")

# Plot the training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.show()
'''