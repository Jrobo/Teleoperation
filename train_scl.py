
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
import torch.nn.functional as F
from modules.data_load_module import load_data, create_data_loaders
from modules.scl import HThetaNetwork,FOmegaNetwork,train_model, evaluate_model

# Load data
train_dataset, val_dataset, test_dataset = load_data('/home/jamil/PyRep/projects/dataset/all_demos_joint_data.npy')

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

# Initialize the networks
h_theta_net = HThetaNetwork()
f_omega_net = FOmegaNetwork()

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(h_theta_net.parameters()) + list(f_omega_net.parameters()), lr=0.001)

# Training loop
num_epochs = 100
train_losses = train_model(h_theta_net, f_omega_net, train_loader, criterion, optimizer, num_epochs)

# Validation
# val_loss = evaluate_model(h_theta_net, f_omega_net, val_loader, criterion)
# print(f'Validation Loss: {val_loss:.4f}')
val_losses = []
for epoch in range(num_epochs):
    val_loss = evaluate_model(h_theta_net, f_omega_net, val_loader, criterion)
    val_losses.append(val_loss)

# Evaluate the model 
test_loss = evaluate_model(h_theta_net, f_omega_net, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')

print("Training complete")

# Save the models
torch.save(h_theta_net.state_dict(), './saved_models/h_theta_net.pth')
torch.save(f_omega_net.state_dict(), './saved_models/f_omega_net.pth')

# save the models
torch.save(h_theta_net.state_dict(), './saved_models/h_theta_net.pth')
torch.save(f_omega_net.state_dict(), './saved_models/f_omega_net.pth')
print("Models saved.")



# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
#plt.plot(range(1, num_epochs + 1), [test_losses[0]] * num_epochs, label='Test')
plt.title('Training, Validation and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
