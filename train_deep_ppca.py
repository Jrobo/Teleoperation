import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from modules.deep_ppca import DeepPPCA
from modules.data_load_module import load_data, create_data_loaders

#Training
def train_model(model, train_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, _ in train_loader:
            optimizer.zero_grad()
            nll = -model.log_likelihood(inputs)
            print("Log P",nll.shape)
            nll=nll.mean()
            nll.backward()
            optimizer.step()
            train_losses.append(nll.item())# Appending

        print(f'Epoch {epoch+1}, Log-Likelihood: {nll.item()}')

#Evaluation
def evaluate_model(model, dataloader):
    model.eval()  # evaluation mode
    total_nll = 0
    with torch.no_grad():  
        for inputs, _ in dataloader:
            nll = -model.log_likelihood(inputs)
            total_nll += nll.item()
    return total_nll / len(dataloader)


# Load data
train_dataset, val_dataset, test_dataset = load_data('/home/jamil/PyRep/projects/dataset/all_demos_joint_data.npy')

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

# Initialize 
#sigma = 1
h_theta_model = DeepPPCA()
train_losses=[]
test_losses=[]

# Train 
optimizer = optim.Adam(h_theta_model.parameters(), lr=0.0001)
num_epochs = 100
train_model(h_theta_model, train_loader, optimizer, num_epochs)

""" params=h_theta_model.parameters();
for param in params:
    print("Parameters are ",param.shape) """
# Evaluate --->test set
#test_nll = evaluate_model(h_theta_model, test_loader)
#test_losses.append(test_nll)
#print(f'Test Negative Log-Likelihood: {test_nll:.4f}')

#Save the model
torch.save(h_theta_model.state_dict(), './saved_models/ppca_model.pth')
print("Models saved.")

# Plot 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.legend()
plt.show()