import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from modules.deep_ppca import DeepPPCA
from modules.data_load_module import load_data, create_data_loaders

# Evaluation
def evaluate_model(model, dataloader):
    model.eval()  # evaluation mode
    total_nll = 0
    total_samples = 0 
    with torch.no_grad():
        for inputs, _ in dataloader:
            log_prob, det_cov = model.log_likelihood(inputs)
            nll = -log_prob.mean() + 0.5 * torch.log(det_cov).mean()
            total_nll += nll.item() * inputs.size(0)  # Sum up 
            total_samples += inputs.size(0)  

    return total_nll / total_samples  # Average lg-likhood per sample


# Training
def train_model(model, train_loader, optimizer, num_epochs):
    train_losses = []  # Initialize 
    test_losses = []   # Initialize 
    for epoch in range(num_epochs):
        for inputs, _ in train_loader:
            optimizer.zero_grad()

            # Transfer Data--->lg-lkhood and det of the cov matrix
            log_prob, det_cov = model.log_likelihood(inputs)
            
            # negative lg-lkhood
            nll = -log_prob.mean() + 0.5 * torch.log(det_cov).mean()
            
            # Backpropagation and optimization
            nll.backward()
            optimizer.step()
        train_losses.append(nll.item())
        
        # Validation
        model.eval()
        val_loss = evaluate_model(model, val_loader)
        val_losses.append(val_loss)

        # showing training and validation lg-likhood 
        print(f'Epoch {epoch+1}, Train Log-Likelihood: {nll.item()}, Val Log-Likelihood: {val_loss}')
    
    #showing final training lg-lkhood all epochs finishes
    print(f'Epoch {epoch+1}, Log-Likelihood: {nll.item()}')
    
    return train_losses, test_losses  # Return the lists of losses

# Load data
train_dataset, val_dataset, test_dataset = load_data('/home/jamil/PyRep/projects/dataset/all_demos_joint_data.npy')

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

# Initialize
#sigma = 1
h_theta_model = DeepPPCA(sigma=0.0001)
train_losses = []
test_losses = []
val_losses=[]

# Train
optimizer = optim.Adam(h_theta_model.parameters(), lr=0.0001)
num_epochs = 1000
train_losses, test_losses =train_model(h_theta_model, train_loader, optimizer, num_epochs)


# Save 
torch.save(h_theta_model.state_dict(), './saved_models/ppca_model.pth')
print("Model saved.")
print("tran losses",len(train_losses))
print("val losses",len(val_losses))
print("length of train loader",len(train_loader))
print("length of train loader",len(val_loader))
print("length of test loader",len(test_loader))
# Plot
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.legend()
plt.show()
