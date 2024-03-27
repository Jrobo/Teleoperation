import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from modules.deep_ppca import DeepPPCA
from modules.data_load_module import load_data, create_data_loaders
import time
import csv
from datetime import datetime
import os
import torch.optim as optim

# Set the random seed 
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# Evaluation
def evaluate_model(model, dataloader):
    model.eval()  # evaluation mode
    total_nll = 0
    total_samples = 0 
    with torch.no_grad():
        for inputs, _ in dataloader:
            log_prob, det_cov = model.log_likelihood(inputs)
            nll = -log_prob.mean() #+ 0.5 * torch.log(det_cov).mean()
            total_nll += nll.item() * inputs.size(0)  # Sum up 
            total_samples += inputs.size(0)  

    return total_nll / total_samples  #Avg lg-likhood per sample

# Training
def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize total_loss 
        # Training
        model.train()  # training mode
        for inputs, _ in train_loader:
            optimizer.zero_grad()

            log_prob, det_cov = model.log_likelihood(inputs)
            nll = -log_prob.mean() # + 0.5 * torch.log(det_cov).mean()

            nll.backward()
            optimizer.step()

            total_loss += nll.item()  # loss for each batch

        average_loss = total_loss / len(train_loader)  # average loss for the epoch
        train_losses.append(average_loss)

        # Validation
        model.eval()  # evaluation mode
        val_loss = evaluate_model(model, val_loader)
        val_losses.append(val_loss)

        # Showing training and validation lg-likelihood
        print(f'Epoch {epoch+1}, Train Log-Likelihood: {average_loss}, Validation Log-Likelihood: {val_loss}')

    # Showing final training lg-likelihood after all epochs finish
    print(f'Finished: Total Epoch {epoch+1}, Final Log-Likelihood: {average_loss}')

    return train_losses, val_losses  # Return the lists of losses

def main(sigma, lr, num_epochs, seed):
    # Set random seed
    set_random_seed(seed)

    # Load data
    train_dataset, val_dataset, test_dataset = load_data('/home/jamil/PyRep/projects/joint_data_with_identifiers.npz')

    # data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Calculate training set size
    train_set_size = len(train_dataset)

    # Initialize
    h_theta_model = DeepPPCA(sigma)
    train_losses = []
    test_losses = []
    val_losses=[]

    # Train
    optimizer = optim.Adam(h_theta_model.parameters(), lr)
    start_time = time.time()  # Record the start time
    train_losses, val_losses = train_model(h_theta_model, train_loader, val_loader, optimizer, num_epochs)
    end_time = time.time()  # Record end time
    training_time = end_time - start_time

    # Save the results to CSV file
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M")
    image_name = f'symlog_sig_{sigma}_eph_{num_epochs}_lr_{lr}_{current_time}.png'
    image_path = os.path.join('./Images_Plots', image_name)

    # file extension with ".npy"
    npy_path = os.path.splitext(image_path)[0] + '.npy'

    # data
    loss_val_data = np.array([train_losses, val_losses])

    # Save the data as an npy file
    np.save(f'results/loss_sigma_{sigma}_epochs_{num_epochs}_lr_{lr}.npy', np.array([train_losses, val_losses]))
    # Define the filename
    csv_file = 'results/tttraining_results.csv'

    # Determine the row index
    row_index = 1  # Default to 1 if file is empty
    if os.path.isfile(csv_file):
        with open(csv_file, mode='r') as file:
            # Check if the file is empty
            if os.stat(csv_file).st_size > 0:
                row_index = sum(1 for line in file)

    # Write data to CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if row_index == 1:  # Check if the file is empty
            writer.writerow(['Index', 'Date Time', 'Sigma', 'Learning Rate', 'Epochs', 'Seed', 'Training Time', 'Training Set Size', 'Image Path','Numpy Path'])
        writer.writerow([row_index, current_time, sigma, lr, num_epochs, seed, round(training_time, 5), train_set_size, image_path,npy_path])

    # Save 
    torch.save(h_theta_model.state_dict(), f'./saved_models/ppca_model.pth')
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
    plt.yscale('symlog')  # Set y-axis scale to symlog
    plt.legend()

    # Include parameters in the plot title
    plt.title(f"Sigma: {sigma}, Epochs: {num_epochs}")
    plt.savefig(image_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DeepPPCA model')
    parser.add_argument('--sigma', type=float, default=0.001, help='Value of sigma.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility.')

    args = parser.parse_args()

    main(args.sigma, args.lr, args.num_epochs, args.seed)
