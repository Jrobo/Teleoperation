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


# Set the random seed for reproducibility
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
            nll = -log_prob.mean() + 0.5 * torch.log(det_cov).mean()
            total_nll += nll.item() * inputs.size(0)  # Sum up 
            total_samples += inputs.size(0)  

    return total_nll / total_samples  #Avg lg-likhood per sample

# Training
def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize total_loss before the loop
        # Training
        model.train()  # training mode
        for inputs, _ in train_loader:
            optimizer.zero_grad()

            log_prob, det_cov = model.log_likelihood(inputs)
            nll = -log_prob.mean() + 0.5 * torch.log(det_cov).mean()

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
    train_dataset, val_dataset, test_dataset = load_data('/home/jamil/PyRep/projects/dataset/joint_data_with_identifiers.npz')

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
    
    end_time = time.time()  # Record the end time
    training_time = end_time - start_time

    # # Save the results to a CSV file
    # now = datetime.now()
    # current_time = now.strftime("%Y-%m-%d_%H-%M")
    # image_name = f'symlog_sig_{sigma}_eph_{num_epochs}_lr_{lr}_{current_time}.png'
    # image_path = os.path.join('./Images_Plots', image_name)
    # with open('results/training_results.csv', mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     if file.tell() == 0:  # Check if the file is empty
    #         writer.writerow(['Index', 'Date Time', 'Sigma', 'Learning Rate', 'Epochs', 'Training Time', 'Train Losses', 'Val Losses', 'Training Set Size', 'Image Path'])
    #     writer.writerow([file.tell() // 70, current_time, sigma, lr, num_epochs, training_time, train_losses, val_losses, train_set_size, image_path])

    # # Save 
    # torch.save(h_theta_model.state_dict(), f'./saved_models/ppca_model.pth')
    # print("Model saved.")
    # print("tran losses",len(train_losses))
    # print("val losses",len(val_losses))
    # print("length of train loader",len(train_loader)) 
    # print("length of train loader",len(val_loader))
    # print("length of test loader",len(test_loader))

    # Plot
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Log-Likelihood')
    plt.yscale('symlog')  # Set y-axis scale to symlog
    plt.legend()

    # # Include parameters in the plot title
    # plt.title(f"Sigma: {sigma}, Epochs: {num_epochs}")
    # plt.savefig(image_path)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DeepPPCA model')
    parser.add_argument('--sigma', type=float, default=0.000000000000000000000000000000000000000000000000000000000000000000000000001, help='Value of sigma.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()

    main(args.sigma, args.lr, args.num_epochs, args.seed)
