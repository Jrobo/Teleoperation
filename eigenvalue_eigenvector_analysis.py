import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.deep_ppca import DeepPPCA
from modules.data_load_module import load_data, create_data_loaders

def generate_uniform_distribution(size):
    return torch.rand(size) * 2 * np.pi

def calculate_eigen_values(H,sigma):
    eig_H_transpose=torch.transpose(H, 1, 2)
    covariance_matrix=torch.matmul(H, eig_H_transpose) + sigma**2
    eigenvalues_cov_matrix, _ = torch.linalg.eig(covariance_matrix)
    #print("Size of covariance matrix:", covariance_matrix.size())
    min_eigenvalues = torch.min(torch.abs(eigenvalues_cov_matrix[:, :-1] - eigenvalues_cov_matrix[:, 1:]), axis=1)
    return min_eigenvalues

def calculate_eigen_vectors(H, H_prime,sigma):
    H_transpose=torch.transpose(H, 1, 2)
    H_prime_transpose=torch.transpose(H_prime, 1, 2)
    #print("Size of transposed matrix:", H_transpose.size())  # Output: torch.Size([1, 7, 7])
    # Calculate covariance matrix
    covariance_matrix = torch.matmul(H, H_transpose) + sigma**2
    covariance_matrix_prime = torch.matmul(H_prime, H_prime_transpose) + sigma**2
    #print("Size of covariance matrix:", covariance_matrix.size()) # Output: torch.Size([1, 7, 7])
    abs_Cov = torch.abs(covariance_matrix-covariance_matrix_prime)
    max_Cov = torch.max(torch.sum(abs_Cov, dim=2))
    #diff_max_eigenvector = max_eigenvectors_H_prime - max_eigenvectors_H
    return max_Cov

def process_dataset(dataset, h_theta_model,sigma):
    diff_max_eigenvectors_list = []
    min_eigenvalues_list = []

    # Create an iterator over the DataLoader
    data_iterator = iter(dataset)

    # Iterate over pairs of consecutive elements
    try:
        while True:
            q, _ = next(data_iterator)  # Current q value
            q_prime, _ = next(data_iterator)  # Next q value
            print(f" shape (q): {q.size()}")
            print(f" shape (q_prime): {q_prime.size()}")
            # h_theta_model takes a tensor as input
            H = h_theta_model(q)
            print("H",H.size())
            H_prime = h_theta_model(q_prime)
            print("H_prime",H_prime.size())
            # Calculate eigen vectors and values
            diff_max_eigenvector = calculate_eigen_vectors(H, H_prime,sigma)
            min_eigenvalues = calculate_eigen_values(H,sigma)
            # Append results to lists
            diff_max_eigenvectors_list.append(diff_max_eigenvector.item())
            min_eigenvalues_list.append(min_eigenvalues[0])
    except StopIteration:
        pass

    return diff_max_eigenvectors_list, min_eigenvalues_list

def plot_histogram(tensor_values, title):
    plt.hist(tensor_values.numpy(), bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')

def create_scatter_plot(diff_max_eigenvectors_list, min_eigenvalues_list, title):
    diff_max_eigenvectors_tensor = torch.tensor(diff_max_eigenvectors_list).clone().detach()
    min_eigenvalues_tensor = torch.tensor(min_eigenvalues_list).clone().detach()
    plt.scatter(diff_max_eigenvectors_tensor, min_eigenvalues_tensor, color='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel('Difference of Max Eigenvectors')
    plt.ylabel('Min Eigenvalues')
    plt.grid(True) 
    plt.xscale('log')
    plt.yscale('log')

# Set random seed for reproducibility
torch.manual_seed(42)

# Load data
train_dataset, val_dataset, test_dataset = load_data('dataset/joint_data_with_identifiers.npz')

# Initialize the model
sigma = 0.1  # You need to define sigma value
h_theta_model = DeepPPCA(sigma)

# Process the desired dataset
dataset_to_use = val_dataset  # Change to train_dataset/uniform dataset putted in dataset folder
diff_max_eigenvectors_list, min_eigenvalues_list = process_dataset(dataset_to_use,h_theta_model,sigma)

# Convert the lists to tensors
diff_max_eigenvectors_tensor = torch.tensor(diff_max_eigenvectors_list)
min_eigenvalues_tensor = torch.tensor(min_eigenvalues_list)

# Plot histogram of absolute differences in eigenvectors
plt.figure(figsize=(22, 6))
plt.subplot(1, 3, 1)
plot_histogram(diff_max_eigenvectors_tensor, 'Histogram: Maximum of of Absolute Differences in Eigenvectors')

# Plot histogram of minimum eigenvalues
plt.subplot(1, 3, 2)
plot_histogram(min_eigenvalues_tensor, 'Histogram : Minimum Eigenvalues')

# Scatter plot
plt.subplot(1, 3, 3)  
create_scatter_plot(diff_max_eigenvectors_tensor, min_eigenvalues_tensor, title="Scatter Plot eigen vector and eigenvalues")

# Display plots
plt.tight_layout()
plt.show()


# def process_dataset(dataset):
#     diff_max_eigenvectors_list = []
#     min_eigenvalues_list = []
#     for q, _ in dataset:
#         print(f"Feature batch shape: {q.size()}")
#         #print(f"Labels batch shape: {label.size()}")
#         H = h_theta_model(q)
#         #delta_q = 0.01
#         #q_prime = q + delta_q
#         H_prime = h_theta_model(q_prime)
#         diff_max_eigenvector = calculate_eigen_vectors(H, H_prime)
#         min_eigenvalues = calculate_eigen_values(H)
#         diff_max_eigenvectors_list.append(diff_max_eigenvector.item())
#         min_eigenvalues_list.append(min_eigenvalues[0])
#     return diff_max_eigenvectors_list, min_eigenvalues_list