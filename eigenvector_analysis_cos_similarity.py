import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from modules.deep_ppca import DeepPPCA
import os
import subprocess

def generate_uniform_distribution(size):
    return torch.rand(size) * 2 * np.pi

def calculate_eigen_values(H, sigma):
    eig_H_transpose = torch.transpose(H, 1, 2)
    covariance_matrix = torch.matmul(H, eig_H_transpose) + sigma**2
    eigenvalues_cov_matrix, _ = torch.linalg.eig(covariance_matrix)
    min_eigenvalues = torch.min(torch.abs(eigenvalues_cov_matrix[:, :-1] - eigenvalues_cov_matrix[:, 1:]), axis=1)
    return min_eigenvalues

def calculate_eigen_vectors(H, H_prime, sigma):
    H_transpose = torch.transpose(H, 1, 2)
    H_prime_transpose = torch.transpose(H_prime, 1, 2)
    covariance_matrix = torch.matmul(H, H_transpose) + sigma**2
    covariance_matrix_prime = torch.matmul(H_prime, H_prime_transpose) + sigma**2  
    #print("Shape of covariance_matrix:", covariance_matrix.shape)
    
    #print(f'Shape of first element in delta_q_desired:',covariance_matrix_prime.shape)
    #print("Size of covariance_matrix_prime:", covariance_matrix_prime.size())  
    abs_Cov = torch.abs(covariance_matrix - covariance_matrix_prime)
    # Reshape the matrices into 2D tensors (batch_size x num_features)
    #covariance_matrix_prime_flat = covariance_matrix_prime.view(1, -1)
    #covariance_matrix_flat = covariance_matrix.view(1, -1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-08)
    cosine_similarity = cos(covariance_matrix,covariance_matrix_prime)
    #print("cosineSimilarity",cosine_similarity.shape)
    #print("Size of covariance_matrix:", covariance_matrix.size())
    max_Cov = torch.max(torch.sum(abs_Cov, dim=2))
    #print("max cov",max_Cov)
    cosine_similarity_list=cosine_similarity.tolist()
    return max_Cov,cosine_similarity_list

def calculate_delta_q_desired(delta_q_list, magnitude_delta_q_list, median_delta_q):
    # store
    delta_q_desired = []

    for delta_q, magnitude in zip(delta_q_list, magnitude_delta_q_list):
        # Normalize delta_q
        normalized_delta_q = delta_q / magnitude
        # desired delta_q
        desired_delta_q = normalized_delta_q * median_delta_q
        # Append to list
        delta_q_desired.append(desired_delta_q)
    
    return delta_q_desired



def process_dataset(dataset, h_theta_model, sigma):
    diff_max_eigenvectors_list = []
    min_eigenvalues_list = []
    cosine_similarity_list=[]
    delta_q_list=[]
    magnitude_delta_q_list=[]
    joint_positions = dataset['joint_positions']
    joint_velocities = dataset['joint_velocities']
    print("dataset loaded")

    num_samples = joint_positions.shape[0]
    #loop 1 to calculate the median
    for i in range(num_samples - 1):
        q = torch.tensor(joint_positions[i]).float()             
        q_prime = torch.tensor(joint_positions[i + 1]).float()   
        delta_q= q - q_prime
        delta_q_list.append(delta_q)
        magnitude = torch.norm(delta_q)                                        # the magnitude of delta_q
        magnitude_delta_q_list.append(magnitude) 

    # median of all delta_q tensors
    median_delta_q = torch.median(torch.stack(delta_q_list), dim=0).values 
    delta_q_desired_list = calculate_delta_q_desired(delta_q_list, magnitude_delta_q_list, median_delta_q)
    #print(f'Length of delta_q_desired: {len(delta_q_desired_list)}')
    #print(f'Shape of first element in delta_q_desired: {delta_q_desired_list.shape}')
    cosine_similarity_list = []
    #print("num_samples",num_samples)
    #loop 2 to find cosine similarity
    for i in range(num_samples - 1):
        q = torch.tensor(joint_positions[i]).float()
        q_prime_delta = torch.tensor(q + delta_q_desired_list[i]).float()  
        #models
        H = h_theta_model(q)
        H_prime_delta = h_theta_model(q_prime_delta)  
        # Calculate eigenvalues and eigenvectors
        diff_max_eigenvector,cosine_similarity = calculate_eigen_vectors(H, H_prime_delta, sigma)
        min_eigenvalues = calculate_eigen_values(H, sigma)
        # Append results to lists
        #cosine_similarity_list.append(diff_max_eigenvector.item())
        # Assuming cosine_similarity is a tensor with 7 elements
        #for cosine_similarity_value in cosine_similarity:
        cosine_similarity_list.append(cosine_similarity)
        diff_max_eigenvectors_list.append(diff_max_eigenvector.item())
        min_eigenvalues_list.append(min_eigenvalues[0].item())
    
    return diff_max_eigenvectors_list, min_eigenvalues_list, cosine_similarity_list 

def plot_cosine_similarity(cosine_similarity_list):
    """
    Plots the average cosine similarity over iterations.

    Parameters:
    cosine_similarity_list (list of list of floats): A list where each sublist contains
    cosine similarity values for an iteration.
    """
    first_column=[] 
    for a in cosine_similarity_list:
        first_column.append(a[0][1])
    # Print the first column
    print("First column length:", len(first_column))
    print("first column",first_column)
    # Generate a range of iterations of the same size
    iterations = list(range(1, len(first_column)+1))
    # # Create the plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(iterations, first_column, marker='o', linestyle='-', color='b')
    # # Add titles and labels
    # plt.title('Plot of Data with Respect to Iteration')
    # plt.xlabel('Iteration')
    # plt.ylabel('Data Value')
    # # Show grid
    # plt.grid(True)
    # # Display the plot
    # plt.show()

    cosine_similarities = cosine_similarity_list

    # Convert the list to a NumPy array for easier handling
    cosine_similarities = np.array(cosine_similarities)

    # Plot the cosine similarities
    plt.figure(figsize=(10, 6))

    # Iterate through each series of cosine similarities and plot them
    for i, series in enumerate(cosine_similarities):
        plt.plot(series, label=f'Series {i + 1}')

    # Add labels, title, legend, and grid to the plot
    plt.xlabel('Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity Over Indices')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


def plot_histogram(tensor_values, title):
    plt.hist(tensor_values.numpy(), bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')

def create_scatter_plot(diff_max_eigenvectors_list, min_eigenvalues_list, title):
    diff_max_eigenvectors_tensor = torch.tensor(diff_max_eigenvectors_list).float().clone().detach()  
    min_eigenvalues_tensor = torch.tensor(min_eigenvalues_list).float().clone().detach()  
    plt.scatter(diff_max_eigenvectors_tensor, min_eigenvalues_tensor, color='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel('Difference of Max Eigenvectors')
    plt.ylabel('Min Eigenvalues')
    plt.grid(True)
    plt.xscale('linear')
    plt.yscale('linear')

# Set random seed for reproducibility
torch.manual_seed(42)

# Load data
uniform_data = np.load('dataset/uniform_dataset.npz')
collected_data = np.load('dataset/joint_data_with_identifiers.npz')  # Corrected path for collected data

# Initialize the model
sigma = 0.1  # You need to define sigma value
h_theta_model = DeepPPCA(sigma)

# Process each dataset
diff_max_eigenvectors_list1, min_eigenvalues_list1,cosine_similarity1 = process_dataset(uniform_data, h_theta_model, sigma)
diff_max_eigenvectors_list2, min_eigenvalues_list2,cosine_similarity2 = process_dataset(collected_data, h_theta_model, sigma)
print("cosine similarity length",len(cosine_similarity2))

num_cols = len(cosine_similarity2[1]) 
print("number f columns", num_cols)

# Convert the lists to tensors
diff_max_eigenvectors_tensor1 = torch.tensor(diff_max_eigenvectors_list1).float().clone().detach()  # Ensure tensor is float
min_eigenvalues_tensor1 = torch.tensor(min_eigenvalues_list1).float().clone().detach()  # Ensure tensor is float

diff_max_eigenvectors_tensor2 = torch.tensor(diff_max_eigenvectors_list2).float().clone().detach()  # Ensure tensor is float
min_eigenvalues_tensor2 = torch.tensor(min_eigenvalues_list2).float().clone().detach()  # Ensure tensor is float

# Find the index of the maximum value along x axis
max_index1 = torch.argmax(diff_max_eigenvectors_tensor1)
max_index2 = torch.argmax(diff_max_eigenvectors_tensor2)

# Remove the maximum value and its associated value
diff_max_eigenvectors_tensor1 = torch.cat((diff_max_eigenvectors_tensor1[:max_index1], diff_max_eigenvectors_tensor1[max_index1+1:]))
min_eigenvalues_tensor1 = torch.cat((min_eigenvalues_tensor1[:max_index1], min_eigenvalues_tensor1[max_index1+1:]))

diff_max_eigenvectors_tensor2 = torch.cat((diff_max_eigenvectors_tensor2[:max_index2], diff_max_eigenvectors_tensor2[max_index2+1:]))
min_eigenvalues_tensor2 = torch.cat((min_eigenvalues_tensor2[:max_index2], min_eigenvalues_tensor2[max_index2+1:]))



# Plot all histograms and scatter plots together
plt.figure(figsize=(22, 12))

# Uniform data histograms and scatter plot
plt.subplot(2, 3, 1)
plot_histogram(diff_max_eigenvectors_tensor1, 'Histogram: Max Absolute Differences in Eigenvectors (Uniform Data)')

plt.subplot(2, 3, 2)
plot_histogram(min_eigenvalues_tensor1, 'Histogram: Min Eigenvalues (Uniform Data)')

plt.subplot(2, 3, 3)
create_scatter_plot(diff_max_eigenvectors_tensor1, min_eigenvalues_tensor1, "Scatter Plot: Eigenvector Differences vs. Eigenvalues (Uniform Data)")
#create_scatter_plot(cosine_similarity,cosine_similarity, "Scatter Plot: Eigenvector Differences vs. Eigenvalues (Uniform Data)")
# plt.subplot(2, 3, 6)
# create_scatter_plot(diff_max_eigenvectors_tensor2, min_eigenvalues_tensor2, "Scatter Plot: Eigenvector Differences vs. Eigenvalues (Collected Data)")

# Compute mean and standard deviation of the data points
mean_diff_max_eigenvectors = torch.mean(diff_max_eigenvectors_tensor2)
std_diff_max_eigenvectors = torch.std(diff_max_eigenvectors_tensor2)

mean_min_eigenvalues = torch.mean(min_eigenvalues_tensor2)
std_min_eigenvalues = torch.std(min_eigenvalues_tensor2)

# Define threshold for including points ( within 1 standard deviation)
threshold_diff_max_eigenvectors = mean_diff_max_eigenvectors + std_diff_max_eigenvectors
threshold_min_eigenvalues = mean_min_eigenvalues + std_min_eigenvalues

# Filter points based on threshold
filtered_diff_max_eigenvectors_tensor2 = diff_max_eigenvectors_tensor2[torch.logical_and(diff_max_eigenvectors_tensor2 < threshold_diff_max_eigenvectors, min_eigenvalues_tensor2 < threshold_min_eigenvalues)]
filtered_min_eigenvalues_tensor2 = min_eigenvalues_tensor2[torch.logical_and(diff_max_eigenvectors_tensor2 < threshold_diff_max_eigenvectors, min_eigenvalues_tensor2 < threshold_min_eigenvalues)]

# Collected data histograms and scatter plot
plt.subplot(2, 3, 4)
plot_histogram(filtered_diff_max_eigenvectors_tensor2, 'Histogram: Max Absolute Differences in Eigenvectors (Collected Data)')

plt.subplot(2, 3, 5)
plot_histogram(filtered_min_eigenvalues_tensor2, 'Histogram: Min Eigenvalues (Collected Data)')

# Scatter plot with filtered points
plt.subplot(2, 3, 6)
create_scatter_plot(filtered_diff_max_eigenvectors_tensor2, filtered_min_eigenvalues_tensor2, "Scatter Plot: Eigenvector Differences vs. Eigenvalues (Collected Data, Filtered)")

plot_cosine_similarity(cosine_similarity2)






# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from modules.deep_ppca import DeepPPCA

# def generate_uniform_distribution(size):
#     return torch.rand(size) * 2 * np.pi

# def calculate_eigen_values(H, sigma):
#     eig_H_transpose = torch.transpose(H, 1, 2)
#     covariance_matrix = torch.matmul(H, eig_H_transpose) + sigma**2
#     eigenvalues_cov_matrix, _ = torch.linalg.eig(covariance_matrix)
#     min_eigenvalues = torch.min(torch.abs(eigenvalues_cov_matrix[:, :-1] - eigenvalues_cov_matrix[:, 1:]), axis=1)
#     return min_eigenvalues

# def calculate_eigen_vectors(H, H_prime, sigma):
#     H_transpose = torch.transpose(H, 1, 2)
#     H_prime_transpose = torch.transpose(H_prime, 1, 2)
#     covariance_matrix = torch.matmul(H, H_transpose) + sigma**2
#     covariance_matrix_prime = torch.matmul(H_prime, H_prime_transpose) + sigma**2
#     abs_Cov = torch.abs(covariance_matrix - covariance_matrix_prime)
#     max_Cov = torch.max(torch.sum(abs_Cov, dim=2))
#     return max_Cov

# def process_dataset(dataset, h_theta_model, sigma):
#     diff_max_eigenvectors_list = []
#     min_eigenvalues_list = []

#     joint_positions = dataset['joint_positions']
#     joint_velocities = dataset['joint_velocities']

#     num_samples = joint_positions.shape[0]
    
#     for i in range(num_samples - 1):
#         q = torch.tensor(joint_positions[i])
#         q_prime = torch.tensor(joint_positions[i + 1])
        
#         H = h_theta_model(q)
#         H_prime = h_theta_model(q_prime)
        
#         diff_max_eigenvector = calculate_eigen_vectors(H, H_prime, sigma)
#         min_eigenvalues = calculate_eigen_values(H, sigma)
        
#         diff_max_eigenvectors_list.append(diff_max_eigenvector.item())
#         min_eigenvalues_list.append(min_eigenvalues[0].item())

#     return diff_max_eigenvectors_list, min_eigenvalues_list

# def plot_histogram(tensor_values, title):
#     plt.hist(tensor_values.numpy(), bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Density')

# def create_scatter_plot(diff_max_eigenvectors_list, min_eigenvalues_list, title):
#     diff_max_eigenvectors_tensor = torch.tensor(diff_max_eigenvectors_list).clone().detach()
#     min_eigenvalues_tensor = torch.tensor(min_eigenvalues_list).clone().detach()
#     plt.scatter(diff_max_eigenvectors_tensor, min_eigenvalues_tensor, color='blue', alpha=0.5)
#     plt.title(title)
#     plt.xlabel('Difference of Max Eigenvectors')
#     plt.ylabel('Min Eigenvalues')
#     plt.grid(True)
#     plt.xscale('linear')
#     plt.yscale('linear')

# # Set random seed for reproducibility
# torch.manual_seed(42)

# # Load data
# uniform_data = np.load('dataset/uniform_dataset.npz')
# collected_data = np.load('dataset/joint_data_with_identifiers.npz')  # Corrected path for collected data

# # Initialize the model
# sigma = 0.1  # You need to define sigma value
# h_theta_model = DeepPPCA(sigma)

# # Process each dataset
# diff_max_eigenvectors_list1, min_eigenvalues_list1 = process_dataset(uniform_data, h_theta_model, sigma)
# diff_max_eigenvectors_list2, min_eigenvalues_list2 = process_dataset(collected_data, h_theta_model, sigma)

# # Convert the lists to tensors
# diff_max_eigenvectors_tensor1 = torch.tensor(diff_max_eigenvectors_list1).clone().detach()
# min_eigenvalues_tensor1 = torch.tensor(min_eigenvalues_list1).clone().detach()

# diff_max_eigenvectors_tensor2 = torch.tensor(diff_max_eigenvectors_list2).clone().detach()
# min_eigenvalues_tensor2 = torch.tensor(min_eigenvalues_list2).clone().detach()

# # Find the index of the maximum value along x axis
# max_index1 = torch.argmax(diff_max_eigenvectors_tensor1)
# max_index2 = torch.argmax(diff_max_eigenvectors_tensor2)

# # Remove the maximum value and its associated value
# diff_max_eigenvectors_tensor1 = torch.cat((diff_max_eigenvectors_tensor1[:max_index1], diff_max_eigenvectors_tensor1[max_index1+1:]))
# min_eigenvalues_tensor1 = torch.cat((min_eigenvalues_tensor1[:max_index1], min_eigenvalues_tensor1[max_index1+1:]))

# diff_max_eigenvectors_tensor2 = torch.cat((diff_max_eigenvectors_tensor2[:max_index2], diff_max_eigenvectors_tensor2[max_index2+1:]))
# min_eigenvalues_tensor2 = torch.cat((min_eigenvalues_tensor2[:max_index2], min_eigenvalues_tensor2[max_index2+1:]))

# # Plot histograms and scatter plots for each dataset separately

# # For uniform_data
# plt.figure(figsize=(22, 6))

# plt.subplot(1, 3, 1)
# plot_histogram(diff_max_eigenvectors_tensor1, 'Histogram: Maximum of Absolute Differences in Eigenvectors (Uniform Data)')

# plt.subplot(1, 3, 2)
# plot_histogram(min_eigenvalues_tensor1, 'Histogram: Minimum Eigenvalues (Uniform Data)')

# plt.subplot(1, 3, 3)
# create_scatter_plot(diff_max_eigenvectors_tensor1, min_eigenvalues_tensor1, title="Scatter Plot: Eigenvector Differences vs. Eigenvalues (Uniform Data)")

# plt.tight_layout()
# plt.show()

# # For collected_data
# plt.figure(figsize=(22, 6))

# plt.subplot(1, 3, 1)
# plot_histogram(diff_max_eigenvectors_tensor2, 'Histogram: Maximum of Absolute Differences in Eigenvectors (Collected Data)')

# plt.subplot(1, 3, 2)
# plot_histogram(min_eigenvalues_tensor2, 'Histogram: Minimum Eigenvalues (Collected Data)')

# plt.subplot(1, 3, 3)
# create_scatter_plot(diff_max_eigenvectors_tensor2, min_eigenvalues_tensor2, title="Scatter Plot: Eigenvector Differences vs. Eigenvalues (Collected Data)")

# plt.tight_layout()
# plt.show()



# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from modules.deep_ppca import DeepPPCA

# def generate_uniform_distribution(size):
#     return torch.rand(size) * 2 * np.pi

# def calculate_eigen_values(H, sigma):
#     eig_H_transpose = torch.transpose(H, 1, 2)
#     covariance_matrix = torch.matmul(H, eig_H_transpose) + sigma**2
#     eigenvalues_cov_matrix, _ = torch.linalg.eig(covariance_matrix)
#     min_eigenvalues = torch.min(torch.abs(eigenvalues_cov_matrix[:, :-1] - eigenvalues_cov_matrix[:, 1:]), axis=1)
#     return min_eigenvalues

# def calculate_eigen_vectors(H, H_prime, sigma):
#     H_transpose = torch.transpose(H, 1, 2)
#     H_prime_transpose = torch.transpose(H_prime, 1, 2)
#     covariance_matrix = torch.matmul(H, H_transpose) + sigma**2
#     covariance_matrix_prime = torch.matmul(H_prime, H_prime_transpose) + sigma**2
#     abs_Cov = torch.abs(covariance_matrix - covariance_matrix_prime)
#     max_Cov = torch.max(torch.sum(abs_Cov, dim=2))
#     return max_Cov

# def process_dataset(dataset, h_theta_model, sigma):
#     diff_max_eigenvectors_list = []
#     min_eigenvalues_list = []

#     joint_positions = dataset['joint_positions']
#     joint_velocities = dataset['joint_velocities']

#     num_samples = joint_positions.shape[0]
    
#     for i in range(num_samples - 1):
#         q = torch.tensor(joint_positions[i])
#         q_prime = torch.tensor(joint_positions[i + 1])
        
#         H = h_theta_model(q)
#         H_prime = h_theta_model(q_prime)
        
#         diff_max_eigenvector = calculate_eigen_vectors(H, H_prime, sigma)
#         min_eigenvalues = calculate_eigen_values(H, sigma)
        
#         diff_max_eigenvectors_list.append(diff_max_eigenvector.item())
#         min_eigenvalues_list.append(min_eigenvalues[0].item())

#     return diff_max_eigenvectors_list, min_eigenvalues_list

# def plot_histogram(tensor_values, title):
#     plt.hist(tensor_values.numpy(), bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Density')

# def create_scatter_plot(diff_max_eigenvectors_list, min_eigenvalues_list, title):
#     diff_max_eigenvectors_tensor = torch.tensor(diff_max_eigenvectors_list).clone().detach()
#     min_eigenvalues_tensor = torch.tensor(min_eigenvalues_list).clone().detach()
#     plt.scatter(diff_max_eigenvectors_tensor, min_eigenvalues_tensor, color='blue', alpha=0.5)
#     plt.title(title)
#     plt.xlabel('Difference of Max Eigenvectors')
#     plt.ylabel('Min Eigenvalues')
#     plt.grid(True) 
#     plt.xscale('linear')
#     plt.yscale('linear')

# # Set random seed for reproducibility
# torch.manual_seed(42)

# # Load data
# uniform_data = np.load('dataset/uniform_dataset.npz')
# collected_data = np.load('dataset/uniform_dataset.npz')
# # Initialize the model
# sigma = 0.1  # You need to define sigma value
# h_theta_model = DeepPPCA(sigma)


# # Process the desired dataset
# dataset_to_use = uniform_data
# # List all the arrays stored in the NPZ file
# print("Arrays in NPZ file:", uniform_data.files)
# for key in uniform_data.files:
#     array_size = uniform_data[key].size
#     print(f"Size of '{key}' array:", array_size)

# print("Arrays in NPZ file:", collected_data.files)
# for key in collected_data.files:
#     array_size = collected_data[key].size
#     print(f"Size of '{key}' array:", array_size)


# diff_max_eigenvectors_list1, min_eigenvalues_list1 = process_dataset(uniform_data, h_theta_model, sigma)
# diff_max_eigenvectors_list2, min_eigenvalues_list2 = process_dataset(uniform_data, h_theta_model, sigma)

# # Convert the lists to tensors
# diff_max_eigenvectors_tensor1 = torch.tensor(diff_max_eigenvectors_list1)
# min_eigenvalues_tensor1 = torch.tensor(min_eigenvalues_list1)

# diff_max_eigenvectors_tensor2 = torch.tensor(diff_max_eigenvectors_list2)
# min_eigenvalues_tensor2 = torch.tensor(min_eigenvalues_list2)

# # Find the index of the maximum value along x axis
# max_index1 = torch.argmax(diff_max_eigenvectors_tensor1)
# max_index2 = torch.argmax(diff_max_eigenvectors_tensor2)
# # Remove the maximum value and its associated value
# diff_max_eigenvectors_tensor1 = torch.cat((diff_max_eigenvectors_tensor1[:max_index1], diff_max_eigenvectors_tensor1[max_index1+1:]))
# min_eigenvalues_tensor1 = torch.cat((min_eigenvalues_tensor1[:max_index1], min_eigenvalues_tensor1[max_index1+1:]))

# diff_max_eigenvectors_tensor2 = torch.cat((diff_max_eigenvectors_tensor2[:max_index2], diff_max_eigenvectors_tensor2[max_index2+1:]))
# min_eigenvalues_tensor2 = torch.cat((min_eigenvalues_tensor2[:max_index2], min_eigenvalues_tensor1[max_index2+1:]))

# # Plot histogram of absolute differences in eigenvectors
# plt.figure(figsize=(22, 6))
# plt.subplot(1, 3, 1)
# plot_histogram(diff_max_eigenvectors_tensor1, 'Histogram: Maximum of Absolute Differences in Eigenvectors')

# # Plot histogram of absolute differences in eigenvectors
# plt.figure(figsize=(22, 6))
# plt.subplot(2, 3, 1)
# plot_histogram(diff_max_eigenvectors_tensor2, 'Histogram: Maximum of Absolute Differences in Eigenvectors')

# # Plot histogram of minimum eigenvalues
# plt.subplot(1, 3, 2)
# plot_histogram(min_eigenvalues_tensor1, 'Histogram: Minimum Eigenvalues')

# # Plot histogram of minimum eigenvalues
# plt.subplot(2, 3, 2)
# plot_histogram(min_eigenvalues_tensor2, 'Histogram: Minimum Eigenvalues')
# # Scatter plot
# plt.subplot(1, 3, 3)
# create_scatter_plot(diff_max_eigenvectors_tensor1, min_eigenvalues_tensor1, title="Scatter Plot eigen vector and eigenvalues")

# # Scatter plot
# plt.subplot(2, 3, 3)
# create_scatter_plot(diff_max_eigenvectors_tensor2, min_eigenvalues_tensor2, title="Scatter Plot eigen vector and eigenvalues")

# # Display plots
# plt.tight_layout()
# plt.show()

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from modules.deep_ppca import DeepPPCA
# from modules.data_load_module import load_data, create_data_loaders

# def generate_uniform_distribution(size):
#     return torch.rand(size) * 2 * np.pi

# def calculate_eigen_values(H,sigma):
#     eig_H_transpose=torch.transpose(H, 1, 2)
#     covariance_matrix=torch.matmul(H, eig_H_transpose) + sigma**2
#     eigenvalues_cov_matrix, _ = torch.linalg.eig(covariance_matrix)
#     #print("Size of covariance matrix:", covariance_matrix.size())
#     min_eigenvalues = torch.min(torch.abs(eigenvalues_cov_matrix[:, :-1] - eigenvalues_cov_matrix[:, 1:]), axis=1)
#     return min_eigenvalues

# def calculate_eigen_vectors(H, H_prime,sigma):
#     H_transpose=torch.transpose(H, 1, 2)
#     H_prime_transpose=torch.transpose(H_prime, 1, 2)
#     #print("Size of transposed matrix:", H_transpose.size())  # Output: torch.Size([1, 7, 7])
#     # Calculate covariance matrix
#     covariance_matrix = torch.matmul(H, H_transpose) + sigma**2
#     covariance_matrix_prime = torch.matmul(H_prime, H_prime_transpose) + sigma**2
#     #print("Size of covariance matrix:", covariance_matrix.size()) # Output: torch.Size([1, 7, 7])
#     abs_Cov = torch.abs(covariance_matrix-covariance_matrix_prime)
#     max_Cov = torch.max(torch.sum(abs_Cov, dim=2))
#     #diff_max_eigenvector = max_eigenvectors_H_prime - max_eigenvectors_H
#     return max_Cov

# def process_dataset(dataset, h_theta_model,sigma):
#     diff_max_eigenvectors_list = []
#     min_eigenvalues_list = []

#     # Create an iterator over the DataLoader
#     data_iterator = iter(dataset)
    
#     additional_value=[]
#     # Iterate over pairs of consecutive elements
#     try:
#         while True:
#             data = next(data_iterator)
#             print("Retrieved data:", data)
#             q, _ = data  # Try to unpack here
#             #q, _ = next(data_iterator)  # Current q value
#             q_prime, _ = next(data_iterator)  # Next q value
#             print(f" shape (q): {q.size()}")
#             print(f" shape (q_prime): {q_prime.size()}")
#             # h_theta_model takes a tensor as input
#             H = h_theta_model(q)
#             print("H",H.size())
#             H_prime = h_theta_model(q_prime)
#             print("H_prime",H_prime.size())
#             # Calculate eigen vectors and values
#             diff_max_eigenvector = calculate_eigen_vectors(H, H_prime,sigma)
#             min_eigenvalues = calculate_eigen_values(H,sigma)
#             # Append results to lists
#             diff_max_eigenvectors_list.append(diff_max_eigenvector.item())
#             min_eigenvalues_list.append(min_eigenvalues[0])
#     except StopIteration:
#         pass

#     return diff_max_eigenvectors_list, min_eigenvalues_list

# def plot_histogram(tensor_values, title):
#     plt.hist(tensor_values.numpy(), bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Density')

# def create_scatter_plot(diff_max_eigenvectors_list, min_eigenvalues_list, title):
#     diff_max_eigenvectors_tensor = torch.tensor(diff_max_eigenvectors_list).clone().detach()
#     print("max value along x axis is",torch.max(diff_max_eigenvectors_tensor))
#     print("min value along x axis is",torch.min(diff_max_eigenvectors_tensor))
#     min_eigenvalues_tensor = torch.tensor(min_eigenvalues_list).clone().detach()
#     plt.scatter(diff_max_eigenvectors_tensor, min_eigenvalues_tensor, color='blue', alpha=0.5)
#     plt.title(title)
#     plt.xlabel('Difference of Max Eigenvectors')
#     plt.ylabel('Min Eigenvalues')
#     plt.grid(True) 
#     plt.xscale('linear')
#     plt.yscale('linear')

# # Set random seed for reproducibility
# torch.manual_seed(42)

# # Load data
# #train_dataset, val_dataset, test_dataset = load_data('dataset/joint_data_with_identifiers.npz')
# uniform_data = np.load('dataset/uniform_dataset.npz')
# # Initialize the model
# sigma = 0.1  # You need to define sigma value
# h_theta_model = DeepPPCA(sigma)

# # List all the arrays stored in the NPZ file
# print("Arrays in NPZ file:", uniform_data.files)
# # Access and print the size of each array
# for key in uniform_data.files:
#     array_size = uniform_data[key].size
#     print(f"Size of '{key}' array:", array_size)


# # Process the desired dataset
# dataset_to_use = uniform_data#val_dataset  # Change to train_dataset/uniform dataset putted in dataset folder
# diff_max_eigenvectors_list, min_eigenvalues_list = process_dataset(dataset_to_use,h_theta_model,sigma)

# # Convert the lists to tensors
# diff_max_eigenvectors_tensor = torch.tensor(diff_max_eigenvectors_list)
# min_eigenvalues_tensor = torch.tensor(min_eigenvalues_list)



# # Find the index of the maximum value along x axis
# max_index = torch.argmax(diff_max_eigenvectors_tensor)

# # Remove the maximum value and its associated value
# diff_max_eigenvectors_tensor = torch.cat((diff_max_eigenvectors_tensor[:max_index], diff_max_eigenvectors_tensor[max_index+1:]))
# min_eigenvalues_tensor = torch.cat((min_eigenvalues_tensor[:max_index], min_eigenvalues_tensor[max_index+1:]))




# # Plot histogram of absolute differences in eigenvectors
# plt.figure(figsize=(22, 6))
# plt.subplot(1, 3, 1)
# plot_histogram(diff_max_eigenvectors_tensor, 'Histogram: Maximum of of Absolute Differences in Eigenvectors')

# # Plot histogram of minimum eigenvalues
# plt.subplot(1, 3, 2)
# plot_histogram(min_eigenvalues_tensor, 'Histogram : Minimum Eigenvalues')

# # Scatter plot
# plt.subplot(1, 3, 3)  
# create_scatter_plot(diff_max_eigenvectors_tensor, min_eigenvalues_tensor, title="Scatter Plot eigen vector and eigenvalues")

# # Display plots
# plt.tight_layout()
# plt.show()


# # def process_dataset(dataset):
# #     diff_max_eigenvectors_list = []
# #     min_eigenvalues_list = []
# #     for q, _ in dataset:
# #         print(f"Feature batch shape: {q.size()}")
# #         #print(f"Labels batch shape: {label.size()}")
# #         H = h_theta_model(q)
# #         #delta_q = 0.01
# #         #q_prime = q + delta_q
# #         H_prime = h_theta_model(q_prime)
# #         diff_max_eigenvector = calculate_eigen_vectors(H, H_prime)
# #         min_eigenvalues = calculate_eigen_values(H)
# #         diff_max_eigenvectors_list.append(diff_max_eigenvector.item())
# #         min_eigenvalues_list.append(min_eigenvalues[0])
# #     return diff_max_eigenvectors_list, min_eigenvalues_list