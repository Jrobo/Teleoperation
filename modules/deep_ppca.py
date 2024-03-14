import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib import animation


# Model
class DeepPPCA(nn.Module):
    def __init__(self, sigma=0.001):
        super(DeepPPCA, self).__init__()
        self.cov_matrix = None
        #self.dataset = {'ret': [], 'robot_state_torch': []}
        #self.fig, self.ax = plt.subplots()
        self.sigma = sigma
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 49)
        self.robot_state_torch = torch.tensor(np.zeros(7))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        H = self.fc3(x)
        H = H.view(-1, 7, 7)
        #print("size of H",H.shape)
        return H
    
    def log_likelihood(self, data):
        H = self(data)
        covariance_matrix = H @ H.transpose(1, 2) + self.sigma**2 * torch.eye(data.size(1))
        mvn = torch.distributions.MultivariateNormal(torch.zeros(data.size(1)), covariance_matrix)
        log_prob = mvn.log_prob(data)
        #print("shape of log [prob]",log_prob.shape())
        det_cov = torch.det(covariance_matrix)
        return log_prob, det_cov
        
        
    def get_transformation(self, data):
            """
            Data is a 7-dimensional column vector
            """
            H = self(data)
            covariance_matrix = H @ H.transpose(1,2) #+ self.sigma**2 * torch.eye(data.size(1))
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix.squeeze())
            idx = eigenvalues.argsort().flip([0])
            #print(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvalues_sign = torch.sign(eigenvalues[idx])
            #print(torch.sign(eigenvectors[0, idx]), eigenvectors[:, idx])
            eigenvectors = eigenvectors[:,idx] * torch.sign(eigenvectors[0, idx])
            #print(eigenvectors)
            """ # Plot heatmap
            plt.imshow(covariance_matrix.squeeze().numpy(), cmap='viridis')
            plt.title('Covariance Matrix Heatmap')
            plt.colorbar()
            plt.show()"""
            return covariance_matrix, eigenvectors

    def get_mode_transformation(self, data, mode):
        """
        Mode0:col 1 & 2 ; Mode1:col 3 & 4 ; Mode2:col 5 & 6 ; Mode3:col 6 & 7
        """
        _, eigenvectors = self.get_transformation(data)
        if mode < 3:
            return eigenvectors[:, mode*2:mode*2+2]
        else:
            return eigenvectors[:, -2:]

    def predict_velocities(self, robot_state, joystick, mode=0):
            robot_state_torch = torch.tensor(robot_state, dtype=torch.float32).unsqueeze(0)
            joystick_torch = torch.tensor(joystick, dtype=torch.float32)
            H_pred=self.get_mode_transformation(robot_state_torch, mode)
            self.robot_state_torch = robot_state_torch
            #print("H prediction before multiplying joystick values",H_pred)
            ret = (H_pred @ joystick_torch).squeeze()  
            return ret
        
    
    def create_and_save_heatmap_animation(self, cov_matrix, eigenvectors, mode_values):
        # figure 
        fig = plt.figure()

        # initialization
        def init():
            # initial heatmap 7x7 
            sns.heatmap(torch.zeros((7, 7)).numpy(), vmax=0.8, square=True, cbar=False)

        # animation
        def animate(i, *args):
            # arguments
            cov_matrix, eigenvectors, mode_values = args

           
            plt.subplot(1, 2, 1)
             # covariance matrix heatmap
            data_cov = cov_matrix.detach().squeeze().numpy()
            sns.heatmap(data_cov, vmax=0.8, square=True, cbar=False, cmap='viridis')
            plt.title('Covariance Matrix Heatmap')  # Title for the subplot

            
            plt.subplot(1, 2, 2)
            mode = mode_values[0]
            columns_to_plot = [mode * 2, mode * 2 + 1]
            data_eig = eigenvectors[:, columns_to_plot].detach().numpy()

            #eigenvectors heatmap
            sns.heatmap(data_eig, vmax=0.8, square=True, cbar=False, cmap='viridis')
            plt.title(f'Eigenvector Columns: {columns_to_plot[0]} and {columns_to_plot[1]}')

        # animation 
        anim = animation.FuncAnimation(fig, animate, fargs=(cov_matrix, eigenvectors, mode_values),
                                    init_func=init, frames=20, interval=50, repeat=False)

        # save
        savefile = r"test3.gif"
        
        
        pillowwriter = animation.PillowWriter(fps=20)
        
        # save GIF
        anim.save(savefile, writer=pillowwriter)

        
        plt.show()

        
        return fig


    def update_plot(self):
            cov_matrix, _ = self.get_transformation(self.robot_state_torch)
            plt.clf()
            plt.imshow(cov_matrix.squeeze().detach().numpy(), cmap='viridis', interpolation='nearest')
            plt.draw()
            plt.pause(0.001)          
     
    def start_animation(self):
        plt.ion()



    # def create_and_save_heatmap_animation(self, cov_matrix):
    #     fig = plt.figure()

    #     def init():
    #         sns.heatmap(torch.zeros((7, 7)).numpy(), vmax=0.8, square=True, cbar=False)

    #     def animate(i):
    #         data = cov_matrix.detach().squeeze().numpy()
    #         sns.heatmap(data, vmax=0.8, square=True, cbar=False)


    #     data_list = []
    #     for j in range(200):
    #         data_list.append(cov_matrix)

    #     anim = animation.FuncAnimation(fig, animate, init_func=init, frames=20, repeat=False)

    #     savefile = r"test3.gif"
    #     pillowwriter = animation.PillowWriter(fps=20)
    #     anim.save(savefile, writer=pillowwriter)

    #     plt.show()
        
    """#plotting the one d vector and covarience matrix
    def create_and_save_heatmap_animation(self,cov_matrix, eigenvectors):
        fig = plt.figure()

        def init():
            sns.heatmap(torch.zeros((7, 7)).numpy(), vmax=0.8, square=True, cbar=False)

        def animate(i, *args):
            cov_matrix, eigenvectors = args
            plt.subplot(1, 2, 1)
            data_cov = cov_matrix.detach().squeeze().numpy()
            sns.heatmap(data_cov, vmax=0.8, square=True, cbar=False, cmap='viridis')


            plt.subplot(1, 2, 2)
            data_eig = eigenvectors[:, i % 7].detach().numpy()
            

            sns.heatmap(data_eig.reshape(1, -1), vmax=0.8, square=True, cbar=False,cmap='viridis')
            plt.title(f'Eigenvector {i}')

        anim = animation.FuncAnimation(fig, animate, fargs=(cov_matrix, eigenvectors), init_func=init, frames=20,interval=500,repeat=False)

        savefile = r"test3.gif"
        pillowwriter = animation.PillowWriter(fps=20)
        anim.save(savefile, writer=pillowwriter)

        plt.show()"""


