import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib import animation
import datetime

# Model
class DeepPPCA(nn.Module):
    def __init__(self, sigma):
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
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))#change to l linear
        H = self.fc3(x)
        H = H.view(-1, 7, 7)
        #print("size of H",H.shape)
        return H
    
    # def log_likelihood(self, data):
    #     H = self(data)
    #     covariance_matrix = H @ H.transpose(1, 2) + self.sigma**2 * torch.eye(data.size(1))
    #     print(covariance_matrix)
    #     #epsilon = 1e-4
    #     #covariance_matrix = covariance_matrix + epsilon * torch.eye(covariance_matrix.size(-1))
    #     mvn = torch.distributions.MultivariateNormal(torch.zeros(data.size(1)), covariance_matrix)
    #     log_prob = mvn.log_prob(data)
    #     #print("shape of log [prob]",log_prob.shape())
    #     det_cov = torch.det(covariance_matrix)
    #     return log_prob, det_cov
    
    def log_likelihood(self, data):
        '''here we collect the data of input (4,7) and get H [ 7, 7] for 4 , so size is (4,7,7,), 
        then we add cov_diag as noise by repeting to 4 rows
        shape of mvn is (4,7)
        Output=data(4,7)'''
        H = self(data)#shape of H torch.Size([4, 7, 7])
        #print("shape of H",H.size())
        n_columns = data.shape[1] #n_columns shape 7
        #print("n_columns shape",n_columns)
        n_rows = data.shape[0]#n rows shape 4
<<<<<<< HEAD
        print("n rows shape",n_rows)
        #ask: why use this formula, not include HHT
        cov_diag = (self.sigma**2 * torch.ones(n_columns)).reshape((1, n_columns))#cov diagonal before repeat torch.Size([1, 7])
        print("cov diagonal before repeat",cov_diag.size())
        '''here we are adding noise of sigma value to all four datas in data loader, so repeated'''
=======
        #print("n rows shape",n_rows)
        cov_diag = (self.sigma**2 * torch.ones(n_columns)).reshape((1, n_columns))#cov diagonal before repeat torch.Size([1, 7])
        #print("cov diagonal before repeat",cov_diag.size())
>>>>>>> ce46ca8cd2b7a743e7e588b3cd36981862958b02
        cov_diag = cov_diag.repeat(n_rows, 1)#cov diag after repeat torch.Size([4, 7])
        #print("cov diag after repeat",cov_diag.size())
        #covariance_matrix = H @ H.transpose(1, 2) + cov_diag

        try:
            mvn = torch.distributions.LowRankMultivariateNormal(torch.zeros_like(data), cov_factor=H, cov_diag=cov_diag)#mvn shape: torch.Size([7])
<<<<<<< HEAD
            print("mvn shape:", mvn.event_shape)#mvn shape: torch.Size([7])
=======
            #print("mvn shape:", mvn.event_shape)
>>>>>>> ce46ca8cd2b7a743e7e588b3cd36981862958b02
            log_prob = mvn.log_prob(data)#Shape of log_prob: torch.Size([4])
            #print("Shape of log_prob:", log_prob.shape)
            det_cov = None #torch.det(covariance_matrix)
        except  Exception as e:     #torch.linalg.LinAlgError:
            print(e)
            print("Covariance matrix is not positive definite.")
            #print(covariance_matrix)
            np.save(f'analysis/Covarience_matrix/cov_mtrx_not_pos_def_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_sigma_{self.sigma}.npy', H.detach().cpu().numpy())
            #np.save('analysis/covariance_matrix_not_positive_definite.npy', covariance_matrix.detach().cpu().numpy())
            # Returning default values to prevent runtime error
            return torch.tensor(0.0), torch.tensor(0.0)

        return log_prob, det_cov

            
    def get_transformation(self, data):
            """
            Data is a 7-dimensional column vector
            """
            H = self(data)
            covariance_matrix = H @ H.transpose(1,2) #+ self.sigma**2 * torch.eye(data.size(1))
            #print("covarience matrix inside get transformation function",covariance_matrix.size)
            try:
                chol_cov_matrix = torch.cholesky(covariance_matrix)
            except torch.linalg.LinAlgError:
                # Handle the case 
                print("Covariance matrix is not positive definite.")
                print(covariance_matrix)
                        
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix.squeeze())
            idx = eigenvalues.argsort().flip([0])
            #print(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvalues_sign = torch.sign(eigenvalues[idx])
            #print(torch.sign(eigenvectors[0, idx]), eigenvectors[:, idx])
            eigenvectors = eigenvectors[:,idx] * torch.sign(eigenvectors[0, idx])
            #print("Eigen vectors calculted in get transformation function shape",eigenvectors.size)
            #print(eigenvectors)
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
            #print("H_pred calculated inside predicted velocuty function",H_pred.size)
            self.robot_state_torch = robot_state_torch
            #print("H prediction before multiplying joystick values",H_pred)
            ret = (H_pred @ joystick_torch).squeeze()  
            return ret
        
    # def create_and_save_heatmap_animation(self, cov_matrix, eigenvectors, mode_values):
    # # Create figure
    #     fig, axs = plt.subplots(1, 2)

    #     # Initialization
    #     def init():
    #         # Initial heatmap 7x7 
    #         sns.heatmap(torch.zeros((7, 7)).numpy(), vmax=0.8, square=True, cbar=False, ax=axs[0])

        # # Animation
        # def animate(frame, *args):
        #     # Arguments
        #     cov_matrix, eigenvectors, mode_values = args[0]

        #     # Clear previous plots
        #     axs[0].cla()
        #     axs[1].cla()

        #     # Covariance matrix heatmap
        #     data_cov = cov_matrix.detach().squeeze().numpy()
        #     sns.heatmap(data_cov, vmax=0.8, square=True, cbar=False, cmap='viridis', ax=axs[0])
        #     axs[0].set_title('Covariance Matrix Heatmap')

        #     # Eigenvectors heatmap
        #     mode = mode_values[0]
        #     columns_to_plot = [mode * 2, mode * 2 + 1]
        #     data_eig = eigenvectors[:, columns_to_plot].detach().numpy()
        #     sns.heatmap(data_eig, vmax=0.8, square=True, cbar=False, cmap='viridis', ax=axs[1])
        #     axs[1].set_title(f'Eigenvector Columns: {columns_to_plot[0]} and {columns_to_plot[1]}')

        # # Animation 
        # anim = animation.FuncAnimation(fig, animate, fargs=([(cov_matrix, eigenvectors, mode_values)]),
        #                             init_func=init, frames=None, interval=50, repeat=False)

       
        # # Save the animation as a movie file
        # savefile = r"test3.mp4"
        # pillowwriter = animation.FFMpegWriter(fps=20)
        # anim.save(savefile, writer=pillowwriter)
        # plt.show()

    # def update_plot(self):
    #         cov_matrix, _ = self.get_transformation(self.robot_state_torch)
    #         plt.clf()
    #         plt.imshow(cov_matrix.squeeze().detach().numpy(), cmap='viridis', interpolation='nearest')
    #         plt.draw()
    #         plt.pause(0.001)          
     
    # def start_animation(self):
    #     plt.ion()

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


