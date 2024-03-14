import torch
import sys
import os
sys.path.append('/home/jamil/PyRep/projects')

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from modules.scl import HThetaNetwork
from modules.joystick_handler import JoystickHandler
from modules.deep_ppca import DeepPPCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ctypes import cdll
cdll.LoadLibrary('libX11.so.6').XInitThreads()

module = DeepPPCA(sigma=0.0001)
module.load_state_dict(torch.load('/home/jamil/PyRep/projects/saved_models/ppca_model.pth'))
module.eval()

# RL env
action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(action_mode)
env.launch()
task = env.get_task(WaterPlants)
descriptions, obs = task.reset()
joystick_handler = JoystickHandler()

# Plot
fig, axs = plt.subplots(3, 1, figsize=(6, 15))

# Initialize the color bar for covariance matrix
cax_cov = fig.add_axes([0.92, 0.7, 0.02, 0.15])
heatmap_cov = sns.heatmap(np.zeros((7, 7)), annot=True, fmt=".4f", cmap='viridis', ax=axs[0], cbar_ax=cax_cov)
axs[0].set_title('Real-Time Covariance Matrix')

# Initialize the color bar for eigenvectors
cax_eig = fig.add_axes([0.92, 0.4, 0.02, 0.15])
heatmap_eigenvectors = sns.heatmap(np.zeros((7, 2)), annot=True, fmt=".4f", cmap='viridis', ax=axs[1], cbar_ax=cax_eig)
axs[1].set_title('Real-Time Eigenvectors')

# Initialize the color bar for eigenvalues
cax_eigenvalues = fig.add_axes([0.92, 0.1, 0.02, 0.15])
heatmap_eigenvalues = sns.heatmap(np.zeros((1, 7)), annot=True, fmt=".4f", cmap='viridis', ax=axs[2], cbar_ax=cax_eigenvalues)
axs[2].set_title('Real-Time Eigenvalues')

mode_values = [1]
eigenvalue_tolerance = 0.001 
recorded_data = []
recorded_data_non_positive_definite = []


recorded_data_path = "/home/jamil/PyRep/projects/saved_models/recorded_data"
os.makedirs(recorded_data_path, exist_ok=True)

try:
    while True:
        # Joystick input
        joystick_handler.listen()
        axis_values = [joystick_handler.x, joystick_handler.y]
        mode = joystick_handler.mode

        print("mode value from joystick input", mode)

        if axis_values is not None:
            # predict velocities
            predicted_velocities = module.predict_velocities(obs.joint_positions, axis_values, mode)
            # covariance matrix and eigenvectors
            robot_state_torch = torch.tensor(obs.joint_positions, dtype=torch.float32).unsqueeze(0)
            covariance_matrix, eigenvectors = module.get_transformation(robot_state_torch)
            eigenvalues = torch.linalg.eigvalsh(covariance_matrix, UPLO='L') 

            # Check eigenval
            similar_eigenvalues = any(torch.abs(eig1 - eig2) < eigenvalue_tolerance for i, eig1 in enumerate(eigenvalues) for eig2 in eigenvalues[i + 1:])

            if similar_eigenvalues:
                # Record values
                recorded_joint_state = obs.joint_state
                recorded_covariance_matrix = covariance_matrix
                recorded_eigenvectors = eigenvectors

                 # Save recorded data 
                recorded_data.append({
                    'joint_state': recorded_joint_state,
                    'covariance_matrix': recorded_covariance_matrix,
                    'eigenvectors': recorded_eigenvectors
                })
                try:
                    chol_cov_matrix = torch.cholesky(covariance_matrix)
                except torch.linalg.LinAlgError:
                    # case : covariance matrix is not positive definite
                    print("Covariance matrix is not positive definite.")
                    print(covariance_matrix)
                    
                    #append to non-positive definite list
                    recorded_data_non_positive_definite.append({
                        'joint_state': obs.joint_state,
                        'covariance_matrix': covariance_matrix,
                        'eigenvectors': eigenvectors
                    })

            # color bars
            cax_cov.clear()
            cax_eig.clear()
            cax_eigenvalues.clear()

            # Clear plots
            axs[0].clear()
            axs[1].clear()
            axs[2].clear()
        
            heatmap_cov = sns.heatmap(covariance_matrix[0].detach().numpy(), annot=True, fmt=".4f", cmap='viridis', ax=axs[0], cbar_ax=cax_cov)
            axs[0].set_title('Real-Time Covariance Matrix')

            columns_to_plot = [mode * 2 % eigenvectors.shape[1], (mode * 2 + 1) % eigenvectors.shape[1]]
            print("columns to plot", columns_to_plot)
            data_eig = eigenvectors[:, columns_to_plot].detach().numpy()
            
            # Update heatmap eigenvectors
            heatmap_eigenvectors = sns.heatmap(data_eig, annot=True, fmt=".4f", cmap='viridis', ax=axs[1], cbar_ax=cax_eig)
            axs[1].set_title('Real-Time Eigenvectors')

            # Update heatmap eigenvalues
            heatmap_eigenvalues = sns.heatmap(eigenvalues.reshape(1, -1).detach().numpy(), annot=True, fmt=".4f", cmap='viridis', ax=axs[2], cbar_ax=cax_eigenvalues)
            axs[2].set_title('Real-Time Eigenvalues')
            
            # Draw
            plt.draw()
            plt.pause(0.000001)

            if joystick_handler.button_one_down:  # gripper->OPEN/CLOSE
                print("Closing gripper")
                env._scene.robot.gripper.actuate(0.0, velocity=0.2)
            else:
                print("Opening gripper")
                env._scene.robot.gripper.actuate(1.0, velocity=0.2)

        # Step environment
        obs, reward, _ = task.step(predicted_velocities.cpu().detach().numpy())

except KeyboardInterrupt:
    recorded_data_np = recorded_data.cpu().numpy()
    recorded_data_non_positive_definite_np = recorded_data_non_positive_definite.cpu().numpy()

    # Save NumPy arrays as .npy files
    save_path = os.path.join(recorded_data_path, 'recorded_data.npy')
    save_path_non_positive_definite = os.path.join(recorded_data_path, 'recorded_data_non_positive_definite.npy')

    np.save(save_path, recorded_data_np)
    np.save(save_path_non_positive_definite, recorded_data_non_positive_definite_np)

    # Close environment
    plt.close()
    env.shutdown()
