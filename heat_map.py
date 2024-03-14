import torch
import sys
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

module = DeepPPCA()
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
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

# Cov Matrix bplot
heatmap_cov = sns.heatmap(np.zeros((7, 7)), annot=True, fmt=".2f", cmap='viridis', ax=axs[0])
axs[0].set_title('Real-Time Covariance Matrix')

# Eigenvect plot
heatmap_eigenvectors = sns.heatmap(np.zeros((7, 2)), annot=True, fmt=".2f", cmap='viridis', ax=axs[1])
axs[1].set_title('Real-Time Eigenvectors')

mode_values = [1]

try:
    while True:
        # Joystick input
        joystick_handler.listen()
        axis_values = [joystick_handler.x, joystick_handler.y]
        mode = joystick_handler.mode

        print("mode value from joystick input", mode)

        if axis_values is not None:
            # Use SCL module to predict velocities
            predicted_velocities = module.predict_velocities(obs.joint_positions, axis_values, mode)
            # Update covariance matrix and eigenvectors
            robot_state_torch = torch.tensor(obs.joint_positions, dtype=torch.float32).unsqueeze(0)
            covariance_matrix, eigenvectors = module.get_transformation(robot_state_torch)

            # Clear previous plots
            axs[0].cla()
            axs[1].cla()

            # Update seaborn heatmaps
            sns.heatmap(covariance_matrix[0].detach().numpy(), annot=True, fmt=".2f", cmap='viridis', ax=axs[0])
            axs[0].set_title('Real-Time Covariance Matrix')

            # sns.heatmap(eigenvectors.detach().numpy(), annot=True, fmt=".2f", cmap='viridis', ax=axs[1])
            # axs[1].set_title('Real-Time Eigenvectors')

            # Extract and update eigenvectors for the specified mode
            columns_to_plot = [mode * 2 % eigenvectors.shape[1], (mode * 2 + 1) % eigenvectors.shape[1]]
            print("columns to plot", columns_to_plot)
            # data_eig = eigenvectors[:, columns_to_plot].detach().numpy()

            # sns.heatmap(data_eig, annot=True, fmt=".2f", cmap='viridis', ax=axs[1])

         
            # Draw
            plt.draw()
            plt.pause(0.01)

            if joystick_handler.button_one_down:  # gripper->OPEN/CLOSE
                print("Closing gripper")
                env._scene.robot.gripper.actuate(0.0, velocity=0.2)
            else:
                print("Opening gripper")
                env._scene.robot.gripper.actuate(1.0, velocity=0.2)

        # Step in the environment
        obs, reward, _ = task.step(predicted_velocities.cpu().detach().numpy())

except KeyboardInterrupt:
    plt.close()
    env.shutdown()
