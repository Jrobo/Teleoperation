import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ctypes import cdll

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from modules.scl import HThetaNetwork
from modules.joystick_handler import JoystickHandler
from modules.deep_ppca import DeepPPCA

# Initialize X11 threads
cdll.LoadLibrary('libX11.so.6').XInitThreads()

# Load pre-trained model
model_path = '/home/jamil/PyRep/projects/saved_models/ppca_model.pth'
module = DeepPPCA(sigma=0.0001)
module.load_state_dict(torch.load(model_path))
module.eval()

# RL environment setup
action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(action_mode)
env.launch()
task = env.get_task(WaterPlants)
descriptions, obs = task.reset()
joystick_handler = JoystickHandler()

# Parameters
mode_values = [1]
eigenvalue_tolerance = 0.001

recorded_data_path = "/home/jamil/PyRep/projects/dataset"
os.makedirs(recorded_data_path, exist_ok=True)
recorded_joint_data = []



try:
    while True:
        # Joystick input
        joystick_handler.listen()
        axis_values = [joystick_handler.x, joystick_handler.y]
        mode = joystick_handler.mode

        print("mode value from joystick input", mode)

        if axis_values is not None:
            # Predict velocities
            predicted_velocities = module.predict_velocities(obs.joint_positions, axis_values, mode)
            
            # Covariance matrix and eigenvectors
            robot_state_torch = torch.tensor(obs.joint_positions, dtype=torch.float32).unsqueeze(0)
            covariance_matrix, eigenvectors = module.get_transformation(robot_state_torch)

            try:
                # Attempt to invert the covariance matrix
                covariance_matrix_inv = np.linalg.inv(covariance_matrix.detach().numpy())
            except np.linalg.LinAlgError:
                # Handle the singularity exception
                print("Singular covariance matrix. Saving joint configuration.")
                recorded_joint_data.append(obs.joint_positions.copy())
            else:
                # Gripper control
                if joystick_handler.button_one_down:
                    print("Closing gripper")
                    env._scene.robot.gripper.actuate(0.0, velocity=0.2)
                else:
                    print("Opening gripper")
                    env._scene.robot.gripper.actuate(1.0, velocity=0.2)

                # Step environment
                obs, reward, _ = task.step(predicted_velocities.cpu().detach().numpy())

except KeyboardInterrupt:
    recorded_joint_data_np = np.array(recorded_joint_data)

    # Save NumPy array
    save_path = os.path.join(recorded_data_path, 'recorded_joint_data_np.npy')
    np.save(save_path, recorded_joint_data_np)

    # Close environment
    plt.close()
    env.shutdown()
