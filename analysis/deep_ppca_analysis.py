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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctypes import cdll
cdll.LoadLibrary('libX11.so.6').XInitThreads()


module = DeepPPCA()
module.load_state_dict(torch.load('/home/jamil/PyRep/projects/saved_models/ppca_model.pth'))
#print("Models loaded successfully.")
module.eval()
# RL environment
action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(action_mode)
env.launch()
task = env.get_task(WaterPlants)
descriptions, obs = task.reset()
joystick_handler = JoystickHandler()
robot_state_torch = torch.tensor(obs.joint_positions, dtype=torch.float32).unsqueeze(0)
covariance_matrix, eigenvectors = module.get_transformation(robot_state_torch)
# print("covariance_matrixc",covariance_matrix)
#robot_state_data = []
#predicted_velocities_data = []
#predicted_velocity_data = torch.Tensor()
# Assuming eigenvectors is a PyTorch tensor
#eigenvectors_torch = eigenvectors.clone().detach().requires_grad_(True).unsqueeze(0)
mode_values = [1]
# cov_matrix = torch.randn((7, 7))
# eigenvectors = torch.randn((7, 7))
print("Covarience Matrix",covariance_matrix)
print("Eigenvector",eigenvectors)
fig=module.create_and_save_heatmap_animation(covariance_matrix,eigenvectors,mode_values) #print("Covarience matrix shape",cov_matrix.shape)# Convert numpy array to list before appending          
            

try:
    while True:
        # Joystick input
        joystick_handler.listen()
        axis_values=[joystick_handler.x, joystick_handler.y]
        #axis_values=[1,0]
        mode=joystick_handler.mode
        mode_values[0] = mode
        if mode_values:
            current_mode = mode_values[0]
            print("Current Mode:", current_mode)
        plt.pause(0.1) 
        #print("Datatype of obs.joint state",type(obs.joint_positions))
        if axis_values is not None:
            # Use SCL module to predict velocities  
             # Append new data to lists           
            #print("robot  state data",robot_state_data.shape)
            predicted_velocities = module.predict_velocities(obs.joint_positions, axis_values, mode) 
            #module.get_transformation()
            #robot_state_data.append(robot_state.tolist())  
            #velocities_data = predicted_velocities.detach().numpy().tolist()
            #if technique == 'deep_ppca':
            #module.run_animation(obs.joint_positions)                  
            #predicted_velocities_data = predicted_velocities.detach().numpy().tolist()
            
            if joystick_handler.button_one_down:# gripper->OPEN/CLOSE
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