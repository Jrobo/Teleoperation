import numpy as np
import torch 
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget, WaterPlants
import torch
from manupulatortest import HThetaNetwork, FOmegaNetwork  
#from module.rl_module import
import matplotlib.pyplot as plt
import pygame
import time
from typing import List, Tuple
from modules.joystick_handler import JoystickHandler
# from rlbench.action_modes.action_mode import ActionMode

#Load from manupulatortest.py file where I done the trainining
# Initialize
h_theta_net = HThetaNetwork()
f_omega_net = FOmegaNetwork()
                    
# Load
h_theta_net.load_state_dict(torch.load('/home/jamil/PyRep/examples/h_theta_net.pth'))
f_omega_net.load_state_dict(torch.load('/home/jamil/PyRep/examples/f_omega_net.pth'))

# evaluation
h_theta_net.eval()
f_omega_net.eval()
print("Model loaded successfully.")

# calculate joint velocities
def get_predicted_velocities(joint_positions: np.ndarray, 
                              joystick_input: List[float],
                              h_theta_net: HThetaNetwork,
                              f_omega_net: FOmegaNetwork) -> torch.Tensor:
    predicted_H = h_theta_net(torch.tensor(joint_positions, dtype=torch.float32))
    inferred_a = f_omega_net(torch.tensor(joint_positions, dtype=torch.float32), 
                             torch.tensor(joystick_input, dtype=torch.float32))
    predicted_velocities = torch.matmul(predicted_H, inferred_a.unsqueeze(-1)).squeeze(-1)
    return predicted_velocities


pygame.init()
pygame.joystick.init()
num_joysticks = pygame.joystick.get_count()

if num_joysticks == 0:
    raise RuntimeError("No joystick found!")

joystick = pygame.joystick.Joystick(0)
time.sleep(2.)

def get_joystick_input(joystick: pygame.joystick.Joystick) -> List[float]:
    joystick.init()
    try:
        while True:
            for event in [pygame.event.wait(),] + pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    x_axis = joystick.get_axis(0)
                    y_axis = joystick.get_axis(1)
                    joystick_input = [x_axis, y_axis]
                    return joystick_input
    except KeyboardInterrupt:
        pass

    pygame.quit()


# RL environment
action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())


# action_mode = MoveArmThenGripper(
# arm_action_mode=JointVelocity(),
#  gripper_action_mode=Discrete()
# )
env = Environment(action_mode)

env.launch()

task = env.get_task(WaterPlants)
descriptions, obs = task.reset()     
# action_mode = ArmJointVelocityActionMode()
# env = Environment(action_mode)
# task = env.get_task(ReachTarget)
# descriptions, obs = task.reset()

#while True:
#    print(get_joystick_input(joystick))
                        
# RLBenh program
try:
    while True:
        # Joy input
        joystick_input = get_joystick_input(joystick)
        # joystick_input = [0., 0.]
        joystick_handler = JoystickHandler()
        joystick_handler.listen()
        joystick_input = joystick_handler.x, joystick_handler.y
        print('joystick values',joystick_input)
        print('joystick values:', joystick_input)
        print('Gripper Closed?', joystick_handler.button_down)
        # print('Gripper Opened:', gripper_opened

        print("joint Position",type(obs.joint_positions))
        print("obs.joint_positions",obs.joint_positions)
        # print(ee_action)
        #  NumPy array to tensor
        joint_positions_tensor = torch.tensor(obs.joint_positions, dtype=torch.float32)

        # Transformation matrix 
        predicted_H = h_theta_net(joint_positions_tensor)
        # joystick input to tensor
        joystick_input_tensor = torch.tensor(joystick_input, dtype=torch.float32)
        print('Shape of joystick values',joystick_input_tensor.shape)
        print("the predicted_H is",predicted_H)
        print("the predicted_H is",predicted_H.shape)
        predicted_velocities = torch.matmul(predicted_H, joystick_input_tensor)
        predicted_velocities = predicted_velocities.squeeze()
        print("the predicted velocity is",predicted_velocities)
        print("predicted_velocities.shape",predicted_velocities.shape)

        # taking the action
        #action = np.array([joystick_input[0], joystick_input[1]] + [0.] * 5)
        obs, reward, terminate = task.step(predicted_velocities.cpu().detach().numpy())
        #obs, reward, terminate = task.step(action)

except KeyboardInterrupt:
    env.shutdown()
