import numpy as np
import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
#from manupulatortest import load_models
from manupulatortest import HThetaNetwork, FOmegaNetwork
import pygame
from typing import List
import time
from modules.rl_module import RLModule
from modules.joystick_handler import JoystickHandler



# # Calculate joint velocities
# def get_predicted_velocities(manipulator_state: List[float] -> torch.Tensor,
#                               joystick_input: List[float] -> torch.Tensor:
#                               from typing import List

# def get_predicted_velocities(manipulator_state: List[float], joystick_input: List[float]) -> torch.Tensor:
    
#     print("Size of manipulator_state before unsqueeze:", manipulator_state.size())
#     print("Size of joystick_input before unsqueeze:", joystick_input.size())

#     predicted_H = h_theta_net(manipulator_state)

#     print('The predicted H is ',predicted_H)

#     inferred_a = f_omega_net(manipulator_state,joystick_input)
    
#     print('Inferred a is ',inferred_a)

#     predicted_velocities = torch.matmul(predicted_H, inferred_a)

#     print('predicted_velocities', predicted_velocities)
    
#     return predicted_velocities 
# def get_predicted_velocities(manipulator_state: torch.Tensor, joystick_input: torch.Tensor) -> torch.Tensor:
#     # Assuming manipulator_state and joystick_input are 1-dimensional tensors
#     manipulator_state = manipulator_state.unsqueeze(0)
#     joystick_input = joystick_input.unsqueeze(0)

#     # Print sizes before concatenation
#     print("Size of manipulator_state before unsqueeze:", manipulator_state.size())
#     print("Size of joystick_input before unsqueeze:", joystick_input.size())

#     # Concatenate along dimension 1
#     concatenated_input = torch.cat((manipulator_state, joystick_input), dim=1)

#     # Print sizes after concatenation
#     print("Size of concatenated_input:", concatenated_input.size())
    
#     # Pass the concatenated input to the network
#     inferred_a = f_omega_net(*torch.unbind(concatenated_input, dim=1))
#     print('Inferred a is ', inferred_a)

#     return inferred_a
#first code
""" def get_predicted_velocities(joint_positions: np.ndarray, 
                              joystick_input: List[float],
                              h_theta_net: HThetaNetwork,
                              f_omega_net: FOmegaNetwork) -> torch.Tensor:
    predicted_H = h_theta_net(torch.tensor(joint_positions, dtype=torch.float32))
    inferred_a = f_omega_net(torch.tensor(joint_positions, dtype=torch.float32), 
                             torch.tensor(joystick_input, dtype=torch.float32))
    predicted_velocities = torch.matmul(predicted_H, inferred_a.unsqueeze(-1)).squeeze(-1)
    return predicted_velocities """

""" pygame.init()
pygame.joystick.init()
num_joysticks = pygame.joystick.get_count()

if num_joysticks == 0:
    raise RuntimeError("No joystick found!")
joystick = pygame.joystick.Joystick(0)
time.sleep(2.)

# Real-time joystick input
def get_joystick_input(joystick: pygame.joystick.Joystick) -> List[float]:
    joystick.init()
    try:
        while True:
            for event in [pygame.event.wait(), ] + pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    x_axis = joystick.get_axis(0)
                    y_axis = joystick.get_axis(1)
                    joystick_input = [x_axis, y_axis]
                    return joystick_input
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit() """

""" # RL environment
action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(action_mode)
env.launch()
task = env.get_task(WaterPlants)
descriptions, obs = task.reset()
print("RL environment setup")

# Joystick initialization
pygame.init()
pygame.joystick.init()
num_joysticks = pygame.joystick.get_count()
print("pygame initialized")
if num_joysticks == 0:
    raise RuntimeError("No joystick found!")

joystick = pygame.joystick.Joystick(0)
time.sleep(2.)

try:
    while True:
        # Joystick input
        joystick_input = get_joystick_input(joystick)
        #joystick_input = list(zip(*joystick_input))
 """
if __name__ == "__main__":
    # Load models
    #h_theta_net, f_omega_net = load_models()

    # Initialize
    h_theta_net = HThetaNetwork()
    f_omega_net = FOmegaNetwork()
    print('h_theta_net and f_omega_net initialized')      

    # Load
    h_theta_net.load_state_dict(torch.load('/home/jamil/PyRep/examples/h_theta_net.pth'))
    f_omega_net.load_state_dict(torch.load('/home/jamil/PyRep/examples/f_omega_net.pth'))
    print('h_theta_net and f_omega_net loaded')

    # Evaluation
    h_theta_net.eval()
    f_omega_net.eval()
    print("Models loaded successfully.")
   # h_theta_net, f_omega_net = load_models()
    rl_module = RLModule()
    print('RL module instantiated')
    joystick_handler = JoystickHandler()
    
    try:
        task, _, obs = rl_module.reset_task(WaterPlants)
        while True:
            joystick_input = joystick_handler.get_joystick_input()
            print('Joystick values:', joystick_input)
            print('Shape of Joystick values:', joystick_input.action_shape)
            joystick_input=torch.tensor(joystick_input, dtype=torch.float32)
            print(type(joystick_input))

            # Joint positions
            # NumPy array to tensor
            manipulator_state = torch.tensor(obs.joint_positions, dtype=torch.float32)
            print('Joint Positions:', manipulator_state)
            print(type(manipulator_state))

            # Transformation matrix 
            predicted_H = h_theta_net(manipulator_state)
            
            print("the predicted_H is",predicted_H)
            print("Shape of the predicted_H is",predicted_H.shape)
            predicted_velocities = torch.matmul(predicted_H, joystick_input)
            predicted_velocities = predicted_velocities.squeeze()
            print("the predicted velocity is",predicted_velocities)
            print("predicted_velocities.shape",predicted_velocities.shape)
            # Step in the environment
            obs, reward, terminate = task.step(predicted_velocities.cpu().detach().numpy())

    except KeyboardInterrupt:
            rl_module.environment.shutdown()
""" except KeyboardInterrupt:
    env.shutdown() """
