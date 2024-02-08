import numpy as np
import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from manupulatortest import HThetaNetwork, FOmegaNetwork
from modules.joystick_handler import JoystickHandler

if __name__ == "__main__":
    # Load models
    h_theta_net = HThetaNetwork()
    f_omega_net = FOmegaNetwork()
    h_theta_net.load_state_dict(torch.load('/home/jamil/PyRep/examples/h_theta_net.pth'))
    f_omega_net.load_state_dict(torch.load('/home/jamil/PyRep/examples/f_omega_net.pth'))
    h_theta_net.eval()
    f_omega_net.eval()
    print("Models loaded successfully.")

    # RL environment
    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    env = Environment(action_mode)
    env.launch()
    task = env.get_task(WaterPlants)
    descriptions, obs = task.reset()
    
    joystick_handler = JoystickHandler()

    try:
        while True:
            # Joystick input
            joystick_input = joystick_handler.get_joystick_input()
            joystick_input = torch.tensor(joystick_input, dtype=torch.float32)

            # Joint positions
            manipulator_state = torch.tensor(obs.joint_positions, dtype=torch.float32)

            # Transformation matrix "H"
            predicted_H = h_theta_net(manipulator_state)
            print("the predicted_H is", predicted_H)

            # Predicted velocities
            predicted_velocities = torch.matmul(predicted_H, joystick_input)
            predicted_velocities = predicted_velocities.squeeze()
            print("the predicted velocity is", predicted_velocities)
            print("predicted_velocities.shape", predicted_velocities.shape)

            # Step in the environment
            obs, reward, terminate = task.step(predicted_velocities.cpu().detach().numpy())

    except KeyboardInterrupt:
        env.shutdown()
