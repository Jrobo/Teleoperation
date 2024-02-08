import numpy as np
import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from manupulatortest import HThetaNetwork
from modules.joystick_handler import JoystickHandler
from modules.mode_switching import get_mode_from_button_value, get_joint_velocities

if __name__ == "__main__":
    # Load models
    h_theta_net = HThetaNetwork()
    h_theta_net.load_state_dict(torch.load('/home/jamil/PyRep/examples/h_theta_net.pth'))
    h_theta_net.eval()
    print("Models loaded successfully.")

    # RL environment
    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    env = Environment(action_mode)
    env.launch()
    task = env.get_task(WaterPlants)
    descriptions, obs = task.reset()

    joystick_handler = JoystickHandler()

    current_mode = 0  # Initial mode
    joy_val=[]
    try:
        while True:
            # Joystick input
            joystick_input = joystick_handler.get_joystick_input()
            print(joystick_input)
            # if len(values) == 2:
            #     joy_val[0] = values  
            #     joy_val[1]  = values  
            # elif len(values)  == 1:
            #     joy_val[2] = values  
            # Mode switch logic
            # mode_button_index = 0  # Adjust the button index based on your joystick configuration
            # mode_button_value = joystick_handler.get_button_value(mode_button_index)
            # Get joint velocities based on current mode
            predicted_velocities = get_joint_velocities(joystick_input)

            # Step in the environment
            obs, reward, terminate = task.step(predicted_velocities)

    except KeyboardInterrupt:
        env.shutdown()
