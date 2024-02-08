import pygame
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
import time
#The following progran I just control the mode
#mode1: joint[0,1]
#mode 2: joint[3,4]
#mode3:joint{5,6}
#mode4:joint[6,7]
def get_mode_from_button_value(mode_button_value):
    
    if mode_button_value == 0:
        return 0
    elif mode_button_value == 1:
        return 1
    elif mode_button_value == 2:
        return 2
    elif mode_button_value == 3:
        return 3
    elif mode_button_value == 4:
        return 4
    else:
        return 0


def get_joint_velocities(x_axis, y_axis, mode_button, current_mode):
    
    # implementation b
    joint_velocities = [0.0] * 8

    if current_mode == 0:
        # Mode 1
        joint_velocities[0] = y_axis
        joint_velocities[1] = x_axis
        print("mode------>0---Controlling joint 0 and 1 ")

    elif current_mode == 1:
        # Mode 2
        joint_velocities[2] = y_axis
        joint_velocities[3] = x_axis
        print("mode------>1---Controlling joint  2 an 3")
        # ... (other joint velocities)

    elif current_mode == 2:
        # Mode 3
        joint_velocities[4] = y_axis
        joint_velocities[5] = x_axis
        print("mode------>2---contolling joint 4 and 5")
  
    elif current_mode == 3:
        # Mode 4
        joint_velocities[5] = y_axis
        joint_velocities[6] = x_axis
        print("mode------>3---controlling joint 5 and 6")

    elif current_mode == 4:
        # Mode 4
        joint_velocities[5] = y_axis
        joint_velocities[6] = x_axis
        print("mode------>4---CURENTLY controlling joint 5 and 6 WOULD BE USED FOR GRIPPER MOVEMENT")
        #         if y_axis > 0:  # Assuming positive y_axis value means closing the gripper
        #     gripper_action = 1.0  
        #     print("Gripper mode - Closing gripper")
        # else:
        #     gripper_action = -1.0  
        #     print("Gripper mode - Opening gripper")
        # joint_velocities[-1] = gripper_action
    else: 
        joint_velocities = [0.0] * 7   

    return joint_velocities

pygame.init()
pygame.joystick.init()
num_joysticks = pygame.joystick.get_count()

if num_joysticks == 0:
    raise RuntimeError("No joystick found!")

joystick = pygame.joystick.Joystick(0)
time.sleep(2.)

# RL environment 
action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(action_mode)
env.launch()
task = env.get_task(WaterPlants)
descriptions, obs = task.reset()

x_axis = 0.0
y_axis = 0.0
mode_button_value = 0
current_mode = 0

try:
    while True:
        # joystick input
        for event in [pygame.event.wait(),] + pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                x_axis = joystick.get_axis(0)
                y_axis = joystick.get_axis(1)
            elif event.type == pygame.JOYBUTTONDOWN:
                
                if event.button in [0, 1, 2, 3,4]:
                    current_mode = get_mode_from_button_value(event.button)

        
        joint_velocities = get_joint_velocities(x_axis, y_axis, mode_button_value, current_mode)
        predicted_velocities = np.array(joint_velocities)
        print("predicted velocity", predicted_velocities)

        # Step
        obs, reward, terminate = task.step(predicted_velocities)

except KeyboardInterrupt:
    env.shutdown()
