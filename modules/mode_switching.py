# examples/modules/mode_switching.py
from typing import List

def get_mode_from_button_value(mode_button_value: int) -> int:
    if mode_button_value == 0:
        return 0
    elif mode_button_value == 1:
        return 1
    elif mode_button_value == 2:
        return 2
    elif mode_button_value == 3:
        return 3
    else:
        return 0

def get_button_value(self, button_index):
    return self.joystick.get_button(button_index) 

def get_joint_velocities(x_axis: float, y_axis: float, current_mode: int) -> List[float]:
    joint_velocities = [0.0] * 7
    x_axis=joy_val[0]
    y_axis=joy_val[1]
    current_mode=joy_val[2]

    if current_mode == 0:
        joint_velocities[0] = y_axis
        joint_velocities[1] = x_axis
        print("mode 0 - Controlling joint 0 and 1")

    elif current_mode == 1:
        joint_velocities[2] = y_axis
        joint_velocities[3] = x_axis
        print("mode 1 - Controlling joint 2 and 3")
        
    elif current_mode == 2:
        joint_velocities[4] = y_axis
        joint_velocities[5] = x_axis
        print("mode 2 - Controlling joint 4 and 5")

    elif current_mode == 3:
        joint_velocities[5] = y_axis
        joint_velocities[6] = x_axis
        print("mode 3 - Controlling joint 5 and 6")

    return joint_velocities
