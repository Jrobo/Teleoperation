import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from manupulatortest import HThetaNetwork
from modules.joystick_handler import JoystickHandler
from modules.scl_module import SCLModule  # SCL module

if __name__ == "__main__":
    # Load models
    h_theta_net = HThetaNetwork()
    h_theta_net.load_state_dict(torch.load('/home/jamil/PyRep/projects/h_theta_net.pth'))
    h_theta_net.eval()
    print("Models loaded successfully.")
    # SCL module
    scl_module = SCLModule(h_theta_net)
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
        """   x_axis, y_axis, button_pressed = joystick_handler.listen()
        print(f"X-Axis: {x_axis}, Y-Axis: {y_axis}, Button Pressed: {button_pressed}")
        axis_values=[x_axis, y_axis] """
        x_axis, y_axis, button_pressed = joystick_handler.listen()
        print(f"X-Axis: {x_axis}, Y-Axis: {y_axis}, Button Pressed: {button_pressed}")
        # joystick input
        axis_values=[x_axis, y_axis]
        if axis_values is not None:
            # Use SCL module to predict velocities  
            predicted_velocities = scl_module.predict_velocities(obs.joint_positions, axis_values)
            print("Predicted Velocities:", predicted_velocities)   
            # button press--->OPEN/CLOSE
            if button_pressed:
                print("Closing gripper")
                env._scene.robot.gripper.actuate(0.0, velocity=0.2)  # Adjust the velocity as needed
            else:
                print("Opening gripper")
                env._scene.robot.gripper.actuate(1.0, velocity=0.2)  # Adjust the velocity as needed         
            # Step in the environment   
            obs, reward, terminate = task.step(predicted_velocities.cpu().detach().numpy())
except KeyboardInterrupt:
    env.shutdown()

