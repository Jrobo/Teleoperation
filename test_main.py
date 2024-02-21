import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from modules.scl import HThetaNetwork
from modules.joystick_handler import JoystickHandler
#from modules.scl_module import SCLModule  # SCL module
from modules.deep_ppca import DeepPPCA
# from modules.mode_switching import ModeSwitching

technique = 'deep_ppca'
# Load models
if technique=='scl':
    module = HThetaNetwork()
    module.load_state_dict(torch.load('/home/jamil/PyRep/projects/saved_models/h_theta_net.pth'))
elif technique=='deep_ppca':
    module = DeepPPCA()
    module.load_state_dict(torch.load('/home/jamil/PyRep/projects/saved_models/ppca_model.pth'))
elif technique=='mode_switching':
    module = ModeSwitching()
else:
    print("technique does not exists")
    exit()

print("Models loaded successfully.")
module.eval()

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
        # x_axis, y_axis, button_pressed = joystick_handler.listen()
        # print(f"X-Axis: {x_axis}, Y-Axis: {y_axis}, Button Pressed: {button_pressed}")
        # # joystick input
        predicted_velocities=torch.empty(size=(0,))
        joystick_handler.listen()
        axis_values=[joystick_handler.x, joystick_handler.y]
        if axis_values is not None:
            # Use SCL module to predict velocities   
            # button press--->OPEN/CLOSE
            print("current_mode",joystick_handler.mode)
        if joystick_handler.button_one_down:
            if joystick_handler.mode == 0:
                print("Mode 0: Closing gripper")
                env._scene.robot.gripper.actuate(0.0, velocity=0.2)
            elif joystick_handler.mode in [1, 2, 3]:
                    print(f"Mode {joystick_handler.mode}: transformation")
                    # use the velocity transformation by changing the mode
                    predicted_velocities = module.predict_velocities(obs.joint_positions, axis_values, mode=joystick_handler.mode)
                    print("Predicted Velocities:", predicted_velocities)                     
                    #transformation = module.get_mode_transformation(data, mode=joystick_handler.mode)
                    #print("transformation",transformation)                   
            else:
                print("Opening gripper")
                env._scene.robot.gripper.actuate(1.0, velocity=0.2) 

    # Step in the environment   
    obs, reward, _ = task.step(predicted_velocities.cpu().detach().numpy())
except KeyboardInterrupt:
    env.shutdown()

