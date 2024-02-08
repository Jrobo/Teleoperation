import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from manupulatortest import HThetaNetwork
from modules.joystick_handler import JoystickHandler
from modules.scl_module import SCLModule  # Import your SCL module

if __name__ == "__main__":
    # Load models
    h_theta_net = HThetaNetwork()
    h_theta_net.load_state_dict(torch.load('/home/jamil/PyRep/examples/h_theta_net.pth'))
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
            joystick_input = joystick_handler.get_joystick_input()

            # Use SCL module to predict velocities
            predicted_velocities = scl_module.predict_velocities(obs.joint_positions, joystick_input)

            # Step in the environment
            obs, reward, terminate = task.step(predicted_velocities.cpu().detach().numpy())

    except KeyboardInterrupt:
        env.shutdown()
