import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from modules.scl import HThetaNetwork
from modules.joystick_handler import JoystickHandler
from modules.deep_ppca import DeepPPCA
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym
import numpy as np
from gym.envs.registration import register
import gym_gazebo2.envs

# Register the OpenAI Gazebo environment
register(
    id='GazeboEnv-v0',
    entry_point='gym_gazebo2.envs:GazeboEnv',
)

# Create the environment
env = gym.make('GazeboEnv-v0')

# Reset the environment
obs = env.reset()

sigma=0.1
technique = 'deep_ppca'
# Load models
if technique=='scl':
    module = HThetaNetwork()
    module.load_state_dict(torch.load('/home/jamil/PyRep/projects/saved_models/h_theta_net.pth'))
elif technique=='deep_ppca':
    module = DeepPPCA(sigma)
    module.load_state_dict(torch.load('/home/jamil/PyRep/projects/saved_models/ppca_model_0.001_eph_30000_lr_0.0001_2024-04-12_02-18.pth'))
elif technique=='mode_switching':
    module = ModeSwitching()
else:
    print("technique does not exists")
    exit()

print("Models loaded successfully.")
module.eval()

# # RL environment
# action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
# env = Environment(action_mode)
# env.launch()
# task = env.get_task(WaterPlants)
# descriptions, obs = task.reset()
joystick_handler = JoystickHandler()

robot_state_data = []
# predicted_velocities_data = []
predicted_velocity_data = torch.Tensor()

#module.start_animation()
try:
    while True:
        # Joystick input
        joystick_handler.listen()
        axis_values=[joystick_handler.x, joystick_handler.y]
        #axis_values=[1,0]
        mode=joystick_handler.mode
        #print("Datatype of obs.joint state",type(obs.joint_positions))

         #if axis_values is not None:
        # Use SCL module to predict velocities  
            # Append new data to lists
        
        #print("robot  state data",robot_state_data.shape)
        print(axis_values)
        predicted_velocities = module.predict_velocities(obs.joint_positions, axis_values, mode) 
        #module.update_animation()
        #robot_state_data.append(robot_state.tolist())  
        #velocities_data = predicted_velocities.detach().numpy().tolist()
        

        #if technique == 'deep_ppca':
        #module.run_animation(obs.joint_positions)                  
        #plt.show()
            # Convert numpy array to list before appending          
        #print(predicted_velocity_data)            # gripper->OPEN/CLOSE
        if joystick_handler.button_one_down: 
            env._scene.robot.gripper.actuate(0.0, velocity=0.2) 
        else:
            env._scene.robot.gripper.actuate(1.0, velocity=0.2)    

        # Step in the environment
        obs, reward, done, info = env.step(predicted_velocities.cpu().detach().numpy())  #done, info
        if done:
            obs = env.reset()  # Reset the environment if episode is done
except KeyboardInterrupt:
    env.close()
        
     