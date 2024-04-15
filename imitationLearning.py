import os
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import WaterPlants

class ImitationLearning(object):

    def predict_action(self, batch):
        return np.random.uniform(size=(len(batch), 7))

    def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
        return 1
    
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=obs_config,  
    headless=False)
env.launch()

task = env.get_task(WaterPlants)

il = ImitationLearning()

demos = task.get_demos(20, live_demos=live_demos)  

flattened_demos = []
demos_joint_velocities = []
demos_joint_positions = []
demo_identifiers = []

for demo_idx, sublist in enumerate(demos):
    for obs in sublist:
        flattened_demos.append(obs)
        demos_joint_velocities.append(obs.joint_velocities.tolist())
        demos_joint_positions.append(obs.joint_positions.tolist())
        demo_identifiers.append(demo_idx)  # Add identifier for each demonstration

print("Length of the data in demos is", len(flattened_demos))

# lists to numpy arrays
demos_joint_velocities = np.array(demos_joint_velocities)
demos_joint_positions = np.array(demos_joint_positions)
demo_identifiers = np.array(demo_identifiers)

# Save 
current_directory = os.getcwd()
npy_filename = 'dataset/joint_data_with_identifiers.npz'
npy_filepath = os.path.join(current_directory, npy_filename)

np.savez(npy_filepath, 
         joint_velocities=demos_joint_velocities,
         joint_positions=demos_joint_positions,
         identifiers=demo_identifiers)

print(f"Joint data with identifiers saved at: {npy_filepath}")

# Training 
for i in range(100):
    print("'Training' iteration %d" % i)

    # batch size
    batch_size = 32

    # batch indices
    batch_indices = np.random.choice(len(flattened_demos), size=batch_size, replace=False)

    # Retrieve batch data
    batch_joint_velocities = demos_joint_velocities[batch_indices]
    batch_joint_positions = demos_joint_positions[batch_indices]
    batch_identifiers = demo_identifiers[batch_indices]

    # Predict actions
    predicted_actions = il.predict_action(batch_joint_velocities)

    # loss
    loss = il.behaviour_cloning_loss(batch_joint_velocities, predicted_actions)

print('Done')
env.shutdown()
