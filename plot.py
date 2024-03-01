""" import numpy as np
import matplotlib.pyplot as plt

# Load saved data
robot_state_data = np.load('dataset/robot_state_data.npy')
predicted_velocity_data = np.load('dataset/Predicted_Velocity.npy')

# Transpose the data
robot_state_data = np.transpose(robot_state_data)

# Create a time array for x-axis
time_steps = np.arange(122)
print(predicted_velocity_data.shape)
# Create subplots for joint velocity and state data
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot joint velocity data
for joint in range(7):
    axes[0].plot(time_steps, predicted_velocity_data[joint], label=f'Joint {joint+1} Predicted Velocity')


axes[0].set_title('Joint Velocity Data')
axes[0].set_ylabel('Velocity Values')
axes[0].legend()

# Plot robot state data
for joint in range(7):
    axes[1].plot(time_steps, robot_state_data[joint], linestyle='dashed', label=f'Joint {joint+1} State')

axes[1].set_title('Robot State Data')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('State Values')
axes[1].legend()

plt.tight_layout()
plt.show()
 """
import numpy as np

""" # Load the NumPy array
file_path = '/home/jamil/PyRep/projects/dataset/all_demos_joint_data.npy'
data = np.load(file_path)

# Print the shape and size
print(f"Shape of the array: {data.shape}")
print(f"Size of the array: {data.size}") """
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
fig = plt.figure()
def init():
    sns.heatmap(np.zeros((10, 10)), vmax=.8, square=True, cbar=False)

def animate(i):
    data = data_list[i]
    sns.heatmap(data, vmax=.8, square=True, cbar=False)

data_list = []
for j in range(200):
    data = np.random.rand(10, 10)
    data_list.append(data)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=20, repeat = False)

savefile = r"test3.gif"
pillowwriter = animation.PillowWriter(fps=20)
anim.save(savefile, writer=pillowwriter)

plt.show()