import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame(np.random.randn(100, 7), columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7'])
sns.pairplot(data)
plt.title('Pair Plot')
plt.show()
import pandas as pd
import seaborn as sns

# Load saved data
robot_state_data = np.load('dataset/robot_state_data.npy')
predicted_velocity_data = np.load('dataset/Predicted_Velocity.npy')

# Transpose the data
robot_state_data = np.transpose(predicted_velocity_data)
# Assuming data is your multivariate Gaussian data with 7 columns
data = pd.DataFrame(robot_state_data, columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7'])
sns.pairplot(data)
plt.title('Scatterplot Matrix')
plt.show()

 

""" import numpy as np
import matplotlib.pyplot as plt

# Assuming your 7 by 7 matrix with random numbers
random_matrix = np.random.rand(7, 7)

# Create subplots for columns 1 and 2, 3 and 4, 5 and 6, 6 and 7
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Subplot for columns 1 and 2
axes[0, 0].imshow(random_matrix[:, 1:3], cmap='viridis', interpolation='nearest')
axes[0, 0].set_title('Columns 1 and 2')

# Subplot for columns 3 and 4
axes[0, 1].imshow(random_matrix[:, 3:5], cmap='viridis', interpolation='nearest')
axes[0, 1].set_title('Columns 3 and 4')

# Subplot for columns 5 and 6
axes[1, 0].imshow(random_matrix[:, 5:7], cmap='viridis', interpolation='nearest')
axes[1, 0].set_title('Columns 5 and 6')

# Subplot for columns 6 and 7
axes[1, 1].imshow(random_matrix[:, 6:], cmap='viridis', interpolation='nearest')
axes[1, 1].set_title('Columns 6 and 7')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show() """
