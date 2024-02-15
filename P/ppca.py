import numpy as np
from numpy.linalg import svd
import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import WaterPlants
from manupulatortest import HThetaNetwork
from modules.joystick_handler import JoystickHandler
from rlbench.action_modes.arm_action_modes import JointVelocity
class PPCAModule:
    def __init__(self, n_components):
        self.n_components = n_components
        self.W = None
        self.mu = None
        self.sigma = None

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        X_centered = X - self.mu

        _, S, Vt = svd(X_centered.T @ X_centered / X.shape[0])

        self.W = Vt[:self.n_components, :].T
        self.sigma = np.sqrt(np.maximum(0, np.sum(S[self.n_components:]) / (X.shape[0] - self.n_components)))

    def transform(self, A):
        Y = A @ self.W
        if self.sigma > 0:
            noise = np.random.normal(0, self.sigma, Y.shape)  # Generate Gaussian noise
            Y += noise  # Add Gaussian noise
        return Y + self.mu

if __name__ == "__main__":
    # Load the joystick handler
    joystick_handler = JoystickHandler()
    h_theta_net = HThetaNetwork()
    h_theta_net.load_state_dict(torch.load('/home/jamil/PyRep/projects/h_theta_net.pth'))
    h_theta_net.eval()
    print("Models loaded successfully.")
  # Load the RL environment
    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    env = Environment(action_mode)
    env.launch()
    task = env.get_task(WaterPlants)
    _, obs = task.reset()

    # Create an instance of PPCAModule
    ppca_module = PPCAModule(n_components=7)

    try:
        while True:
            # Joystick input
            x_axis, y_axis, button_pressed = joystick_handler.listen()
            print(f"X-Axis: {x_axis}, Y-Axis: {y_axis}, Button Pressed: {button_pressed}")

            # Joystick input
            axis_values = np.array([x_axis, y_axis]).reshape(1, -1)

            if axis_values is not None:
                # Fit the model
                ppca_module.fit(obs.joint_velocities.reshape(-1, 1))

                # Check if fitting was successful
                if ppca_module.W is not None:
                    # Use PPCA module to predict velocities
                    predicted_velocities = ppca_module.transform(axis_values)
                    print("Predicted Velocities:", predicted_velocities)
                else:
                    print("Fitting PPCA model failed. Check the fit method for issues.")

                # Gripper control based on button press
                if button_pressed:
                    print("Closing gripper")
                    env._scene.robot.gripper.actuate(0.0, velocity=0.2)  # Adjust the velocity as needed
                else:
                    print("Opening gripper")
                    env._scene.robot.gripper.actuate(1.0, velocity=0.2)  # Adjust the velocity as needed

                # Step in the environment
                _, obs, _, _ = task.step(predicted_velocities)

    except KeyboardInterrupt:
        env.shutdown()
