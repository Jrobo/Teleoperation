# scl_module.py
import torch

class SCLModule:
    def __init__(self, h_theta_net):
        self.h_theta_net = h_theta_net

    def predict_velocities(self, manipulator_state, joystick_input):
        # Joint positions
        manipulator_state_tensor = torch.tensor(manipulator_state, dtype=torch.float32)

        # Transformation matrix "H"
        predicted_H = self.h_theta_net(manipulator_state_tensor)
        print("the predicted_H is", predicted_H)
        print("Shape of predicted_H:", predicted_H.shape)

        # Predicted velocities
        # Transpose or reshape joystick_input to have dimensions (2x1)
        joystick_input = torch.tensor(joystick_input, dtype=torch.float32).reshape(-1, 1)
        print(" joystick input",joystick_input)
        print("Joystick input shape",joystick_input.shape)
        predicted_velocities = torch.matmul(predicted_H, joystick_input)
        predicted_velocities = predicted_velocities.squeeze()
        print("the predicted velocity is", predicted_velocities)
        print("predicted_velocities.shape", predicted_velocities.shape)

        return predicted_velocities
