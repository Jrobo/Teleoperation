import pygame
from typing import Tuple, List

class JoystickHandler:

    def __init__(self, joystick_index=0):
        pygame.init()
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(joystick_index)
        self.joystick.init()
        self.x = 0.0
        self.y = 0.0
        self.button_one_down = False    # button for opening/closing the gripper
        self.mode = 0                   # which mode is selected by the joystick

    def listen(self):

        for event in pygame.event.get():
            mode = self.mode
            if event.type == pygame.JOYAXISMOTION:
                self.x = self.joystick.get_axis(0)
                self.y = self.joystick.get_axis(1)

            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.button_one_down = True
                # Check if the button pressed is within the specified range
                if event.button == 2:
                    mode -= 1
                if event.button == 3:
                    mode += 1

                #self.mode = mode % 4
                mode = mode if mode >= 0 else 0
                self.mode = mode if mode <= 3 else 3

            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 0:
                    self.button_one_down = False

     

