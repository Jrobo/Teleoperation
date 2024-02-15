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
        self.button_one_down = False
        self.mode = 0

    def listen(self):# -> Tuple[float, float, bool, int]:

        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.x = self.joystick.get_axis(0)
                self.y = self.joystick.get_axis(1)

            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.button_one_down = True
                # Check if the button pressed is within the specified range
                if event.button == 2:
                    self.mode -= 1
                if event.button == 3:
                    self.mode += 1

                self.mode = self.mode if self.mode >= 0 else 0
                self.mode = self.mode if self.mode < 4 else 3

            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 0:
                    self.button_one_down = False

        # return self.x, self.y, self.button_one_down, current_mode


'''import pygame
from typing import List

class JoystickHandler:

    def __init__(self, joystick_index=0):
        pygame.init()
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(joystick_index)
        self.joystick.init()
        self.x = 0
        self.y = 0
        self.button_down = False


    def listen(self) -> List[float]:

        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.x = self.joystick.get_axis(0)
                self.y = self.joystick.get_axis(1)

            elif event.type == pygame.JOYBUTTONDOWN:
                self.button_down = True

            elif event.type == pygame.JOYBUTTONUP:
                self.button_down = False'''
