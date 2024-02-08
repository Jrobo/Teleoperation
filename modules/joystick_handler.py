import pygame
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
                self.button_down = False