import numpy as np
import pygame
import math

from models.Action import Action
from models.Colors import Colors

KART_STARTING_POSITION = (140, 440)
KART_SIZE_PIXELS = (50, 30)
STEP_TIMEOUT = 15000


class RacingGame:
    """Environment wrapper for the Racing game."""

    def __init__(self, board_size_blocks: tuple[int, int]):
        pygame.init()
        pygame.font.init()
        self.text_font = pygame.font.SysFont("Comic Sans MS", 30)

        self.board_size_blocks = board_size_blocks
        # self.block_size_pixels = block_size_pixels
        self.is_display_initalized = False
        pygame.display.set_caption("Racing Game")
        self.clock = pygame.time.Clock()

        self.track_image = pygame.image.load("img/race_track_001.png")
        self.kart_image = pygame.image.load("img/kart.png")

        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.kart_position = KART_STARTING_POSITION

        # rotation in degrees
        # 0° = driving forward into the east direction
        self.kart_rotation = 270

        self.score = 0
        self.step_counter = 0
        return self.get_state()

    def step(self, action: Action):
        """Takes an action and updates the game state."""

        # action moves the steering wheel about 10°
        if action == Action.LEFT:
            # if self.kart_rotation >= 0.1:
            # self.kart_rotation -= (10 / 360)
            self.kart_rotation -= 2
        elif action == Action.RIGHT:
            # if self.kart_rotation <= 0.9:
            # self.kart_rotation += (1 / 360)
            self.kart_rotation += 2

        # stay in degree value range (0-360)
        if self.kart_rotation > 360:
            self.kart_rotation -= 360
        if self.kart_rotation < 0:
            self.kart_rotation += 360

        # calculate position change (1 step in depending rotation)
        # convert to radiants
        # theta = self.kart_rotation * 2 * math.pi
        rad = math.radians(self.kart_rotation)
        new_position = [
            self.kart_position[0] + 1 * math.cos(rad),
            self.kart_position[1] + 1 * math.sin(rad),
        ]
        self.kart_position = new_position

        # calculate if kart is in field

        # remove decimals
        x = int(self.kart_position[0])
        y = int(self.kart_position[1])

        # get track color of current position
        current_track_color = self.track_image.get_at((x, y))[:3]

        # stop game if out of track
        if current_track_color == Colors.GRASS.value:
            return self.get_state(), -10, True

        # stop game if finished
        if current_track_color == Colors.RED.value:

            # use remaining time as reward
            reward = STEP_TIMEOUT - self.step_counter

            return self.get_state, reward, True

        self.step_counter += 1

        # kart took too long to finish
        if self.step_counter > STEP_TIMEOUT:
            return self.get_state(), -50, True

        return self.get_state(), 1, False

    def get_state(self):
        """Gets the current state representation."""
        state = [
            self.kart_position[0],
            self.kart_position[1],
            self.kart_rotation,
            self.step_counter,  # Include step counter in the state
        ]

        return np.array(state, dtype=np.float32)

    def render(self):
        """Renders the game on the screen."""

        # inizialize screen on first render
        if not self.is_display_initalized:
            self.is_display_initalized = True
            self.display = pygame.display.set_mode(self.board_size_blocks)

        # draw racetrack
        self.display.blit(self.track_image, (0, 0))

        self.__draw_stats()
        self.__draw_kart()

        pygame.display.flip()
        self.clock.tick(60)

    def __draw_stats(self):
        text_surface = self.text_font.render(
            "Professional Racer", False, Colors.BLACK.value
        )
        self.display.blit(text_surface, (15, 15))

        text_surface = self.text_font.render(
            str(self.step_counter), False, Colors.BLACK.value
        )
        self.display.blit(text_surface, (500, 15))

    def __draw_kart(self):
        rect_surface = pygame.Surface(KART_SIZE_PIXELS, pygame.SRCALPHA)
        rect_surface.blit(pygame.transform.rotate(self.kart_image, 270), (0, 0))
        
        rotated_surface = pygame.transform.rotate(
            rect_surface, 360 - self.kart_rotation
        )
        rotated_rect = rotated_surface.get_rect(center=self.kart_position)

        self.display.blit(rotated_surface, rotated_rect.topleft)

        # draw the exact position of the kart (debugging)
        # pygame.draw.rect(
        #     self.display,
        #     Colors.RED.value,
        #     pygame.Rect(
        #         self.kart_position[0],
        #         self.kart_position[1],
        #         2,
        #         2,
        #     ),
        # )

    def handle_events(self) -> Action:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # self.close()
                exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            return Action.LEFT
        if keys[pygame.K_d]:
            return Action.RIGHT
        return Action.FORWARD
