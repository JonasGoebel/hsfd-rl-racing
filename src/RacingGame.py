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
        self.is_display_initalized = False
        pygame.display.set_caption("Racing Game")
        self.clock = pygame.time.Clock()

        self.track_image = pygame.image.load("src/img/race_track_001_legacy.png")
        self.kart_image = pygame.image.load("src/img/kart.png")

        # Checkpoints
        self.checkpoints = [(160, 260), (400, 250), (510, 400), (770, 285), (870, 600), (675, 665), (640, 840), (160, 830), (145, 500)]
        self.checkpoint_counter = 0
    

    def reset(self) -> np.array:
        """ Resets the game to starting position.
        Returns: Game state
        """
        self.kart_position = KART_STARTING_POSITION

        # rotation in degrees
        # 0° = driving forward into the east direction
        self.kart_rotation = 270

        self.score = 0
        self.step_counter = 0
        self.checkpoint_counter = 0

        for (x, y) in self.checkpoints:
            pygame.draw.circle(self.track_image, Colors.BLACK.value, (int(x), int(y)), 5)  # Blue for world coords
        pygame.draw.circle(self.track_image, Colors.BLUE.value, (int(self.checkpoints[0][0]), int(self.checkpoints[0][1])), 5)  # Blue for world coords
        

        return self.get_state()

    # returns (next_state, dist_to_next_checkpoint, is_on_track)
    def step(self, action: np.array) -> tuple[np.array, bool, bool]:
        """Takes an action and updates the game state."""

        # Continuous Action space
        self.kart_rotation += 2* action[0]
        # stay in degree value range (0-360)
        self.kart_rotation = self.kart_rotation % 360

        # calculate position change (1 step in depending rotation)
        # convert to radiants
        # theta = self.kart_rotation * 2 * math.pi
        rad = math.radians(self.kart_rotation)
        new_position = [
            self.kart_position[0] + 1 * math.cos(rad),
            self.kart_position[1] + 1 * math.sin(rad),
        ]
        self.kart_position = new_position

        # calculate if kart is on field
        is_done, color_done = self.is_done()

        if color_done == Colors.RED:
            return (self.get_state(), is_done, self.step_counter)

        return (self.get_state(), is_done, None)

    # returns the state (kart_position_x, kart_position_y, kart_rotation, dist)
    def get_state(self) -> np.array:
        """Gets the current state representation."""
        state = [
            # normalize to 
            self.normalize(self.kart_position[0], self.board_size_blocks[0]),
            self.normalize(self.kart_position[1], self.board_size_blocks[1]),
            self.normalize(self.kart_rotation, 360),
            self.dist_to_next_checkpoint() / self.board_size_blocks[0]
            #self.step_counter,  # Include step counter in the state
        ]

        return np.array(state, dtype=np.float32)
    
    # normalizes game metrics to range [-1;1]
    def normalize(self, metric, max_val):
        return 2 * (float(metric) / float(max_val)) - 1

    def is_done(self) -> tuple[bool, Colors]:
        self.step_counter += 1
        if self.step_counter > STEP_TIMEOUT:
            return True, None
        
        # convert cart position to pixels
        x = int(self.kart_position[0])
        y = int(self.kart_position[1])

        # get track color of current position
        color = self.track_image.get_at((x, y))[:3]

        if color == Colors.GRASS.value:
            return True, Colors.GREEN
        elif color == Colors.RED.value:
            return True, Colors.RED
        return False, None

    def dist_to_next_checkpoint(self):
        pos_kart = np.array(self.kart_position)
        pos_checkpoint = np.array( self.checkpoints[self.checkpoint_counter] )

        # euclidean distance
        dist = np.linalg.norm(pos_kart - pos_checkpoint)
        if dist < 50:
            # grey out previous checkpoint
            (x, y) = self.checkpoints[self.checkpoint_counter]
            pygame.draw.circle(self.track_image, Colors.BLACK.value, (int(x), int(y)), 5)  # Blue for world coords

            # update checkpoint
            self.checkpoint_counter = (self.checkpoint_counter+1) % len(self.checkpoints)
            (x, y) = self.checkpoints[self.checkpoint_counter]
            pygame.draw.circle(self.track_image, Colors.BLUE.value, (int(x), int(y)), 5)  # Blue for world coords

        return dist


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
        self.clock.tick(200)

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
