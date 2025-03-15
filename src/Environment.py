from RacingGame import RacingGame
import numpy as np

# Wrapper class for RacingGame
# Extends Game and provides useful environment information
class Environment(RacingGame):

    def __init__(self):
        game_size_pixels = (1000, 1000)
        super().__init__(game_size_pixels)

        # state dim (4): pos_x, pos_y, rotation, dist_to_next_checkpoint
        self.state_dim = 4
        # action_dim (1): steering
        self.action_dim = 1
        # Continuous Values in [-1,1]
        self.max_action = 1

    def step(self, state, action):
        next_state, is_done = super().step(action)

        # calculate reward w.r.t. delta_checkpoint_dist and whether is done
        reward = self.reward(state[3], next_state[3], is_done)

        return next_state, reward, is_done
    
    def reward(self, current_dist, next_dist, is_done):
        if is_done:
            return -50 # big punishment

        current_dist = np.round(100 * current_dist, 3)
        next_dist = np.round(100 * next_dist, 3)

        # Scale the reward relative to max. possible reward in one step
        improvement = current_dist - next_dist
        max_improvement = 0.1 # figured out through testing

        # only happens during checkpoint change
        if improvement < -max_improvement-.1 or improvement > max_improvement+.1:
            return 0
        
        # higher rewards for faster routes        
        reward = np.round( (improvement/max_improvement)*1,  2)

        return reward
    


