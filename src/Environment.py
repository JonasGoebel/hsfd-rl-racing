from RacingGame import RacingGame
import numpy as np

# Wrapper class for RacingGame
# Extends Game and provides useful environment information
class Environment(RacingGame):

    def __init__(self, mode="train"):
        game_size_pixels = (1000, 1000)
        super().__init__(game_size_pixels, mode)

        # state dim (4): pos_x, pos_y, rotation, dist_to_next_checkpoint
        self.state_dim = 4
        # action_dim (1): steering
        self.action_dim = 1
        # Continuous Values in [-1,1]
        self.max_action = 1

    def step(self, state, action):
        # if step_counter not None, racer has passed finish line
        next_state, is_done, step_counter = super().step(action)

        # calculate reward w.r.t. delta_checkpoint_dist and whether is done
        reward = self.reward(state[3], next_state[3], is_done, step_counter)

        return next_state, reward, is_done
    
    def reward(self, current_dist, next_dist, is_done, step_counter):
        # Racer reached finish line
        if step_counter:
            # reward fewer steps towards goalline
            # assume steps below 2000 to not be possible
            return 500- (step_counter-2000)
        
        # has not reached finish line
        if is_done:
            return -50 # big punishment
        
        # on track, reward progress towards checkpoint
        current_dist = np.round(100 * current_dist, 3)
        next_dist = np.round(100 * next_dist, 3)

        # Scale the reward relative to max. possible reward in one step
        improvement = current_dist - next_dist
        max_improvement = 0.1 # figured out through testing

        # only happens during checkpoint change
        if improvement < -max_improvement-.1 or improvement > max_improvement+.1:
            # TODO: reward quicker paths to checkpoint
            return 0
        
        # quadratically higher rewards for faster improvement towards checkpoint
        alpha = 2 # quadratic reward scaling
        relative_reward = improvement/max_improvement # reward in [-1;1]
        reward = np.sign(relative_reward) * np.round( abs(relative_reward)**alpha,  2)

        return reward
    


