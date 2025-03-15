import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.ptr = 0
        self.max_size = max_size
        self.size = 0
        self.full_log = False

        # Preallocate memory for fast access
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        if self.size >= self.max_size:
            if not self.full_log:
                print("---------Buffer full---------")
                self.full_log = True

        """Efficiently store experience in preallocated arrays."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size  # Overwrite oldest
        self.size = min(self.size + 1, self.max_size)

    # returns a batch of random samples
    def sample(self, batch_size):
        """Fast sampling using NumPy indexing."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )