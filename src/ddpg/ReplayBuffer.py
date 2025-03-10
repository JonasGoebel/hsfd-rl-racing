import torch
import torch.optim as optim
import numpy as np


class ReplayBuffer:
    # def __init__(self, capacity=100000):
    def __init__(self, max_size, state_dim, action_dim):
        # self.buffer = []
        # self.capacity = capacity
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        # if len(self.buffer) >= self.capacity:
            # self.buffer.pop(0)
        # self.buffer.append((state, action, reward, next_state, done))
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size  # ring buffer
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=64):
        
        # print(self.buffer)
        # print(type(self.buffer))
        
        # print("1======")
        # self.buffer = np.array(self.buffer, dtype=object)
        # print("2======")
        # print(self.buffer.shape)
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.states[indices], dtype=torch.float32),
            torch.tensor(self.actions[indices], dtype=torch.float32),
            torch.tensor(self.rewards[indices], dtype=torch.float32),
            torch.tensor(self.next_states[indices], dtype=torch.float32),
            torch.tensor(self.dones[indices], dtype=torch.float32),
        )
        
        # batch = np.random.choice(self.buffer, batch_size, replace=False)
        # state, action, reward, next_state, done = zip(*batch)
        # return (
        #     torch.tensor(np.array(state), dtype=torch.float32),
        #     torch.tensor(np.array(action), dtype=torch.float32),
        #     torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(1),
        #     torch.tensor(np.array(next_state), dtype=torch.float32),
        #     torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1),
        # )
