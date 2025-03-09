import torch
import torch.nn as nn
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # q-value
        )

    def forward(self, state, action):
        # combine state and action as one input
        x = torch.cat([state, action], dim=1)
        return self.net(x)
