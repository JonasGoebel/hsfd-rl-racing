import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),  # Ausgabe im Bereich [-1,1]
        )
        self.max_action = max_action

        #self.net[-1].weight.data.mul_(0.01)  # Scale down weights of last layer
        #self.net[-1].bias.data.mul_(0.01)  # Scale down biases as well

    def forward(self, state):
        x = self.net(state)
        # Net output is [-1,1], therefore scale to desired max values
        return self.max_action * x
