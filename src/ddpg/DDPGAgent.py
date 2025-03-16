import torch
import torch.nn.functional as F
import numpy as np
import os

from ddpg.Actor import Actor
from ddpg.Critic import Critic

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005):
        '''
        Initialize the DDPG Agent
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)           # create Actor
        self.critic = Critic(state_dim, action_dim).to(self.device)                     # create Critic

        self.target_actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        # Initialize target networks to match the original networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, state, noise_strength=0.1):
        '''
        Initialize the DDPG Agent
        '''

        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).detach().cpu().numpy()
        # Ornstein-Uhlenbeck nose
        action += np.random.normal(0, noise_strength, size=action.shape)
        #action += noise_strength * np.random.uniform(-1, 1, size=action.shape) # Add exploration noise
        return np.clip(action, -1, 1)  # Ensure actions are within valid range


    def train(self, replay_buffer, batch_size=64):
        '''
        Train
        '''

        # return if replay_buffer does not have enough samples
        if replay_buffer.size < batch_size:
            return

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Convert to PyTorch tensors and move to GPU
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)


        # Compute target Q-value: y = r + γ * Q'(s', π'(s'))
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + self.gamma * (1 - dones) * self.target_critic(next_states, next_actions)

        # Update critic network
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network (maximize Q-value)
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks (soft update)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, filename):
        '''
        Save the models' state_dicts to a file.
        '''
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.target_actor.state_dict(), filename + "_target_actor.pth")
        torch.save(self.target_critic.state_dict(), filename + "_target_critic.pth")
        print(f"Models saved to {filename}")

    def load_models(self, filename):
        '''
        Load the models' state_dicts from a file.
        '''
        model_files = {
            "actor": filename + "_actor.pth",
            "critic": filename + "_critic.pth",
            "target_actor": filename + "_target_actor.pth",
            "target_critic": filename + "_target_critic.pth",
        }

        # Check if all files exist before loading
        if all(os.path.exists(path) for path in model_files.values()):
            for model_name, file_path in model_files.items():
                getattr(self, model_name).load_state_dict(torch.load(file_path))

            # Move models to the correct device
            self.actor.to(self.device)
            self.critic.to(self.device)
            self.target_actor.to(self.device)
            self.target_critic.to(self.device)

            print(f"Models loaded from \"{filename}\"")
        else:
            print(f"No model loaded: Not all files found at \"{filename}. Training from scratch...\"")