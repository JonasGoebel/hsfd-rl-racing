import torch
import torch.optim as optim
import pygame
import numpy as np

from RacingGame import RacingGame
from ddpg.Actor import Actor
from ddpg.Critic import Critic
from ddpg.ReplayBuffer import ReplayBuffer
from ddpg.train import train_ddpg
from ddpg.OUNoise import OUNoise


def main():
    game_size_pixels = (1000, 1000)
    game = RacingGame(game_size_pixels)

    is_done = False
    while not is_done:
        action = game.handle_events()

        new_state, reward, is_done = game.step(action)
        game.render()


def train():
    num_episodes = 1000
    max_steps = 200

    state_dim = 4
    action_dim = 1 # (only steering)
    max_action=1    # -1 = left, 0=forward, 1=right
    
    ou_noise = OUNoise(action_dim=1)

    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)

    # copy target networks with weights
    target_actor = Actor(state_dim, action_dim, max_action)
    target_critic = Critic(state_dim, action_dim)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer(10000, state_dim, action_dim)

    game_size_pixels = (1000, 1000)
    game = RacingGame(game_size_pixels)

    for episode in range(num_episodes):
        pygame.event.get()
        print("episode", episode)
        state = game.reset()
        total_reward = 0

        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            action = actor(state_tensor).detach().numpy()[0]
            
            # modify action using gaussian noise
            # noise_std = 0.2
            # noise = np.random.normal(0, noise_std, size=action.shape) 
            # action = np.clip(action + noise, -1, 1)
            
            # modify action using OU noise
            action = np.clip(action + ou_noise.noise(), -1, 1)
            
            next_state, reward, done = game.step(action)
            game.render()

            replay_buffer.add(state, action, reward, next_state, done)
            
            # if len(replay_buffer.buffer) > 64:
            if replay_buffer.size > 64:
                train_ddpg(
                    actor,
                    critic,
                    target_actor,
                    target_critic,
                    actor_optimizer,
                    critic_optimizer,
                    replay_buffer,
                )

            state = next_state
            total_reward += reward

            if done:
                print("Total reward:", total_reward)
                break


if __name__ == "__main__":
    # main()
    train()
