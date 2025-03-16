import pickle
import os
import argparse

from Environment import Environment
from ddpg.DDPGAgent import DDPGAgent
from ddpg.ReplayBuffer import ReplayBuffer

def save_replay_buffer(replay_buffer, filename='replay_buffer.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(replay_buffer, f)
        print("Saved Replay Buffer")

def load_replay_buffer(filename='replay_buffer.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train(model_path="", num_episodes=3000, episode_length=2500, exploration=.2, batch_size=128):
    env = Environment("train")
    agent, replay_buffer = initialize_training(env, model_path)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(episode_length):
            # Select action and apply to environment
            action = agent.select_action(state, noise_strength=exploration)
            next_state, reward, done = env.step(state, action)

            # Store experience in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # Train the agent
            agent.train(replay_buffer, batch_size=batch_size)

            episode_reward += reward

            if done:
                break

            state = next_state

        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

        if (episode + 1) % 100 == 0:
            save_progress(agent, replay_buffer, model_path)

    save_progress(agent, replay_buffer, model_path)

def initialize_training(env, model_path):
    """Initializes the agent and replay buffer, loading existing data if available."""
    state_dim, action_dim, max_action = env.state_dim, env.action_dim, float(env.max_action)

    agent = DDPGAgent(state_dim, action_dim, max_action)
    agent.load_models(model_path)

    if os.path.exists( os.path.join(model_path, "my_replay_buffer.pkl") ):
        replay_buffer = load_replay_buffer( os.path.join(model_path, "my_replay_buffer.pkl") )
        print(f"Loaded replay buffer with {replay_buffer.size} experiences")
    else:
        replay_buffer = ReplayBuffer(1000000, state_dim, action_dim) # stores around 1.600 epochs
        print(f"Created new Replay Buffer")

    return agent, replay_buffer

def save_progress(agent, replay_buffer, model_path):
    """Saves the trained models and replay buffer."""
    agent.save_models(model_path)
    save_replay_buffer(replay_buffer, os.path.join(model_path, "my_replay_buffer.pkl"))

def test(model_path=""):
    env = Environment("test")

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = float(env.max_action)

    agent = DDPGAgent(state_dim, action_dim, max_action)
    agent.load_models(model_path)

    while True:  # Run indefinitely
        state = env.reset()

        while True:  # Single episode loop
            env.render()
            env.handle_events()

            # Select action and apply to environment
            action = agent.select_action(state, noise_strength=0)  # No exploration noise
            next_state, _, done = env.step(state, action)

            if done:
                break  # Reset environment after finishing episode

            state = next_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--test", action="store_true", help="Test the model")

    args = parser.parse_args()

    if args.train:
        train(model_path="model_latest")
    elif args.test:
        test(model_path="model_latest")