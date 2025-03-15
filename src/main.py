import pickle
import os

from Environment import Environment
from ddpg.DDPGAgent import DDPGAgent
from ddpg.ReplayBuffer import ReplayBuffer

def save_replay_buffer(replay_buffer, filename='replay_buffer.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(replay_buffer, f)

def load_replay_buffer(filename='replay_buffer.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    env = Environment()

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = float(env.max_action)

    # state_dim 3: (pos_x, pos_y, rotation)
    # action_dim 1: steering (left 0, forward 1, right 2)
    # max_action 1: steering only in [-1;1]
    agent = DDPGAgent(state_dim, action_dim, max_action)
    agent.load_models("trained")

    num_episodes = 10000
    episode_length = 5000 # veeery long

    if os.path.exists("my_replay_buffer.pkl"):
        replay_buffer = load_replay_buffer("my_replay_buffer.pkl")
        print(f"Loaded replay buffer with {replay_buffer.size} experiences")
    else:
        replay_buffer = ReplayBuffer(1000000, state_dim, action_dim) # stores around 400 episodes

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        #env.handle_events()
        # stabilize first

        # usually until done (while True), fixed value prevents infinite loops
        for _ in range(episode_length):
            #if (episode > 500):
            #env.render()
            #env.handle_events()

            # Select action and apply to environment
            action = agent.select_action(state, noise_strength=.3)
            next_state, reward, done = env.step(state, action)

            # Store experience in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # Train the agent
            agent.train(replay_buffer, batch_size=512)

            episode_reward += reward

            if done:
                break

            state = next_state

        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

        if (episode+1)%100 == 0:
            agent.save_models("trained")
            save_replay_buffer(replay_buffer, 'my_replay_buffer.pkl')

    agent.save_models("trained")
    save_replay_buffer('my_replay_buffer.pkl')



if __name__ == "__main__":
    main()
