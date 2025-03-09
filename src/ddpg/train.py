import torch
import torch.nn as nn


def train_actor(actor, critic, actor_optimizer, replay_buffer):
    state, _, _, _, _ = replay_buffer.sample()

    # prediction using the actor model
    action = actor(state)

    q_value = critic(state, action)
    loss = -q_value.mean()

    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()


def train_critic(critic, target_critic, target_actor, optimizer, replay_buffer, gamma=0.99):
    state, action, reward, next_state, done = replay_buffer.sample()

    with torch.no_grad():
        # prediction using the actor model
        next_action = target_actor(next_state)

        target_q_value = target_critic(next_state, next_action)
        y = reward + (1 - done) * gamma * target_q_value

    q_value = critic(state, action)

    loss = nn.MSELoss()(q_value, y)

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_ddpg(
    actor,
    critic,
    target_actor,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    replay_buffer,
    gamma=0.99,
    tau=0.99,
):

    train_critic(critic, target_critic, critic_optimizer, replay_buffer, gamma)
    train_actor(actor, critic, actor_optimizer, replay_buffer)

    # soft-updates for target networks
    for param, target_param in zip(
        critic.parameters(), target_critic.parameters()
    ):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

    for param, target_param in zip(
        actor.parameters(), target_actor.parameters()
    ):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )
