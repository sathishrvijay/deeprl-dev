#!/usr/bin/env python3.11
import enum
import typing as tt
from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn.functional as fn
import numpy as np

from cartpole_agent_nn import AgentNet


# HPARAMS
OBS_DIM = 4
HIDDEN_LAYER_DIM = 128
N_ACTIONS = 2
BATCH_SIZE = 16
PRUNE_PERCENTILE = 70

# Dataclasses to store the episodes for pruning and training runs
@dataclass
class EpisodeStep:
    obs: np.ndarray
    action: int

@dataclass
class Episode:
    episode_return: float
    steps: []


def generate_episodes(env: gym.Env, net: AgentNet, batch_size: int):
    """This function collects agent experience and stores the episodes in a replay buffer
    for training purposes"""
    obs, _ = env.reset()
    # print(f"init_obs: {obs}")
    batch = []
    episode_return = 0
    episode_steps = []

    while True:
        obs_v = torch.tensor(obs, dtype=torch.float32)
        # print(f"obs_v: {obs_v}")
        action_logits_v = net(obs_v)  # forward pass through network
        action_probas = fn.softmax(action_logits_v, dim=0).detach().numpy()
        # sample actions according to probability distribution
        sampled_action = np.random.choice(len(action_probas), p=action_probas)
        next_obs, reward, is_done, is_trunc, _ = env.step(sampled_action)

        # store the reward step and episode
        episode_return = episode_return + reward
        step = EpisodeStep(obs, sampled_action)
        episode_steps.append(step)

        if is_done or is_trunc:
            # store episode and add episode to current batch
            e = Episode(episode_return, episode_steps)
            batch.append(e)

            # reset to beginning of new episode
            episode_return = 0
            episode_steps = []
            next_obs, _ = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs


def core_training_loop(
    env: gym.Env,
    net: AgentNet,
    obs_v,
    action_labels_v,
    iter_no: int
    ):
    """Note that there are no mini-batches, we generate a batch, then train in a loop until we reach
    the exit condition of agent passing the test!"""
    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    optimizer.zero_grad()
    action_logits_v = net(obs_v)
    loss_v = objective(action_logits_v, action_labels_v)
    loss_v.backward()
    optimizer.step()
    # print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
    #             iter_no, loss_v.item(), reward_m, reward_b))
    print("%d: loss=%.3f" % (iter_no, loss_v.item()))


def unpack_batch(batch: []):
    """Takes a batch of episodes in the replay buffer and converts to tensors of observation, actions
    and rewards for training
    - Recall that each episode is an episode return and a sequence of steps which itself is a (obs, action) tuple
    """
    obs = []
    action = []
    returns = []
    for idx, episode in enumerate(batch):
        # print(f"Episode {idx}:")
        returns.extend([episode.episode_return])
        obs.extend([step.obs for step in episode.steps])
        action.extend([step.action for step in episode.steps])

    obs_v = torch.FloatTensor(np.vstack(obs))
    action_v = torch.LongTensor(action)

    # print(f"returns: {returns}")
    # print(f"Obs: {obs_v}")
    # print(f"actions: {action_v}")
    # print(f"returns shape: {len(returns)}")
    # print(f"Obs: {obs_v.shape}")
    # print(f"actions: {action_v.shape}")
    return obs_v, action_v, returns


if __name__ == "__main__":
    # Init agent and env
    env = gym.make("CartPole-v1")
    n_actions = int(env.action_space.n)
    # print(f"n_actions: {n_actions}")

    agent_nn = AgentNet(OBS_DIM, HIDDEN_LAYER_DIM, n_actions)
    # print(agent_nn.forward(torch.rand(4)))

    # count = 0
    # while count < 5:
    #     print(count)
    #     print()
    #     batch = generate_episodes(env, agent_nn, BATCH_SIZE)
    #     #print(f"batch_size: {len(batch)}; Eps 1: {batch[1]}")
    #     # for iter_no, batch in enumerate(batch):
    #     #    print(batch)
    #     count += 1

    # Remaining steps TODO 06/10/2025
    # prune episodes
    # Track performance in TensorBoard to visualize
    # Add termination condition when agent completes training

    for iter_no, batch in enumerate(generate_episodes(env, agent_nn, BATCH_SIZE)):
        obs_v, action_v, rewards = unpack_batch(batch)
        core_training_loop(env, agent_nn, obs_v, action_v, iter_no)
        if iter_no == 100:
            break
