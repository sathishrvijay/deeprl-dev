import gymnasium as gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim

import typing as tt


"""This is the implementation of Cartpole RL using the PTAN wrapper libraries.
The goal is to demonstrate how much less code we need to write with these wrappers.

This will implement DQN w/
- An experience buffer that will uniform random sample mini batches for training
- Use a target network
- Will use Q-learning w/ epsilon-greedy for exploration
- Epsilon decay schedule
"""

# HPARAMS
OBS_DIM = 4
HIDDEN_LAYER_DIM = 128
N_ACTIONS = 2
GAMMA = 0.99
ALPHA = 1e-3
MIN_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.995
MAX_EPOCHS = 1000   # total number of epochs to collect experience/train/test on

BATCH_SIZE = 16
REPLAY_BUFFER_SIZE = 1000
TGT_NET_SYNC = 20   # sync every 20 steps

class AgentNet(nn.Module):
    def __init__(self, state_space_dim: int, h_layer_dim: int, n_actions: int):
        super(AgentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_space_dim, h_layer_dim),
            nn.ReLU(),
            nn.Linear(h_layer_dim, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


def unpack_batch(batch: tt.List[ptan.experience.ExperienceFirstLast],
    target_net: nn.Module,
    gamma: float):
    """Note: Since in general an experience sub-trajectory can be n-steps,
    the terminology used here is last state instead of next state.
    Additionally, reward is equal to the cumulative discounted rewards from intermediate steps
    All of this is subsumed within ptan.experience.ExperienceFirstLast"""
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    for exp in batch:
        # Each observation sub-trajectory in the replay buffer is a SARS' tuple
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        # Note: torch cannot deal with None type
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
        done_masks.append(exp.last_state is None)

    # Array stacking should work by default
    #states_v = torch.as_tensor(np.stack(states))
    #last_states_v = torch.as_tensor(np.stack(last_states))
    states_v, actions_v, rewards_v, last_states_v = \
        torch.tensor(states), torch.tensor(actions), torch.tensor(rewards), torch.tensor(last_states)

    last_state_q_v = target_net(last_states_v)
    # Note: extract max Q per obs (hence dim=1); [0] is because function returns arrays of values and
    # indices, but we only want the values
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, rewards_v + gamma * best_last_q_v


def core_training_loop(
    net: nn.Module,
    tgt_net: nn.Module,
    replay_buffer: ptan.experience.ExperienceReplayBuffer,
    optimizer,
    objective
    ):
    """Note that there are no mini-batches, a batch gets generated from agent's experience
    with the current policy, then trained on the batch.
    This process is repeated in a loop until the termination condition is reached."""

    batch = replay_buffer.sample(BATCH_SIZE)
    states_v, actions_v, target_return_v = unpack_batch(batch, tgt_net.target_model, GAMMA)

    optimizer.zero_grad()
    q_v = net(states_v)
    # Note: gather the Q values for the correponding actions for each obs
    q_v = q_v.gather(dim=1, index=actions_v.unsqueeze(-1)).squeeze(-1)
    loss_v = objective(q_v, target_return_v)
    loss_v.backward()
    optimizer.step()


if __name__ == "__main__":
    # instantiate key elements
    # - Cartpole env
    # - network & target network
    # - agent policy
    # - Replay buffer and expwrience generation
    # Core training loop
    # - Generate SARS replay buffer observations from source net
    # - Uniform random sample minibatch from replay buffer and unpack
    # - Compute returns from target net;
    # - Compute TD error, compute loss and backprop
    # - Decay epsilon
    # - Sync target net to training net every TGT_NET_SYNC steps
    # - Train until convergence

    # setup the environment
    env = gym.make("CartPole-v1")

    # setup the agent and target net
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = AgentNet(n_states, HIDDEN_LAYER_DIM, n_actions)
    tgt_net = ptan.agent.TargetNet(net)

    # setup the Agent policy, experience generation and Replay buffer
    base_action_selector = ptan.actions.ArgmaxActionSelector()
    experience_action_selector = \
        ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0, selector=base_action_selector)
    agent = ptan.agent.DQNAgent(net, experience_action_selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_BUFFER_SIZE)

    # intialize training
    step = 0
    episode = 0
    solved = False
    optimizer = optim.Adam(net.parameters(), ALPHA)
    objective = nn.functional.mse_loss

    while not solved:
        step += 1
        replay_buffer.populate(1)

        # Need to initialize replay buffer with enough experience before starting training
        if len(replay_buffer) < 2*BATCH_SIZE:
            continue

        core_training_loop(net, tgt_net, replay_buffer, optimizer, objective)
        experience_action_selector.epsilon *= EPSILON_DECAY_RATE

        if step % TGT_NET_SYNC == 0:
            tgt_net.sync()

        # test for success (i.e. most recent episode lasted > 150 steps)
        for reward, steps in exp_source.pop_rewards_steps():
            episode += 1
            print(f"{step}: episode {episode} done, reward={reward:.2f}, "
                  f"epsilon={experience_action_selector.epsilon:.2f}")
            solved = reward > 150

    if solved:
        print("Whee!")
