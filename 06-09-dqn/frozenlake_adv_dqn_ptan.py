import gymnasium as gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import typing as tt


"""This is the implementation of FrozenLake RL using the PTAN wrapper libraries.
The goal is to demonstrate how much less code we need to write with these wrappers.

This will implement advanced DQN features like
- A prioritized experience buffer to improve training convergencei
- Double DQN to reduce maximization bias
- Dueling DQN network
"""

# HPARAMS
#RL_ENV = "FrozenLake8x8-v1"
RL_ENV = "FrozenLake-v1"
HIDDEN_LAYER_DIM = 128
GAMMA = 0.99
ALPHA = 3e-4
MIN_EPSILON = 0.05
EPSILON_DECAY_FRAMES = 20000
MAX_EPOCHS = 1000   # total number of epochs to collect experience/train/test on
BATCH_SIZE = 64

# Priority buffer related
REPLAY_BUFFER_SIZE = 10000
BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP = 50
TGT_NET_SYNC_PER_ITERS = 20   # sync every 20 steps
PRIORITY_BUF_ALPHA = 0.6
PRIORITY_BUF_BETA_START = 0.4
PRIORITY_BUF_WARMUP_FRAMES = 5000
PRIORITY_BUF_BETA_FRAMES = 20000

class AgentNet(nn.Module):
    def __init__(self, state_space_dim: int, h_layer_dim: int, n_actions: int):
        super(AgentNet, self).__init__()
        self.state_space_dim = state_space_dim
        self.net = nn.Sequential(
            nn.Linear(state_space_dim, h_layer_dim),
            nn.ReLU(),
            nn.Linear(h_layer_dim, n_actions)
        )

    def forward(self, x: torch.Tensor):
        # FrozenLake uses discrete states, so we need to one hot encodes the states before use
        if x.dtype == torch.long or x.dtype == torch.int:
            x = torch.nn.functional.one_hot(x, num_classes=self.state_space_dim).float()
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

    # Array stacking during conv should work by default, but direct conversion from list of numpy array
    # to tensor is very slow, hence the np.stack(...)
    actions_v, rewards_v = torch.tensor(actions), torch.tensor(rewards)
    states_v, last_states_v = \
        torch.tensor(np.stack(states)), torch.tensor(np.stack(last_states))

    # Debug
    # if torch.max(rewards_v) == 1.0:
    #     print(f"we have a successful episode!")
    #     breakpoint()

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
    objective,
    beta
    ):
    """Mini batches are sampled from Prioritized replay buffer according to priorities
    This is basically sampling from buffer proportional to TD errors.
    * Priorities are updated based on absolute TD errors, not the MSE loss
    * Reweight the losses to correct for sampling bias from Prio buffer
    """

    batch, indices, weights  = replay_buffer.sample(BATCH_SIZE, beta=beta)
    states_v, actions_v, target_return_v = unpack_batch(batch, tgt_net.target_model, GAMMA)

    optimizer.zero_grad()
    # ensure float (FrozenLake is torch.long())
    q_v = net(states_v)
    # Note: gather the Q values for the correponding actions for each obs
    q_v = q_v.gather(dim=1, index=actions_v.unsqueeze(-1)).squeeze(-1)

    # Apply IS correction to loss calculation
    weights_v = torch.tensor(weights, dtype=torch.float32)
    loss_v = objective(q_v, target_return_v, reduction='none')
    loss_v = (loss_v * weights_v).mean()
    loss_v.backward()
    # breakpoint()
    # update the priorities in the buffer (basically TD error per obs) for current mini batch
    td_errors_v = (target_return_v - q_v).detach().abs()
    priorities = td_errors_v.cpu().numpy() + 1e-5
    replay_buffer.update_priorities(indices, priorities)

    optimizer.step()


def play_trials(test_env: gym.Env, net: AgentNet) -> float:
    """Note that we want a separate env for trials that doesn't mess with training env.
    We use a deterministic agent that makes the optimal moves during episode play w/o exploration
    because training is independent and already exploratory
    """
    _, _ = env.reset()
    base_action_selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, base_action_selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(test_env, agent, gamma=GAMMA)
    reward = 0.0
    episode_count = 0
    exp_iterator = iter(exp_source)
    while episode_count < 20:
        while True:
            exp = next(exp_iterator)
            reward += exp.reward
            if exp.last_state is None:
                episode_count += 1
                break

    reward /= episode_count
    #print(f"Average return: {reward:.2f}")
    return reward


if __name__ == "__main__":
    # instantiate key elements
    # - FrozenLake env
    # - network & target network
    # - agent policy
    # - Replay buffer and experience generation
    # Core training loop
    # - Generate SARS replay buffer observations from source net
    # - sample minibatch from replay buffer and unpack
    # - Compute returns from target net;
    # - Compute TD error, compute loss and backprop
    # - Update schedules
    # - Sync target net to training net every TGT_NET_SYNC steps
    # - Simulate trials & train until convergence

    # setup the environment
    env = gym.make(RL_ENV, is_slippery=True)
    test_env = gym.make(RL_ENV, is_slippery=True)

    # setup the agent and target net
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    net = AgentNet(n_states, HIDDEN_LAYER_DIM, n_actions)
    tgt_net = ptan.agent.TargetNet(net)

    # setup the Agent policy, experience generation and Replay buffer
    base_action_selector = ptan.actions.ArgmaxActionSelector()
    experience_action_selector = \
        ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0, selector=base_action_selector)
    agent = ptan.agent.DQNAgent(net, experience_action_selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    # Initialize priority replay buff
    frame_idx = 0 # used for Priority Buffer beta annealing
    beta = beta_start = PRIORITY_BUF_BETA_START
    # Allow warm start with uniform sampling in the beginning by setting alpha = 0.0
    replay_buffer = \
        ptan.experience.PrioritizedReplayBuffer(exp_source, buffer_size=REPLAY_BUFFER_SIZE,
            alpha=1e-5)
    replay_buffer.populate(REPLAY_BUFFER_SIZE)

    # intialize training
    iter_no = 0
    trial = 0
    solved = False
    optimizer = optim.Adam(net.parameters(), ALPHA)
    objective = nn.functional.mse_loss
    max_return = 0.0

    while not solved:
        iter_no += 1
        # breakpoint()
        replay_buffer.populate(BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP)
        core_training_loop(net, tgt_net, replay_buffer, optimizer, objective, beta)

        # Update all the various schedules (alpha, beta, epsilon)
        frame_idx += BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP
        if frame_idx > PRIORITY_BUF_WARMUP_FRAMES:   # 5k frames ≈ 100 episodes on 4×4 map
            replay_buffer._alpha = PRIORITY_BUF_ALPHA
        beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / PRIORITY_BUF_BETA_FRAMES)
        experience_action_selector.epsilon = \
            max(MIN_EPSILON, 1.0 - float(frame_idx) / float(EPSILON_DECAY_FRAMES))


        if iter_no % TGT_NET_SYNC_PER_ITERS == 0:
            print(f"{iter_no}: frame {frame_idx}, beta={beta:.4f}")
            tgt_net.sync()

        # Test trials to check success condition
        average_return = play_trials(test_env, tgt_net.target_model)
        max_return = average_return if (max_return < average_return) else max_return
        trial += 1
        print(f"(iter: {iter_no}, trial: {trial}) - avg_return={average_return:.2f}, max_return={max_return:.2f} "
                  f"alpha={replay_buffer._alpha:.2f}; beta={beta:.2f}; eps={experience_action_selector.epsilon:.2f}")
        solved = average_return > 0.8

    if solved:
        print("Whee!")
