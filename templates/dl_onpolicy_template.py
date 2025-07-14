import gymnasium as gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from datetime import datetime, timedelta

import typing as tt

from models import
from utils import PerformanceTracker, print_training_header, print_final_summary


"""This is the implementation of <> RL using the PTAN wrapper libraries. TODO fill out

"""

# HPARAMS
RL_ENV = ""
HIDDEN_LAYER_DIM = 256
HLAYER1_DIM = 128
HLAYER2_DIM = 64
# Split original layer2 between Advantage and Value function
HLAYER2V_DIM = 16
HLAYER2A_DIM = 48
GAMMA = 0.99
ALPHA = 3e-3
MIN_EPSILON = 0.05
MAX_EPOCHS = 2000   # total number of epochs to collect experience/train/test on
BATCH_SIZE = 32


def unpack_batch(batch: tt.List[ptan.experience.ExperienceFirstLast],
    target_net: ptan.agent.TargetNet,
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
    actions_v, rewards_v = torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32)
    states_v, last_states_v = \
        torch.tensor(np.stack(states), dtype=torch.float32), torch.tensor(np.stack(last_states), dtype=torch.float32)

    # Debug
    # if torch.max(rewards_v) == 1.0:
    #     print(f"we have a successful episode!")
    #     breakpoint()

    returns_v = ...

    # return SARS observations
    return states_v, actions_v, returns_v, last_states_v


def core_training_loop(
    net: nn.Module,
    optimizer,
    objective,
    beta
    ):
    """Fill in
    """

    # TODO sample experience
    states_v, actions_v, target_return_v = unpack_batch(batch, tgt_net, GAMMA)

    optimizer.zero_grad()

    q_v = net(states_v)
    # Note: gather the Q values for the correponding actions for each obs
    q_v = q_v.gather(dim=1, index=actions_v.unsqueeze(-1)).squeeze(-1)

    # Apply IS correction to loss calculation
    weights_v = torch.tensor(weights, dtype=torch.float32)

    loss_v = objective(q_v * scale, target_return_v * scale, reduction='none')
    loss_v = (loss_v * weights_v).mean()
    loss_v.backward()
    # breakpoint()
    # update the priorities in the buffer (basically TD error per obs) for current mini batch
    td_errors_v = (target_return_v - q_v).detach().abs()
    priorities = td_errors_v.cpu().numpy() + 1e-5
    replay_buffer.update_priorities(indices, priorities)

    optimizer.step()


def play_trials(test_env: gym.Env, net: nn.Module) -> float:
    """Note that we want a separate env for trials that doesn't mess with training env.
    We use a deterministic agent that makes the optimal moves during episode play w/o exploration
    because training is independent and already exploratory
    """
    _, _ = test_env.reset()  # Use test_env instead of env
    base_action_selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, base_action_selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(test_env, agent, gamma=GAMMA)
    reward = 0.0
    episode_count = 0
    exp_iterator = iter(exp_source)
    while episode_count < 10:  # Reduced from 20 to 10 for faster evaluation
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
    # - LunarLander env
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
    env = gym.make(RL_ENV)
    test_env = gym.make(RL_ENV)

    # setup the agent and target net
    n_states = env.observation_space.shape[0]  # LunarLander has Box(8,) observation space
    n_actions = env.action_space.n
    # net = dqn_models.DQNOneHL(n_states, HIDDEN_LAYER_DIM, n_actions)
    net = None
    if DUELING_DQN is True:
        net = dqn_models.DuelDQNTwoHL(n_states, HLAYER1_DIM, HLAYER2V_DIM, HLAYER2A_DIM, n_actions)
    else:
        net = dqn_models.DQNTwoHL(n_states, HLAYER1_DIM, HLAYER2_DIM, n_actions)
    tgt_net = ptan.agent.TargetNet(net)

    # setup the Agent policy, experience generation and Replay buffer
    base_action_selector = ptan.actions.ArgmaxActionSelector()
    experience_action_selector = \
        ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0, selector=base_action_selector)
    agent = ptan.agent.DQNAgent(net, experience_action_selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=3)


    # Initialize training with performance tracking
    iter_no = 0
    trial = 0
    solved = False
    optimizer = optim.Adam(net.parameters(), ALPHA)
    objective = nn.functional.mse_loss
    max_return = -1000.0

    # Initialize performance tracker and print training header
    perf_tracker = PerformanceTracker()
    network_config = f"{HLAYER1_DIM}-{HLAYER2_DIM} network"
    hyperparams = {
        'lr': ALPHA,
        'batch_size': BATCH_SIZE,
        'warmup_frames': PRIORITY_BUF_WARMUP_FRAMES,
        'ramp_frames': PRIORITY_BUF_RAMP_FRAMES,
        'reward_normalization': REWARD_NORMALIZATION
    }
    print_training_header(RL_ENV, network_config, hyperparams)

    while not solved:
        iter_no += 1
        iter_start_time = time.time()

        # Training step
        core_training_loop(net, tgt_net, replay_buffer, optimizer, objective, beta,
            reward_norm=REWARD_NORMALIZATION)

        training_time = time.time() - iter_start_time

        # Update all the various schedules (alpha, beta, epsilon)
        frame_idx += BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP

        # Epsilon decay for exploration
        experience_action_selector.epsilon = \
            max(MIN_EPSILON, 1.0 - float(frame_idx) / float(EPSILON_DECAY_FRAMES))

        if iter_no % TGT_NET_SYNC_PER_ITERS == 0:
            print(f"TARGET NET SYNC!! {iter_no}: frame {frame_idx},"
                  f"alpha={current_alpha:.4f}, beta={beta:.4f}")
            tgt_net.sync()

        # Print periodic performance summary every 500 iterations
        if iter_no % 500 == 0:
            perf_tracker.print_checkpoint(iter_no, frame_idx)

        # Flush out buffer and repopulate
        if iter_no % 1000 == 0:
            print("WARNING: RESETTING BUFFER!!")
            replay_buffer.populate(REPLAY_BUFFER_SIZE)

        # Test trials to check success condition
        eval_start_time = time.time()
        average_return = play_trials(test_env, tgt_net.target_model)
        eval_time = time.time() - eval_start_time

        max_return = average_return if (max_return < average_return) else max_return
        trial += 1

        # Log performance metrics
        perf_metrics = perf_tracker.log_iteration(iter_no, frame_idx, average_return, training_time)

        # Enhanced logging with timing information
        print(f"(iter: {iter_no:4d}, trial: {trial:4d}) - "
              f"avg_return={average_return:7.2f}, max_return={max_return:7.2f} | "
              f"alpha={current_alpha:.2f}, beta={beta:.2f}, eps={experience_action_selector.epsilon:.2f} | "
              f"train_time={training_time:.3f}s, eval_time={eval_time:.3f}s, "
              f"fps={perf_metrics['current_fps']:.1f}, total_time={perf_metrics['total_elapsed']/60:.1f}m")
        solved = average_return > 200.0  # LunarLander is considered solved at 200+ average reward

    # Training completed - print comprehensive summary using utility function
    final_summary = perf_tracker.get_summary()
    print_final_summary(
        solved=solved,
        average_return=average_return,
        target_reward=200.0,
        final_summary=final_summary,
        frame_idx=frame_idx,
        current_alpha=current_alpha,
        beta=beta,
        epsilon=experience_action_selector.epsilon,
        iter_no=iter_no,
        tgt_net_sync_iters=TGT_NET_SYNC_PER_ITERS
    )
