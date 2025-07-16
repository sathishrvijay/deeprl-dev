import gymnasium as gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from datetime import datetime, timedelta

import typing as tt
from functools import partial

from models import pgtr_models
from utils import PerformanceTracker, print_training_header, print_final_summary

"""This is the implementation of A2C with LunarLander-v1 RL using the PTAN wrapper libraries.
A2C serves as a performant baseline for Policy Gradient methods and a precursor to more advanced
PG methods like A3C, DDPG, SAC, PPO and others.
TODO: 07/15/25 - Add overall metrics tracking as well as tracking of total loss,
value loss, entropy bonus and advantage etc
"""

# HPARAMS
RL_ENV = "LunarLander-v2"
N_ENVS = 16
HIDDEN_LAYER_DIM = 256
HLAYER1_DIM = 128
# Split original layer2 between Advantage and Value function
HLAYER2V_DIM = 16
HLAYER2A_DIM = 48

CLIP_GRAD = 0.3   # typical values of clipping the L2 norm is 0.1 to 1.0
N_TD_STEPS = 10 # Number of steps aggregated per experience (n in n-step TD)
N_ROLLOUT_STEPS = 16 # formal rollout definition; batch_size = N_ENVS * N_ROLLOUT_STEPS

GAMMA = 0.995
LR_START = 7e-4
LR_DECAY_FRAMES = 5e7
ENTROPY_BONUS_BETA = 1e-3


def unpack_batch(batch: tt.List[ptan.experience.ExperienceFirstLast],
    net: ptan.agent.NNAgent,
    n_steps: int
    ):
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

    # return states, actions, returns
    with torch.no_grad():
        ls_values_v, _ = net(last_states_v)

    ls_values_v = ls_values_v.squeeze(1).data
    # Note: important step for computing returns correctly w/ loop unroll
    ls_values_v *= GAMMA ** n_steps
    returns_v = rewards_v + ls_values_v
    return states_v, actions_v, returns_v


def core_training_loop(
    net: nn.Module,
    batch: list,
    optimizer
    ):
    """In A2C, the entire generated batch of episodes is used for training in every epoch.
    Note: Actor head returns logits
    Returns: Dictionary containing loss components for tracking
    """
    optimizer.zero_grad()

    states_v, actions_v, target_return_v = unpack_batch(batch, net, N_ROLLOUT_STEPS)
    values_v, action_logits_v = net(states_v)
    action_probas_v = nn.functional.softmax(action_logits_v, dim=1)
    log_action_probas_v = nn.functional.log_softmax(action_logits_v, dim=1)

    # loss = mse_loss + pg_gain + entropy_bonus
    critic_loss_v = nn.functional.mse_loss(values_v.squeeze(-1), target_return_v)

    # Key notes for PG loss calc -
    # 1. detach values_v to prevent propagating PG loss into Critic network
    # 2. Normalize advantage for the batch (suggested by o3 to avoid learning plateaus)
    # 3. Gather probas only for the selected actions for PG loss
    # 4. Don't forget the negative sign for gradient ascent
    adv_v = target_return_v - values_v.squeeze(-1).detach()
    adv_v = (adv_v - adv_v.mean()) / (adv_v.std() + 1e-8)
    log_actions_v = log_action_probas_v.gather(dim=1, index=actions_v.unsqueeze(1)).squeeze(-1)
    pg_loss_v = -(adv_v * log_actions_v).mean()

    # Note: Entropy has a negative sign, but its a bonus, so double negatives cancel out
    entropy_bonus_v = ENTROPY_BONUS_BETA * (action_probas_v * log_action_probas_v).mean()

    loss_v = critic_loss_v + pg_loss_v + entropy_bonus_v
    loss_v.backward()
    # Note: clip gradients beore updating network weights
    torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
    optimizer.step()

    # Return loss components for tracking
    return {
        'total_loss': loss_v.item(),
        'critic_loss': critic_loss_v.item(),
        'actor_loss': pg_loss_v.item(),
        'entropy_loss': entropy_bonus_v.item()
    }


def play_trials(test_env: gym.Env, net: nn.Module) -> float:
    """Note that we want a separate env for trials that doesn't mess with training env.
    We might want to use a deterministic agent that makes the most probable moves for eval.
    """
    _, _ = test_env.reset()  # Use test_env instead of env
    experience_action_selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.ActorCriticAgent(net, experience_action_selector, apply_softmax=True)
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
    # - Setup env
    # - network
    # - agent class & policy
    # Core training loop
    # - Generate SARS observations from training net
    # - sample minibatch and unpack
    # - Compute returns
    # - Compute losses and backprop
    # - Update schedules if any
    # - Simulate trials & train until convergence

    # setup the parallel environment collection
    # env = gym.make(RL_ENV)
    env_fns = [partial(gym.make, RL_ENV) for _ in range(N_ENVS)]
    vector_env = gym.vector.SyncVectorEnv(env_fns)
    # don't need a vectorized trial env
    test_env = gym.make(RL_ENV)

    # setup the agent and target net
    n_states = vector_env.single_observation_space.shape[0]  # LunarLander has Box(8,) observation space
    n_actions = vector_env.single_action_space.n
    net = pgtr_models.A2CDiscreteAction(n_states, HLAYER1_DIM,
        HLAYER2V_DIM, HLAYER2A_DIM, n_actions)

    # Setup the Agent & policy
    # TODO: We need VectorExperience for multiple environments
    experience_action_selector = ptan.actions.ProbabilityActionSelector()
    # Note: network returns logits, so we need to apply softmax before
    # stochastically sampling actions
    agent = ptan.agent.ActorCriticAgent(net, experience_action_selector, apply_softmax=True)
    exp_source = ptan.experience.VectorExperienceSourceFirstLast(vector_env, agent, gamma=GAMMA,
        steps_count=N_TD_STEPS)

    # Initialize training with performance tracking
    iter_no = 0.0
    trial = 0
    solved = False
    optimizer = optim.Adam(net.parameters(), LR_START)
    objective = nn.functional.mse_loss
    max_return = -1000.0

    # Initialize performance tracker and print training header
    perf_tracker = PerformanceTracker()
    network_config = f"{HLAYER1_DIM}-{HLAYER2V_DIM}/{HLAYER2A_DIM} A2C network"
    hyperparams = {
        'n_envs': N_ENVS,
        'n_td_steps': N_TD_STEPS,
        'n_rollout_steps': N_ROLLOUT_STEPS,
        'batch_size': N_ENVS * N_ROLLOUT_STEPS,
        'lr_start': LR_START,
        'gamma': GAMMA,
        'entropy_beta': ENTROPY_BONUS_BETA,
        'clip_grad': CLIP_GRAD
    }
    print_training_header(RL_ENV, network_config, hyperparams)

    batch = []
    exp_iterator = iter(exp_source)
    # each iteration of exp_source yields N_ENVS experiences
    batch_size = N_ROLLOUT_STEPS * N_ENVS
    frame_idx = 0  # Track total frames of experience generated

    while not solved:
        iter_no += 1.0
        # Ensure LR never decays all the way to 0
        current_lr = max(1e-5, LR_START*(1.0 - frame_idx/LR_DECAY_FRAMES))
        for g in optimizer.param_groups:
            g["lr"] = current_lr
        iter_start_time = time.time()

        # Collect experiences from parallel envs for training
        while len(batch) < batch_size:
            exp = next(exp_iterator)
            batch.append(exp)
        frame_idx += batch_size

        # Training step
        loss_dict = core_training_loop(net, batch, optimizer)
        batch.clear()
        training_time = time.time() - iter_start_time

        # Print periodic performance summary every 500 iterations
        if iter_no % 10000 == 0:
            perf_tracker.print_checkpoint(int(iter_no), frame_idx)

        # Test trials to check success condition
        eval_start_time = time.time()
        average_return = play_trials(test_env, net)
        eval_time = time.time() - eval_start_time

        max_return = average_return if (max_return < average_return) else max_return
        trial += 1

        # Log performance metrics
        perf_metrics = perf_tracker.log_iteration(
            int(iter_no), frame_idx, average_return, training_time, eval_time, loss_dict
        )

        if iter_no % 100 == 0:
            # Enhanced logging with timing information
            print(f"(iter: {iter_no:6.0f}, trial: {trial:4d}) - "
                f"avg_return={average_return:7.2f}, max_return={max_return:7.2f} | "
                f"lr={current_lr:.6f}, "
                f"train_time={training_time:.3f}s, eval_time={eval_time:.3f}s, "
                f"fps={perf_metrics['current_fps']:.1f}, total_time={perf_metrics['total_elapsed']/60:.1f}m | "
                f"losses: total={loss_dict['total_loss']:.4f}, "
                f"critic={loss_dict['critic_loss']:.4f}, "
                f"actor={loss_dict['actor_loss']:.4f}, "
                f"entropy={loss_dict['entropy_loss']:.4f}"
            )
        solved = average_return > 200.0  # LunarLander is considered solved at 200+ average reward

    # Training completed - print comprehensive summary using utility function
    final_summary = perf_tracker.get_summary()
    print_final_summary(
        solved=solved,
        average_return=average_return,
        target_reward=200.0,
        final_summary=final_summary,
        frame_idx=frame_idx,
        current_alpha=current_lr,
        epsilon=0.0,  # Not using epsilon in this implementation
        iter_no=int(iter_no)
    )
