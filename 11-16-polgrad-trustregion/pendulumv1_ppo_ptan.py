import gymnasium as gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math

import typing as tt
from functools import partial

from models import agents, pgtr_models
from utils import PerformanceTracker, print_training_header, print_final_summary

"""This is the implementation of PPO with Pendulum-v1 RL for CA spaces using the PTAN wrapper
libraries. Uses separate Actor and Critic networks with different optimizers and learning rates.
TODO: 07/30
- Start w/ core training loop mod - changes from A2C related to Actor Surrogate loss
- Actor & critic nets and Agent - can be same as A2C
- Experience collection - changes required here for full trajectory rollouts
- unpack batch - requires GAE advantage calc here instead of simple Target return
"""

# HPARAMS
RL_ENV = "Pendulum-v1"
N_ENVS = 16

# Separate network dimensions
CRITIC_HIDDEN1_DIM = 128
CRITIC_HIDDEN2_DIM = 32
ACTOR_HIDDEN1_DIM = 128
ACTOR_HIDDEN2_DIM = 64

N_TD_STEPS = 4 # Number of steps aggregated per experience (n in n-step TD)
N_ROLLOUT_STEPS = 16 # formal rollout definition; batch_size = N_ENVS * N_ROLLOUT_STEPS

GAMMA = 0.99  # Slightly lower for Pendulum's shorter episodes
# Separate learning rates for actor and critic
CRITIC_LR_START = 7e-4
CRITIC_LR_END = 1e-4
ACTOR_LR_START = 3e-4
ACTOR_LR_END = 5e-5
LR_DECAY_FRAMES = 5e7

# PG related - adjusted for continuous actions
ENTROPY_BONUS_BETA_START = 1e-2  # Higher initial exploration
ENTROPY_BONUS_BETA_END = 1e-4    # Lower final exploration
ENTROPY_DECAY_FRAMES = 2e6       # Decay over 2M frames
CLIP_GRAD = 0.3   # typical values of clipping the L2 norm is 0.1 to 1.0

# Pendulum success threshold
PENDULUM_SOLVED_REWARD = -200.0  # Pendulum is solved when avg reward > -200


def unpack_batch(batch: tt.List[ptan.experience.ExperienceFirstLast],
    net: nn.Module,
    n_steps: int
    ):
    """Note: Since in general an experience sub-trajectory can be n-steps,
    the terminology used here is last state instead of next state.
    Additionally, reward is equal to the cumulative discounted rewards from intermediate steps
    All of this is subsumed within ptan.experience.ExperienceFirstLast
    Modified for continuous actions - actions are now continuous values instead of discrete indices.
    """
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
    actions_v = torch.tensor(np.stack(actions), dtype=torch.float32)  # Continuous actions are float
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    states_v = torch.tensor(np.stack(states), dtype=torch.float32)
    last_states_v = torch.tensor(np.stack(last_states), dtype=torch.float32)

    # return states, actions, returns
    with torch.no_grad():
        _, _, ls_values_v = net(last_states_v)

    ls_values_v = ls_values_v.squeeze(1).data
    # Note: important step for computing returns correctly w/ loop unroll
    ls_values_v *= GAMMA ** n_steps
    # zero out the terminated episodes
    done_masks_v = torch.tensor(done_masks, dtype=torch.bool)
    ls_values_v[done_masks_v] = 0.0

    returns_v = rewards_v + ls_values_v
    return states_v, actions_v, returns_v


def core_training_loop(
    net: nn.Module,
    batch: list,
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    frame_idx: int,
    iter_no=None,
    reward_norm=True,
    debug=True
    ):
    """In A2C, the entire generated batch of episodes is used for training in every epoch.
    Modified for continuous actions using Gaussian policy with log variance parameterization.
    Returns: Dictionary containing loss components for tracking
    """
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    states_v, actions_v, target_return_v = unpack_batch(batch, net, N_TD_STEPS)
    actions_mu_v, actions_logvar_v, values_v = net(states_v)

    # Compute current entropy bonus with decay
    entropy_progress = min(1.0, frame_idx / ENTROPY_DECAY_FRAMES)
    current_entropy_beta = ENTROPY_BONUS_BETA_START * (1 - entropy_progress) + ENTROPY_BONUS_BETA_END * entropy_progress

    # Critic loss - same as discrete case
    if reward_norm:
        # Pendulum rewards are in [-16, 0] range, so normalize differently than LunarLander
        values_v = values_v / 16.0
        target_return_v = target_return_v / 16.0
    critic_loss_v = nn.functional.mse_loss(values_v.squeeze(-1), target_return_v)

    # Compute advantages
    adv_v = target_return_v - values_v.squeeze(-1).detach()
    # Normalize advantages
    adv_std = max(1e-3, adv_v.std(unbiased=False) + 1e-8)
    adv_v = (adv_v - adv_v.mean()) / adv_std

    # Compute log probabilities for continuous actions
    # Gaussian log probability: -0.5 * (log(2π) + log_var + (x-μ)²/σ²)
    log_prob_v = -0.5 * (
        torch.log(torch.tensor(2 * math.pi)) +
        actions_logvar_v +
        (actions_v - actions_mu_v).pow(2) / torch.exp(actions_logvar_v)
    )
    log_prob_v = log_prob_v.sum(dim=-1)  # Sum over action dimensions

    # Policy gradient loss
    pg_loss_v = -(adv_v * log_prob_v).mean()

    # Entropy bonus for continuous actions
    # Gaussian entropy: 0.5 * log(2πe * σ²) = 0.5 * (log(2πe) + log_var)
    entropy_v = 0.5 * (actions_logvar_v + math.log(2 * math.pi * math.e))
    entropy_v = entropy_v.sum(dim=-1).mean()  # Sum over action dims, mean over batch
    entropy_bonus_v = current_entropy_beta * entropy_v

    # Variance regularization to prevent collapse
    var_penalty_v = 1e-4 * torch.exp(actions_logvar_v).mean()

    # Separate loss computation and backpropagation
    # Critic loss (only affects critic network)
    critic_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(net.get_critic_parameters(), CLIP_GRAD)
    critic_optimizer.step()
    critic_scheduler.step()

    # Actor loss (only affects actor network)
    actor_loss_v = pg_loss_v - entropy_bonus_v + var_penalty_v
    actor_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(net.get_actor_parameters(), CLIP_GRAD)
    actor_optimizer.step()
    actor_scheduler.step()

    # Total loss for logging (not used for backprop)
    total_loss_v = critic_loss_v + actor_loss_v

    # Debug - check policy statistics
    if debug and iter_no % 100 == 0:
        with torch.no_grad():
            mean_action = actions_mu_v.mean(dim=0).cpu().numpy()
            std_action = torch.exp(actions_logvar_v / 2).mean(dim=0).cpu().numpy()
            print(f"Action μ={mean_action[0]:.3f}, σ={std_action[0]:.3f}, entropy_β={current_entropy_beta:.4e}")

    # Return loss components for tracking
    return {
        'total_loss': total_loss_v.item(),
        'critic_loss': critic_loss_v.item(),
        'actor_loss': actor_loss_v.item(),
        'entropy_raw': entropy_v.item(),
        'entropy_loss': entropy_bonus_v.item(),
        'var_penalty': var_penalty_v.item(),
        'current_entropy_beta': current_entropy_beta
    }


def play_trials(test_env: gym.Env, net: nn.Module) -> float:
    """Note that we want a separate env for trials that doesn't mess with training env.
    We use deterministic actions (mean only) for evaluation to get consistent performance measurement.
    Modified for continuous actions.
    """
    _, _ = test_env.reset()  # Use test_env instead of env
    # Use deterministic action selection for evaluation
    agent = agents.AgentContinuousA2C(net, deterministic=True)
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
    env_fns = [partial(gym.make, RL_ENV) for _ in range(N_ENVS)]
    vector_env = gym.vector.SyncVectorEnv(env_fns)
    # don't need a vectorized trial env
    test_env = gym.make(RL_ENV)

    # setup the agent and target net - using continuous action networks
    n_states = vector_env.single_observation_space.shape[0]  # Pendulum has Box(3,) observation space
    n_actions = vector_env.single_action_space.shape[0]      # Pendulum has Box(1,) action space
    net = pgtr_models.ContinuousA2C(
        n_states, n_actions,
        CRITIC_HIDDEN1_DIM, CRITIC_HIDDEN2_DIM,
        ACTOR_HIDDEN1_DIM, ACTOR_HIDDEN2_DIM
    )

    # Setup the Agent & policy - action selection policy built in
    agent = agents.AgentContinuousA2C(net)
    exp_source = ptan.experience.VectorExperienceSourceFirstLast(vector_env, agent, gamma=GAMMA,
        steps_count=N_TD_STEPS)

    # Initialize training with performance tracking
    iter_no = 0.0
    trial = 0
    solved = False

    # Separate optimizers for actor and critic
    # A2C reports more stable results w/ RMSProp for critic, Adam for actor
    critic_optimizer = optim.RMSprop(net.get_critic_parameters(), lr=CRITIC_LR_START, eps=1e-5, alpha=0.99)
    actor_optimizer = optim.Adam(net.get_actor_parameters(), lr=ACTOR_LR_START, eps=1e-5)

    max_return = -1000.0

    # each iteration of exp_source yields N_ENVS experiences
    batch_size = N_ROLLOUT_STEPS * N_ENVS

    # Separate schedulers for actor and critic
    critic_scheduler = optim.lr_scheduler.LinearLR(
        critic_optimizer, start_factor=1.0, end_factor=CRITIC_LR_END / CRITIC_LR_START,
        total_iters=LR_DECAY_FRAMES // batch_size)
    actor_scheduler = optim.lr_scheduler.LinearLR(
        actor_optimizer, start_factor=1.0, end_factor=ACTOR_LR_END / ACTOR_LR_START,
        total_iters=LR_DECAY_FRAMES // batch_size)

    # Initialize performance tracker and print training header
    perf_tracker = PerformanceTracker()
    network_config = f"Continuous A2C: Critic({CRITIC_HIDDEN1_DIM}-{CRITIC_HIDDEN2_DIM}), Actor({ACTOR_HIDDEN1_DIM}-{ACTOR_HIDDEN2_DIM})"
    hyperparams = {
        'n_envs': N_ENVS,
        'n_td_steps': N_TD_STEPS,
        'n_rollout_steps': N_ROLLOUT_STEPS,
        'batch_size': N_ENVS * N_ROLLOUT_STEPS,
        'critic_lr_start': CRITIC_LR_START,
        'critic_lr_end': CRITIC_LR_END,
        'actor_lr_start': ACTOR_LR_START,
        'actor_lr_end': ACTOR_LR_END,
        'lr_decay_frames': LR_DECAY_FRAMES,
        'gamma': GAMMA,
        'entropy_beta_start': ENTROPY_BONUS_BETA_START,
        'entropy_beta_end': ENTROPY_BONUS_BETA_END,
        'entropy_decay_frames': ENTROPY_DECAY_FRAMES,
        'clip_grad': CLIP_GRAD,
        'solved_threshold': PENDULUM_SOLVED_REWARD
    }
    print_training_header(RL_ENV, network_config, hyperparams)

    batch = []
    exp_iterator = iter(exp_source)
    frame_idx = 0  # Track total frames of experience generated
    while not solved:
        iter_no += 1.0
        iter_start_time = time.time()

        # Collect experiences from parallel envs for training
        while len(batch) < batch_size:
            exp = next(exp_iterator)
            batch.append(exp)
        frame_idx += batch_size

        # Training step with separate optimizers
        loss_dict = core_training_loop(
            net, batch, actor_optimizer, critic_optimizer,
            actor_scheduler, critic_scheduler, frame_idx, iter_no
        )
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
            # Enhanced logging with timing information and separate learning rates
            critic_lr = critic_optimizer.param_groups[0]["lr"]
            actor_lr = actor_optimizer.param_groups[0]["lr"]
            print(f"(iter: {iter_no:6.0f}, trial: {trial:4d}) - "
                f"avg_return={average_return:7.2f}, max_return={max_return:7.2f} | "
                f"critic_lr={critic_lr:.5e}, actor_lr={actor_lr:.5e}, "
                f"train_time={training_time:.3f}s, eval_time={eval_time:.3f}s, "
                f"fps={perf_metrics['current_fps']:.1f}, total_time={perf_metrics['total_elapsed']/60:.1f}m | "
                f"losses: total={loss_dict['total_loss']:.4f}, "
                f"critic={loss_dict['critic_loss']:.4f}, "
                f"actor={loss_dict['actor_loss']:.4f}, "
                f"entropy_raw={loss_dict['entropy_raw']:.4f}, "
                f"entropy_bonus={loss_dict['entropy_loss']:.4e}, "
                f"var_penalty={loss_dict['var_penalty']:.4e}, "
                f"entropy_β={loss_dict['current_entropy_beta']:.4e}"
            )

        solved = average_return > PENDULUM_SOLVED_REWARD  # Pendulum is solved when avg reward > -200

    # Training completed - print comprehensive summary using utility function
    final_summary = perf_tracker.get_summary()
    critic_lr = critic_optimizer.param_groups[0]["lr"]
    actor_lr = actor_optimizer.param_groups[0]["lr"]
    print_final_summary(
        solved=solved,
        average_return=average_return,
        target_reward=PENDULUM_SOLVED_REWARD,
        final_summary=final_summary,
        frame_idx=frame_idx,
        current_alpha=(actor_lr, critic_lr),
        epsilon=0.0,  # Not using epsilon in this implementation
        iter_no=int(iter_no)
    )
