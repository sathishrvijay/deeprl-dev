import gymnasium as gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
from dataclasses import dataclass
import pdb

import typing as tt
from functools import partial

from models import agents, pgtr_models
from utils import PerformanceTracker, print_training_header, print_final_summary
from utils.common_utils import unpack_batch, unpack_batch_with_gae

"""This is the implementation of PPO with MountainCarContinuous-v0 RL using the PTAN wrapper libraries.
PPO implementation can use the exact same Actor and Critic networks and Agent as Continuous A2C. Only changes are:
- uses GAE advantage computation instead of simple TD
- uses multiple epochs of training per generated batch
- uses surrogate loss instead of policy gradient
Note: During training, adjust the learning rates for actor and critic so that advantage ratios are close to 1.0.
"""

# HPARAMS
RL_ENV = "MountainCarContinuous-v0"
N_ENVS = 16

# Separate network dimensions
CRITIC_HIDDEN1_DIM = 128
CRITIC_HIDDEN2_DIM = 32
ACTOR_HIDDEN1_DIM = 128
ACTOR_HIDDEN2_DIM = 64

N_TD_STEPS = 1 # Number of steps aggregated per experience, not useful with PPO since we use GAE
N_ROLLOUT_STEPS = 128 # formal rollout definition; batch_size = N_ENVS * N_ROLLOUT_STEPS = 2048

GAMMA = 0.99  # Good for MountainCarContinuous episodes (~200 steps)
# Balanced learning rates for PPO - critic and actor should be similar
CRITIC_LR_START = 1e-6  # Reduced from 7e-4 for better stability
CRITIC_LR_END = 5e-7    # Proportionally reduced
ACTOR_LR_START = 3e-6
ACTOR_LR_END = 7e-7
LR_DECAY_FRAMES = 5e7

# PG related - adjusted for continuous actions
ENTROPY_BONUS_BETA_START = 3e-3  # Higher initial exploration
ENTROPY_BONUS_BETA_END = 1e-5    # Lower final exploration
ENTROPY_DECAY_FRAMES = 1e7       # Extended from 1e6 to 1e7 for longer exploration period
CLIP_GRAD = 0.3   # typical values of clipping the L2 norm is 0.1 to 1.0

# MountainCarContinuous success threshold (one time +100 bonus for climbing the hill w/ small energy penalty otherwise)
SOLVED_REWARD = 90.0  

# PPO specific hyperparameters - OPTIMIZED FOR EFFICIENCY
MAX_TRAINING_FRAMES = 5e7
MINIBATCH_SIZE = 256                # Increased from 64 for better efficiency (8 minibatches instead of 32)
MAX_EPOCHS_PER_BATCH = 4            # REDUCED from 256 to 4 - standard PPO range
GAE_LAMBDA = 0.95                   # GAE lambda parameter for advantage estimation (0.9-0.99 typical)
PPO_CLIP_EPSILON = 0.2              # PPO clipping parameter - now parameterized instead of hardcoded
PPO_TARGET_KL = 0.015               # Target KL divergence for early stopping (standard: 0.01-0.02)


@dataclass
class PPOBatch:
    """Container for all batch data needed for PPO training. 
    Note: Unlike other algorithms, PPO needs to keep track of old log probabilities and values for 
    the surrogate loss calculation. Computing and storing once per batch is much more efficient than
    recomputing for every minibatch since the data is reused for multiple epochs.
    
    For proper KL divergence computation, we also store the old policy distribution parameters (mu, logvar)
    which are captured before any network updates."""
    states_v: torch.Tensor
    actions_v: torch.Tensor
    old_logprobas_v: torch.Tensor
    old_mu_v: torch.Tensor
    old_logvar_v: torch.Tensor
    old_values_v: torch.Tensor
    target_returns_v: torch.Tensor
    advantages_v: torch.Tensor
    
    def __len__(self):
        return len(self.states_v)
    
    def sample_minibatch(self, minibatch_size: int):
        """Sample a random minibatch from this batch data"""
        batch_size = len(self)
        if batch_size <= minibatch_size:
            # Return all indices if batch is smaller than minibatch size
            indices = torch.arange(batch_size)
        else:
            # Randomly sample indices without replacement
            indices = torch.randperm(batch_size)[:minibatch_size]
        
        return PPOBatch(
            states_v=self.states_v[indices],
            actions_v=self.actions_v[indices],
            old_logprobas_v=self.old_logprobas_v[indices],
            old_mu_v=self.old_mu_v[indices],
            old_logvar_v=self.old_logvar_v[indices],
            old_values_v=self.old_values_v[indices],
            target_returns_v=self.target_returns_v[indices],
            advantages_v=self.advantages_v[indices]
        )


def prepare_ppo_batch(batch: tt.List[ptan.experience.ExperienceFirstLast], net: nn.Module) -> PPOBatch:
    """Convert raw experience batch into structured PPOBatch for PPO training.
    This computes all necessary data once per batch collection, including old policy
    log probabilities and values that are needed for PPO's surrogate loss calculation.
    
    PPO uses GAE vs classical TD for better advantage computation.
    
    Args:
        batch: List of experiences collected from parallel environments
        net: The policy network (before any updates in this training iteration)
        
    Returns:
        PPOBatch containing all preprocessed data ready for multiple epochs of training
    """
    # Unpack the raw experience batch using GAE with parallel environment support
    states_v, actions_v, target_returns_v, advantages_v = unpack_batch_with_gae(
        batch, net, N_TD_STEPS, GAMMA, GAE_LAMBDA, n_envs=N_ENVS
    )
    
    # Compute old policy data (before any parameter updates)
    # This is crucial for PPO - we need the log probabilities, distribution parameters (mu, logvar),
    # and values from the policy that was used to collect the experiences
    with torch.no_grad():
        old_logproba_v, old_mu_v, old_logvar_v, old_values_v = net(states_v)
        old_values_v = old_values_v.squeeze(-1)
        
        # Normalize GAE advantages (standard practice for PPO/A2C)
        # This helps with training stability and convergence
        adv_std = max(1e-3, (advantages_v.std(unbiased=False) + 1e-8).item())
        advantages_v = (advantages_v - advantages_v.mean()) / adv_std
    
    return PPOBatch(
        states_v=states_v,
        actions_v=actions_v,
        old_logprobas_v=old_logproba_v,
        old_mu_v=old_mu_v,
        old_logvar_v=old_logvar_v,
        old_values_v=old_values_v,
        target_returns_v=target_returns_v,
        advantages_v=advantages_v
    )


def core_training_loop(
    net: nn.Module,
    minibatch: PPOBatch,
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    frame_idx: int,
    iter_no=None,
    clip_epsilon=PPO_CLIP_EPSILON,  # Now uses the hyperparameter instead of hardcoded value
    debug=True
    ):
    """PPO training step using structured minibatch data.
    Key differences from A2C:
    1. Uses clipped surrogate loss instead of standard policy gradient
    2. Uses pre-computed old log probabilities for importance sampling ratio
    3. Uses pre-computed advantages from PPOBatch
    Returns: Dictionary containing loss components for tracking
    """
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    # Get current policy outputs for the minibatch
    logproba_actions_v, actions_mu_v, actions_logvar_v, values_v = net(minibatch.states_v)
    values_v = values_v.squeeze(-1)

    # Compute current entropy bonus with decay
    entropy_progress = min(1.0, frame_idx / ENTROPY_DECAY_FRAMES)
    current_entropy_beta = ENTROPY_BONUS_BETA_START * (1 - entropy_progress) + ENTROPY_BONUS_BETA_END * entropy_progress

    # Critic loss - same as A2C
    critic_loss_v = nn.functional.mse_loss(values_v, minibatch.target_returns_v)

    # Compute importance sampling ratio (per minibatch observation)
    ratios = torch.exp(logproba_actions_v - minibatch.old_logprobas_v)
    
    # PPO clipped surrogate loss
    surr1 = ratios * minibatch.advantages_v
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * minibatch.advantages_v
    
    # Policy loss is negative of minimum (we want to maximize)
    pg_loss_v = -torch.min(surr1, surr2).mean()
    
    # Compute true KL divergence for diagonal Gaussian policies (for early stopping)
    # Use the stored old policy distribution parameters from PPOBatch (captured before any network updates)
    with torch.no_grad():
        # KL divergence between two diagonal Gaussians:
        # KL(old || new) = 0.5 * sum_i [ (σ²_old / σ²_new) + (μ_new - μ_old)² / σ²_new - 1 + log(σ²_new / σ²_old) ]
        # With log variance: σ² = exp(log_var)
        kl_per_dim = 0.5 * (
            torch.exp(minibatch.old_logvar_v - actions_logvar_v) +  # σ²_old / σ²_new
            (actions_mu_v - minibatch.old_mu_v).pow(2) / torch.exp(actions_logvar_v) +  # (μ_new - μ_old)² / σ²_new
            actions_logvar_v - minibatch.old_logvar_v - 1.0  # log(σ²_new / σ²_old) - 1
        )
        kl_divergence = kl_per_dim.sum(dim=-1).mean().item()  # Sum over action dims, mean over batch
    
    # Entropy bonus for continuous actions
    # Gaussian entropy: 0.5 * log(2πe * σ²) = 0.5 * (log(2πe) + log_var)
    entropy_v = 0.5 * (actions_logvar_v + math.log(2 * math.pi * math.e))
    entropy_v = entropy_v.sum(dim=-1).mean()  # Sum over action dims, mean over batch
    entropy_bonus_v = current_entropy_beta * entropy_v

    # Variance regularization to prevent excessive exploration (reduced penalty)
    var_penalty_v = 1e-5 * torch.exp(actions_logvar_v).mean()
    
    # Action regularization to prevent extreme actions outside reasonable bounds [-0.8, 0.8]
    action_reg_v = 1e-4 * torch.clamp(actions_mu_v.abs() - 0.8, min=0.0).mean()

    # Separate loss computation and backpropagation
    # Critic loss (only affects critic network)
    critic_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(net.get_critic_parameters(), CLIP_GRAD)
    critic_optimizer.step()
    critic_scheduler.step()

    # Actor loss (only affects actor network)
    actor_loss_v = pg_loss_v - entropy_bonus_v + var_penalty_v + action_reg_v
    actor_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(net.get_actor_parameters(), CLIP_GRAD)
    actor_optimizer.step()
    actor_scheduler.step()

    # Total loss for logging (not used for backprop)
    total_loss_v = critic_loss_v + actor_loss_v

    # Debug - check policy statistics
    if debug and iter_no is not None and iter_no % 100 == 0:
        with torch.no_grad():
            mean_action = actions_mu_v.mean(dim=0).cpu().numpy()
            std_action = torch.exp(actions_logvar_v / 2).mean(dim=0).cpu().numpy()
            mean_ratio = ratios.mean().item()
            print(f"Action μ={mean_action[0]:.3f}, σ={std_action[0]:.3f}, "
                  f"ratio={mean_ratio:.3f}, kl={kl_divergence:.4f}, "
                  f"entropy_β={current_entropy_beta:.4e}")

    # Return loss components for tracking
    return {
        'total_loss': total_loss_v.item(),
        'critic_loss': critic_loss_v.item(),
        'actor_loss': actor_loss_v.item(),
        'pg_loss': pg_loss_v.item(),
        'entropy_raw': entropy_v.item(),
        'entropy_loss': entropy_bonus_v.item(),
        'var_penalty': var_penalty_v.item(),
        'action_reg': action_reg_v.item(),
        'current_entropy_beta': current_entropy_beta,
        'kl_divergence': kl_divergence,
        'mean_ratio': ratios.mean().item(),
        'ratio_clipped_fraction': ((ratios < 1.0 - clip_epsilon) | (ratios > 1.0 + clip_epsilon)).float().mean().item()
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
    obs_space = vector_env.single_observation_space
    action_space = vector_env.single_action_space
    n_states = obs_space.shape[0] if obs_space.shape is not None else 2  # MountainCarContinuous has Box(2,) observation space
    n_actions = action_space.shape[0] if action_space.shape is not None else 1      # MountainCarContinuous has Box(1,) action space
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
    average_return = -100.0  # Initialize to handle potential unbound variable

    # Separate optimizers for actor and critic
    # PPO typically uses Adam for both actor and critic with balanced learning rates
    critic_optimizer = optim.Adam(net.get_critic_parameters(), lr=CRITIC_LR_START, eps=1e-5)
    actor_optimizer = optim.Adam(net.get_actor_parameters(), lr=ACTOR_LR_START, eps=1e-5)

    max_return = -100.0

    # each iteration of exp_source yields N_ENVS experiences
    batch_size = N_ROLLOUT_STEPS * N_ENVS

    # Separate schedulers for actor and critic
    critic_scheduler = optim.lr_scheduler.LinearLR(
        critic_optimizer, start_factor=1.0, end_factor=CRITIC_LR_END / CRITIC_LR_START,
        total_iters=int(LR_DECAY_FRAMES // batch_size))
    actor_scheduler = optim.lr_scheduler.LinearLR(
        actor_optimizer, start_factor=1.0, end_factor=ACTOR_LR_END / ACTOR_LR_START,
        total_iters=int(LR_DECAY_FRAMES // batch_size))

    # Initialize performance tracker and print training header
    perf_tracker = PerformanceTracker()
    network_config = f"PPO: Critic({CRITIC_HIDDEN1_DIM}-{CRITIC_HIDDEN2_DIM}), Actor({ACTOR_HIDDEN1_DIM}-{ACTOR_HIDDEN2_DIM})"
    hyperparams = {
        'n_envs': N_ENVS,
        'n_td_steps': N_TD_STEPS,
        'n_rollout_steps': N_ROLLOUT_STEPS,
        'batch_size': N_ENVS * N_ROLLOUT_STEPS,
        'minibatch_size': MINIBATCH_SIZE,
        'max_epochs_per_batch': MAX_EPOCHS_PER_BATCH,
        'ppo_clip_epsilon': PPO_CLIP_EPSILON,
        'ppo_target_kl': PPO_TARGET_KL,
        'critic_lr_start': CRITIC_LR_START,
        'critic_lr_end': CRITIC_LR_END,
        'actor_lr_start': ACTOR_LR_START,
        'actor_lr_end': ACTOR_LR_END,
        'lr_decay_frames': LR_DECAY_FRAMES,
        'gamma': GAMMA,
        'gae_lambda': GAE_LAMBDA,
        'entropy_beta_start': ENTROPY_BONUS_BETA_START,
        'entropy_beta_end': ENTROPY_BONUS_BETA_END,
        'entropy_decay_frames': ENTROPY_DECAY_FRAMES,
        'clip_grad': CLIP_GRAD,
        'solved_threshold': SOLVED_REWARD
    }
    print_training_header(RL_ENV, network_config, hyperparams)

    batch = []
    exp_iterator = iter(exp_source)
    frame_idx = 0  # Track total frames of experience trained on
    while not solved and iter_no <= MAX_TRAINING_FRAMES:
        iter_no += 1.0
        iter_start_time = time.time()

        # Collect experiences from parallel envs for training
        while len(batch) < batch_size:
            exp = next(exp_iterator)
            batch.append(exp)

        # Prepare structured batch data for PPO training
        ppo_batch = prepare_ppo_batch(batch, net)
        
        # Perform multiple epochs of training per batch (now much more reasonable: 4 epochs instead of 256)
        num_epochs_per_batch = 0
        # Initialize with fallback values to ensure loss_dict is never None
        loss_dict = {
            'total_loss': 0.0, 'critic_loss': 0.0, 'actor_loss': 0.0, 'pg_loss': 0.0,
            'entropy_raw': 0.0, 'entropy_loss': 0.0, 'var_penalty': 0.0, 'action_reg': 0.0,
            'current_entropy_beta': 0.0, 'kl_divergence': 0.0, 'mean_ratio': 1.0, 'ratio_clipped_fraction': 0.0
        }
        
        for epoch in range(MAX_EPOCHS_PER_BATCH):
            # Sample a random minibatch from the structured batch data (now 256 instead of 64)
            minibatch = ppo_batch.sample_minibatch(MINIBATCH_SIZE)
            
            # Training step with PPO minibatch - overwrites loss_dict
            loss_dict = core_training_loop(
                net, minibatch, actor_optimizer, critic_optimizer,
                actor_scheduler, critic_scheduler, frame_idx, iter_no
            )
            
            num_epochs_per_batch += 1
            frame_idx += len(minibatch)
            
            # Early stopping if KL divergence exceeds target (prevents policy from changing too much)
            # Standard PPO practice: stop if KL > target_kl after at least 1 epoch
            if epoch >= 1 and 'kl_divergence' in loss_dict and loss_dict['kl_divergence'] > PPO_TARGET_KL:
                print(f"iter: {iter_no} - Early stopping at epoch {epoch} due to KL divergence {loss_dict['kl_divergence']:.4f} > target {PPO_TARGET_KL}")
                break

        batch.clear()  # Clear batch after multiple epochs of training

        training_time = time.time() - iter_start_time

        # Print periodic performance summary every 10000 iterations
        if iter_no % 10000 == 0:
            perf_tracker.print_checkpoint(int(iter_no), frame_idx)

        # Test trials to check success condition
        eval_start_time = time.time()
        average_return = play_trials(test_env, net)
        eval_time = time.time() - eval_start_time

        max_return = average_return if (max_return < average_return) else max_return
        trial += 1

        # Log performance metrics - loss_dict is now guaranteed to be defined
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
                f"pg_loss={loss_dict['pg_loss']:.4f}, "
                f"entropy_raw={loss_dict['entropy_raw']:.4f}, "
                f"entropy_bonus={loss_dict['entropy_loss']:.4e}, "
                f"var_penalty={loss_dict['var_penalty']:.4e}, "
                f"action_reg={loss_dict['action_reg']:.4e}, "
                f"kl_div={loss_dict['kl_divergence']:.4f}, "
                f"mean_ratio={loss_dict['mean_ratio']:.3f}, "
                f"clipped_frac={loss_dict['ratio_clipped_fraction']:.3f}, "
                f"entropy_β={loss_dict['current_entropy_beta']:.4e}"
            )

        # Check for training termination
        solved = average_return > SOLVED_REWARD  # MountainCarContinuous is solved when avg reward > 90


    # Training completed - print comprehensive summary using utility function
    final_summary = perf_tracker.get_summary()
    critic_lr = critic_optimizer.param_groups[0]["lr"]
    actor_lr = actor_optimizer.param_groups[0]["lr"]
    print_final_summary(
        solved=solved,
        average_return=average_return,
        target_reward=SOLVED_REWARD,
        final_summary=final_summary,
        frame_idx=frame_idx,
        current_alpha=(actor_lr, critic_lr),
        epsilon=0.0,  # Not using epsilon in this implementation
        iter_no=int(iter_no)
    )
