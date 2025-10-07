import typing as tt
import torch
import torch.nn as nn
import numpy as np
import ptan


def unpack_batch_ddpg_sac(batch: tt.List[ptan.experience.ExperienceFirstLast]):
    """
    Note: unpacking batch is different for DDQN/SAC because Q(s', a') and therefore
    target return is only available online during training
    Note: Since, in general, an experience sub-trajectory can be n-steps,
    the terminology used here is last state instead of next state.
    Additionally, reward is equal to the cumulative discounted rewards from
    intermediate steps. All of this is subsumed within ptan.experience.ExperienceFirstLast
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
    done_masks_v = torch.tensor(done_masks, dtype=torch.bool)

    # Note: for DDPG/SAC, Q(s', a') cannot be computed before a' is known from Actor
    # Hence we return the partial return (aka rewards_v) at this point
    return states_v, actions_v, rewards_v, last_states_v, done_masks_v


def unpack_batch_with_gae(batch: tt.List[ptan.experience.ExperienceFirstLast],
    net: nn.Module,
    n_steps: int,
    gamma: float,
    gae_lambda: float,
    n_envs: int = 1
    ):
    """Unpack batch and compute GAE (Generalized Advantage Estimation) advantages.
    
    GAE computes advantages using:
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
    where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error
    
    This provides a better bias-variance tradeoff compared to simple TD advantages.
    Modified for continuous actions - actions are now continuous values instead of discrete indices.
    
    When using parallel environments (n_envs > 1), the batch is expected to be ordered as:
    [env0_t0, env1_t0, ..., envN_t0, env0_t1, env1_t1, ..., envN_t1, ...]
    This ordering is produced by ptan.experience.VectorExperienceSourceFirstLast.
    
    Args:
        batch: List of experiences from parallel environments (interleaved by timestep)
        net: Network for computing value estimates
        n_steps: Number of steps for n-step returns
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        n_envs: Number of parallel environments (default=1 for single env)
    """
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    
    # First pass: collect all data
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
        done_masks.append(exp.last_state is None)

    # Convert to tensors
    actions_v = torch.tensor(np.stack(actions), dtype=torch.float32)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    states_v = torch.tensor(np.stack(states), dtype=torch.float32)
    last_states_v = torch.tensor(np.stack(last_states), dtype=torch.float32)
    done_masks_v = torch.tensor(done_masks, dtype=torch.bool)

    # Compute state values for GAE
    with torch.no_grad():
        # Current state values
        _, _, _, current_values_v = net(states_v)
        current_values_v = current_values_v.squeeze(-1)
        
        # Next state values 
        _, _, _, next_values_v = net(last_states_v)
        next_values_v = next_values_v.squeeze(-1)
        
        # Apply n-step discount to next values
        next_values_v *= gamma ** n_steps
        # Zero out terminated episodes
        next_values_v[done_masks_v] = 0.0

        # Compute TD errors (deltas)
        deltas_v = rewards_v + next_values_v - current_values_v

        # Compute GAE advantages
        # When using parallel environments, we need to compute GAE separately for each environment's trajectory
        advantages_v = torch.zeros_like(deltas_v)
        
        if n_envs > 1:
            # Verify batch size is compatible with n_envs
            batch_size = len(deltas_v)
            if batch_size % n_envs != 0:
                raise ValueError(f"Batch size {batch_size} is not divisible by n_envs {n_envs}. "
                                f"Expected batch collected from VectorExperienceSourceFirstLast with "
                                f"consistent environment count.")
            
            n_rollout_steps = batch_size // n_envs
            
            # Reshape to separate parallel environment trajectories
            # Input ordering: [env0_t0, env1_t0, ..., envN_t0, env0_t1, env1_t1, ..., envN_t1, ...]
            # After reshape: (n_rollout_steps, n_envs) where:
            #   - Each row represents one timestep across all environments
            #   - Each column represents one environment's trajectory over time
            deltas_per_env_v = deltas_v.view(n_rollout_steps, n_envs)
            done_masks_per_env_v = done_masks_v.view(n_rollout_steps, n_envs)
            advantages_per_env_v = torch.zeros_like(deltas_per_env_v)
            
            # Compute GAE separately for each environment (working backwards in time)
            # gae_v maintains one accumulator per environment (shape: n_envs) and computes GAE 
            # in parallel for each environment in an efficient vectorized way by reshaping to (n_rollout_steps, n_envs)
            gae_v = torch.zeros(n_envs)
            for t in reversed(range(n_rollout_steps)):
                # Compute GAE: A_t = δ_t + (γλ) * A_{t+1}
                # This is vectorized across all environments
                gae_v = deltas_per_env_v[t] + gamma * gae_lambda * gae_v
                advantages_per_env_v[t] = gae_v
                
                # Reset GAE at episode boundaries (after computing advantage for terminal state)
                # When working backwards, if current state is terminal, reset GAE for the previous timestep
                # This ensures each episode's GAE computation is independent
                gae_v = gae_v * (~done_masks_per_env_v[t]).float()
            
            # Flatten back to original shape (batch_size,)
            advantages_v = advantages_per_env_v.view(-1)
        else:
            # Single environment case - original sequential logic
            gae_v = 0
            for t in reversed(range(len(deltas_v))):
                # Compute GAE for current timestep
                gae_v = deltas_v[t] + gamma * gae_lambda * gae_v
                advantages_v[t] = gae_v
                
                # Reset GAE at episode boundaries
                if done_masks_v[t]:
                    gae_v = 0

        # Compute GAE returns = advantages + current values
        returns_v = advantages_v + current_values_v

    return states_v, actions_v, returns_v, advantages_v


def unpack_batch(batch: tt.List[ptan.experience.ExperienceFirstLast],
    net: nn.Module,
    n_steps: int,
    gamma: float
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
        _, _, _, ls_values_v = net(last_states_v)

    ls_values_v = ls_values_v.squeeze(1).data
    # Note: important step for computing returns correctly w/ loop unroll
    ls_values_v *= gamma ** n_steps
    # zero out the terminated episodes
    done_masks_v = torch.tensor(done_masks, dtype=torch.bool)
    ls_values_v[done_masks_v] = 0.0

    returns_v = rewards_v + ls_values_v
    return states_v, actions_v, returns_v
