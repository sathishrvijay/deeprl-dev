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
    gae_lambda: float
    ):
    """Unpack batch and compute GAE (Generalized Advantage Estimation) advantages.
    
    GAE computes advantages using:
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
    where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error
    
    This provides a better bias-variance tradeoff compared to simple TD advantages.
    Modified for continuous actions - actions are now continuous values instead of discrete indices.
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
        deltas = rewards_v + next_values_v - current_values_v

        # Compute GAE advantages using backward pass
        advantages_v = torch.zeros_like(deltas)
        gae_v = 0
        
        # Work backwards through the batch to compute GAE
        for t in reversed(range(len(deltas))):
            if done_masks_v[t]:
                # Reset GAE at episode boundaries
                gae_v = 0
            
            gae_v = deltas[t] + gamma * gae_lambda * gae_v
            advantages_v[t] = gae_v

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
