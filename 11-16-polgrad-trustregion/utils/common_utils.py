import typing as tt
import torch
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
