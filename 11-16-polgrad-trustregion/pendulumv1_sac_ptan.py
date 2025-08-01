import gymnasium as gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

import typing as tt
from functools import partial

from models import agents, pgtr_models
from utils import PerformanceTracker, print_training_header, print_final_summary

"""This is the implementation of SAC with Pendulum-v1 RL using the PTAN wrapper libraries.
Can reuse a lot of existing modules:
- reparametrization trick: for sampling actions for experience and training (used in CA A2C)
- Critc net: can reuse DDPG critic; GPT recommends 2 separate critic nets w/ shared backbone
- Critic schedulers and optimizers: Separate per critic net due to above
- Actor net: can reuse A2C actor (outputs mu and logvar for actions)
- Target nets: only critics use target net
- SAC model: Basically same as A2C model, but needs 2 critics instead of 1; already has the
reparametrization trick implemented
- SAC agent: can reuse the A2C agent since OU noise is not required
- unpack_batch: can use same as DPPG since it uses Q-values
- loss func changes (straightforward changes from DDPG)
- N_TD_STEPS: Since SAC is expected SARSA, rollout is fine for experience collection
"""

# HPARAMS
RL_ENV = "Pendulum-v1"
N_ENVS = 8

# Separate network dimensions
CRITIC_HIDDEN1_DIM = 128
CRITIC_HIDDEN2_DIM = 32
ACTOR_HIDDEN1_DIM = 128
ACTOR_HIDDEN2_DIM = 64

N_TD_STEPS = 4 # Number of steps aggregated per experience (n in n-step TD)
N_ROLLOUT_STEPS = 16 # formal rollout definition; batch_size = N_ENVS * N_ROLLOUT_STEPS

GAMMA = 0.99  # Slightly lower for Pendulum's shorter episodes
# Separate learning rates for actor and critic - reduced for stability
CRITIC_LR_START = 1e-3
CRITIC_LR_END = 1e-4
ACTOR_LR_START = 3e-5
ACTOR_LR_END = 1e-5
LR_DECAY_FRAMES = int(5e7)

# PG related - adjusted for continuous actions
CLIP_GRAD = 0.3   # typical values of clipping the L2 norm is 0.1 to 1.0

# Pendulum success threshold
RNORM_SCALE_FACTOR = 5.0  # reward normalization scale factor
SAC_ENTROPYTEMP_COEF = 0.1  # depends on action space dim, set lower if dim increases
# ACTION_PENALTY_COEF = 0.01   #prevents actor from going to max torque
PENDULUM_SOLVED_REWARD = -200.0  # Pendulum is solved when avg reward > -200

# Replay buffer related
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 5000
BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP = 50


def unpack_batch_ddqn(batch: tt.List[ptan.experience.ExperienceFirstLast]):
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


def critic_training_pass(tgt_net, critic_id: int, optimizers, schedulers,
    states_v, actions_v, target_return_v
    ):
    """TODO: Add entropy bonus term"""
    assert critic_id in [1, 2]
    critic_net = tgt_net.model.get_critic_net(critic_id)
    optimizer, scheduler = optimizers[critic_id-1], schedulers[critic_id-1]
    optimizer.zero_grad()

    qvalues_v = critic_net(states_v, actions_v)
    # Note: Entropy Bonus term in return target also scales same as original target
    critic_loss_v = nn.functional.mse_loss(qvalues_v.squeeze(-1) / RNORM_SCALE_FACTOR,
                                           target_return_v / RNORM_SCALE_FACTOR)

    # Safety check for NaN/Inf values
    if torch.isnan(critic_loss_v) or torch.isinf(critic_loss_v):
        # breakpoint()
        print(f"WARNING: critic_id={critic_id} Invalid critic loss detected: {critic_loss_v.item()}")
        print(f"Q-values: min={qvalues_v.min().item():.4f}, max={qvalues_v.max().item():.4f}")
        # print(f"Targets: min={target_return_v.min():.4f}, max={target_return_v.max():.4f}")

    # Critic loss (only affects critic network)
    critic_params = tgt_net.model.get_critic_parameters(critic_id)
    critic_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(critic_params, CLIP_GRAD)
    optimizer.step()
    scheduler.step()

    return critic_loss_v


def core_training_loop(
    tgt_net: ptan.agent.TargetNet,
    replay_buffer: ptan.experience.ExperienceReplayBuffer,
    actor_optimizer,
    critic_optimizers,
    actor_scheduler,
    critic_schedulers,
    frame_idx: int,
    iter_no=None,
    reward_norm=True,
    debug=True
    ):
    """DDPG/SAC samples mini-batches from Replay Buffer because it is Off Policy PG
    Returns: Dictionary containing loss components for tracking
    """
    batch = replay_buffer.sample(BATCH_SIZE)
    states_v, actions_v, target_return_v, last_states_v, done_masks_v = \
        unpack_batch_ddqn(batch)

    # Note: Separate loss computation and backpropagation for Critic and Actor
    # Critic loss TODO: MSE(r + gamma * QT(s', muT(s)) - Q(s, a)); T <- Target net
    # s' <- from collected experience
    # TODO: move this piece into critic training pass as well
    target_net = tgt_net.target_model
    with torch.no_grad():
        ls_logproba_actions_v, ls_qv1_v, ls_qv2_v = target_net(last_states_v)
        ls_qv1_v = (GAMMA ** N_TD_STEPS) * ls_qv1_v.squeeze(1).data
        ls_qv2_v = (GAMMA ** N_TD_STEPS) * ls_qv2_v.squeeze(1).data
        # zero out if s' is a terminal state
        ls_qv1_v[done_masks_v], ls_qv2_v[done_masks_v] = 0.0, 0.0
        target_return_v += torch.min(ls_qv1_v, ls_qv2_v)
        # Add Entropy bonus to Target
        # logproba_actions_v =
        target_return_v -= SAC_ENTROPYTEMP_COEF * ls_logproba_actions_v.squeeze(-1).data

    # Q(s, a) <- from collected experience on main net (because of partial SGD)
    critic_loss_v = torch.tensor(0.0)
    for critic_id in [1, 2]:
        critic_loss_v += critic_training_pass(tgt_net, critic_id,
            critic_optimizers, critic_schedulers,
            states_v, actions_v, target_return_v)

    # Actor loss:
    # In SAC, w/ 2 critics, by default first critic is used
    # Sample action from Actor w/ reparametrization trick for differentiability
    actor_optimizer.zero_grad()
    critic_params = list(tgt_net.model.get_critic_parameters(critic_id=1))
    for p in critic_params:
        p.requires_grad_(False)

    logproba_actions_v, qvalues1_v, _ = tgt_net.model(states_v)
    # Invert sign to gradient ascent
    actor_loss_v = -(qvalues1_v - SAC_ENTROPYTEMP_COEF * logproba_actions_v) / RNORM_SCALE_FACTOR
    actor_loss_v = actor_loss_v.squeeze(-1).mean()

    # NOTE: Add action penalty to limit to smaller torque values and avoid reward hacking
    # actor_loss_v += ACTION_PENALTY_COEF * (current_actions_v ** 2).mean()

    actor_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(tgt_net.model.get_actor_parameters(), CLIP_GRAD)
    actor_optimizer.step()
    actor_scheduler.step()

    # unfreeze critic
    for p in critic_params:
        p.requires_grad_(True)

    # Total loss for logging (not used for backprop)
    total_loss_v = critic_loss_v + actor_loss_v

    # Smooth blend target net parameters in continuous action for training stability
    # wT = (1 - tau) *wT + tau *w
    # literature calls this param tau; GPT suggests tau <- 0.005
    # ptan uses alpha = 1 - tau
    tgt_net.alpha_sync(alpha=0.995)

    # # Debug - check policy statistics and loss values
    # if debug and iter_no % 100 == 0:
    #     with torch.no_grad():
    #         current_actions_v = current_actions_v.mean(dim=0).cpu().numpy()
    #         print(f"Action μ={current_actions_v[0]:.3f}, "
    #               f"Q-val range=[{qvalues_v.min():.2f}, {qvalues_v.max():.2f}], "
    #               f"Target range=[{target_return_v.min():.2f}, {target_return_v.max():.2f}]")

    # Return loss components for tracking
    return {
        'total_loss': total_loss_v.item(),
        'critic_loss': critic_loss_v.item(),
        'actor_loss': actor_loss_v.item(),
    }


def play_trials(test_env: gym.Env, test_agent: ptan.agent.BaseAgent) -> float:
    """Note that we want a separate env for trials that doesn't mess with training env.
    We use deterministic actions (mean only) for evaluation to get consistent performance measurement.
    Modified for continuous actions.
    """
    _, _ = test_env.reset()
    exp_source = ptan.experience.ExperienceSourceFirstLast(test_env, test_agent, gamma=GAMMA)
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

    # setup the agent and target net
    n_states = vector_env.single_observation_space.shape[0]  # Pendulum has Box(3,) observation space
    n_actions = vector_env.single_action_space.shape[0]      # Pendulum has Box(1,) action space
    net = pgtr_models.SAC(
        n_states, n_actions,
        CRITIC_HIDDEN1_DIM, CRITIC_HIDDEN2_DIM,
        ACTOR_HIDDEN1_DIM, ACTOR_HIDDEN2_DIM
    )

    # Note: we don't actually need target net for Actor
    tgt_net = ptan.agent.TargetNet(net)

    # Setup the agent, action policy, experience generation and experience buffer
    agent = agents.AgentContinuousA2C(net)
    exp_source = ptan.experience.VectorExperienceSourceFirstLast(vector_env, agent, gamma=GAMMA,
        steps_count=N_TD_STEPS)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_BUFFER_SIZE)

    # don't need a vectorized trial env
    test_env = gym.make(RL_ENV)
    test_agent = agents.AgentContinuousA2C(net, deterministic=True)

    # Initialize training with performance tracking
    iter_no = 0.0
    trial = 0
    solved = False
    max_return = -1000.0

    # each iteration of exp_source yields N_ENVS experiences
    batch_size = N_ROLLOUT_STEPS * N_ENVS

    # Separate optimizers & schedulers for actor and critics
    # A2C reports more stable results w/ RMSProp for critic, Adam for actor
    critic1_optimizer = optim.RMSprop(net.get_critic_parameters(), lr=CRITIC_LR_START,
        eps=1e-5, alpha=0.99)
    critic2_optimizer = optim.RMSprop(net.get_critic_parameters(critic_id=2), lr=CRITIC_LR_START,
        eps=1e-5, alpha=0.99)
    actor_optimizer = optim.Adam(net.get_actor_parameters(), lr=ACTOR_LR_START, eps=1e-5)

    critic1_scheduler = optim.lr_scheduler.LinearLR(
        critic1_optimizer, start_factor=1.0, end_factor=CRITIC_LR_END / CRITIC_LR_START,
        total_iters=LR_DECAY_FRAMES // batch_size)
    critic2_scheduler = optim.lr_scheduler.LinearLR(
        critic2_optimizer, start_factor=1.0, end_factor=CRITIC_LR_END / CRITIC_LR_START,
        total_iters=LR_DECAY_FRAMES // batch_size)
    actor_scheduler = optim.lr_scheduler.LinearLR(
        actor_optimizer, start_factor=1.0, end_factor=ACTOR_LR_END / ACTOR_LR_START,
        total_iters=LR_DECAY_FRAMES // batch_size)

    # Initialize performance tracker and print training header
    perf_tracker = PerformanceTracker()
    network_config = f"SAC: Critic({CRITIC_HIDDEN1_DIM}-{CRITIC_HIDDEN2_DIM}), Actor({ACTOR_HIDDEN1_DIM}-{ACTOR_HIDDEN2_DIM})"
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
        'clip_grad': CLIP_GRAD,
        'solved_threshold': PENDULUM_SOLVED_REWARD
    }
    print_training_header(RL_ENV, network_config, hyperparams)

    exp_iterator = iter(exp_source)
    frame_idx = 0  # Track total frames of experience generated
    # warm start for replay buffer
    replay_buffer.populate(REPLAY_BUFFER_SIZE)
    while not solved:
        iter_no += 1.0
        iter_start_time = time.time()
        # Collect experiences every iteration
        replay_buffer.populate(BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP)
        frame_idx += BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP

        # Training step with separate optimizers
        loss_dict = core_training_loop(
            tgt_net, replay_buffer,
            actor_optimizer, [critic1_optimizer, critic2_optimizer],
            actor_scheduler, [critic1_scheduler, critic2_scheduler],
            frame_idx, iter_no
        )
        training_time = time.time() - iter_start_time

        # Print periodic performance summary every 500 iterations
        if iter_no % 2000 == 0:
            print("WARNING: RESETTING BUFFER!!")
            replay_buffer.populate(REPLAY_BUFFER_SIZE)
            perf_tracker.print_checkpoint(int(iter_no), frame_idx)

        # Test trials to check success condition
        eval_start_time = time.time()
        average_return = play_trials(test_env, test_agent)
        eval_time = time.time() - eval_start_time

        max_return = average_return if (max_return < average_return) else max_return
        trial += 1

        # Log performance metrics
        perf_metrics = perf_tracker.log_iteration(
            int(iter_no), frame_idx, average_return, training_time, eval_time, loss_dict
        )

        if iter_no % 100 == 0:
            # Enhanced logging with timing information and separate learning rates
            critic1_lr = critic1_optimizer.param_groups[0]["lr"]
            critic2_lr = critic2_optimizer.param_groups[0]["lr"]
            actor_lr = actor_optimizer.param_groups[0]["lr"]
            print(f"(iter: {iter_no:6.0f}, trial: {trial:4d}) - "
                f"avg_return={average_return:7.2f}, max_return={max_return:7.2f} | "
                f"critic1_lr={critic1_lr:.5e}, critic2_lr={critic2_lr:.5e}, actor_lr={actor_lr:.5e}, "
                f"train_time={training_time:.3f}s, eval_time={eval_time:.3f}s, "
                f"fps={perf_metrics['current_fps']:.1f}, total_time={perf_metrics['total_elapsed']/60:.1f}m | "
                f"losses: total={loss_dict['total_loss']:.4f}, "
                f"critic={loss_dict['critic_loss']:.4f}, "
                f"actor={loss_dict['actor_loss']:.4f}, "
            )

        solved = average_return > PENDULUM_SOLVED_REWARD  # Pendulum is solved when avg reward > -200

    # Training completed - print comprehensive summary using utility function
    final_summary = perf_tracker.get_summary()
    critic1_lr = critic1_optimizer.param_groups[0]["lr"]
    # critic2_lr = critic1_optimizer.param_groups[0]["lr"]
    actor_lr = actor_optimizer.param_groups[0]["lr"]
    print_final_summary(
        solved=solved,
        average_return=average_return,
        target_reward=PENDULUM_SOLVED_REWARD,
        final_summary=final_summary,
        frame_idx=frame_idx,
        current_alpha=(actor_lr, critic1_lr),
        epsilon=0.0,  # Not using epsilon in this implementation
        iter_no=int(iter_no)
    )
