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

"""This is the implementation of DDPG with Pendulum-v1 RL using the PTAN wrapper libraries.
TODOs 07/22/25:
- OU noise implementation for exploration in DDPG model
- soft target network parameters update
- update the loss function in core_training_loop()
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
CLIP_GRAD = 0.3   # typical values of clipping the L2 norm is 0.1 to 1.0

# Pendulum success threshold
PENDULUM_SOLVED_REWARD = -200.0  # Pendulum is solved when avg reward > -200

# Replay buffer related
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP = 50


def unpack_batch_ddqn(batch: tt.List[ptan.experience.ExperienceFirstLast],
    net: nn.Module,
    n_steps: int
    ):
    """
    Note: unpacking batch is different for DDQN because Q(s', a') and therefore
    target return is only available online during training
    Note: Since, in general, an experience sub-trajectory can be n-steps,
    the terminology used here is last state instead of next state.
    Additionally, reward is equal to the cumulative discounted rewards from
    intermediate steps. All of this is subsumed within ptan.experience.ExperienceFirstLast
    Note: DDPG only emits the mean action vector unlike other CA A2C algorithms
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

    # Note: for DDPG, Q(s', a') cannot be computed before a' is known from Actor
    # Hence we return the partial return (aka rewards_v) until this point
    return states_v, actions_v, rewards_v, last_states_v, done_masks_v


def core_training_loop(
    tgt_net: ptan.agent.TargetNet,
    replay_buffer: ptan.experience.ExperienceReplayBuffer,
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    frame_idx: int,
    iter_no=None,
    reward_norm=True,
    debug=True
    ):
    """DDPG samples mini-batches from Replay Buffer because it is Off Policy PG
    Returns: Dictionary containing loss components for tracking
    """
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    batch = replay_buffer.sample(BATCH_SIZE)
    states_v, actions_v, target_return_v, last_states_v, done_masks_v = \
        unpack_batch_ddqn(batch, tgt_net.target_model, N_TD_STEPS)

    # Note: Separate loss computation and backpropagation for Critic and Actor
    # Critic loss: MSE(r + gamma * QT(s', muT(s)) - Q(s, a)); T <- Target net
    # s' <- from collected experience
    target_net = tgt_net.target_model
    ls_actions_mu_v, ls_qvalues_v = target_net(last_states_v)
    ls_qvalues_v = ls_qvalues_v.squeeze(1).data
    ls_qvalues_v *= GAMMA ** N_TD_STEPS
    # zero out the terminated episodes
    ls_qvalues_v[done_masks_v] = 0.0
    target_return_v += ls_qvalues_v

    # Q(s, a) <- from collected experience on main net (because of partial SGD)
    critic_net = tgt_net.model.get_critic_net()
    qvalues_v = critic_net(states_v, actions_v)
    if reward_norm:
        # Pendulum rewards are in [-16, 0] range, so normalize differently than LunarLander
        qvalues_v = qvalues_v / 16.0
        target_return_v = target_return_v / 16.0
    critic_loss_v = nn.functional.mse_loss(qvalues_v.squeeze(-1), target_return_v)

    # Critic loss (only affects critic network)
    critic_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(net.get_critic_parameters(), CLIP_GRAD)
    critic_optimizer.step()
    critic_scheduler.step()

    # Actor loss: Invert sign and propagate back from Critic
    actor_net = tgt_net.model.get_actor_net()
    current_actions_v = actor_net(states_v)
    actor_loss_v = -critic_net(states_v, current_actions_v)
    actor_loss_v = actor_loss_v.mean()
    actor_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(net.get_actor_parameters(), CLIP_GRAD)
    actor_optimizer.step()
    actor_scheduler.step()

    # Total loss for logging (not used for backprop)
    total_loss_v = critic_loss_v + actor_loss_v

    # Smooth blend target net parameters in continuous action for training stability
    # ptan uses alpha = 1 - tau
    # literature calls this param tau; GPT suggests tau <- 0.005
    # wT = (1 - tau) *wT + tau *w
    tgt_net.alpha_sync(alpha=1 - 5e-3)

    # Debug - check policy statistics
    if debug and iter_no % 100 == 0:
        with torch.no_grad():
            current_actions_v = current_actions_v.mean(dim=0).cpu().numpy()
            print(f"Action μ={current_actions_v[0]:.3f}")

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
    net = pgtr_models.DDPG(
        n_states, n_actions,
        CRITIC_HIDDEN1_DIM, CRITIC_HIDDEN2_DIM,
        ACTOR_HIDDEN1_DIM, ACTOR_HIDDEN2_DIM
    )
    tgt_net = ptan.agent.TargetNet(net)

    # Setup the agent, action policy, experience generation and experience buffer
    agent = agents.AgentDDPG(net)
    exp_source = ptan.experience.VectorExperienceSourceFirstLast(vector_env, agent, gamma=GAMMA,
        steps_count=N_TD_STEPS)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_BUFFER_SIZE)

    # don't need a vectorized trial env
    test_env = gym.make(RL_ENV)
    test_agent = agents.AgentDDPG(net, deterministic=True)

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
        'clip_grad': CLIP_GRAD,
        'solved_threshold': PENDULUM_SOLVED_REWARD
    }
    print_training_header(RL_ENV, network_config, hyperparams)

    exp_iterator = iter(exp_source)
    frame_idx = 0  # Track total frames of experience generated
    # warm start for replay buffer
    replay_buffer.populate(REPLAY_BUFFER_SIZE)
    # breakpoint()
    while not solved:
        iter_no += 1.0
        iter_start_time = time.time()
        # Collect experiences every iteration
        replay_buffer.populate(BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP)
        frame_idx += BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP

        # Training step with separate optimizers
        loss_dict = core_training_loop(
            tgt_net, replay_buffer, actor_optimizer, critic_optimizer,
            actor_scheduler, critic_scheduler, frame_idx, iter_no
        )
        training_time = time.time() - iter_start_time

        # TODO - add soft sync
        #tgt_net.sync()

        # Print periodic performance summary every 500 iterations
        if iter_no % 10000 == 0:
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
