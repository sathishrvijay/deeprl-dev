import gymnasium as gym
import ptan
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time


from functools import partial

from models import agents, pgtr_models
from utils import common_utils, PerformanceTracker, print_training_header, print_final_summary

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
N_ROLLOUT_STEPS = 8 # formal rollout definition;

GAMMA = 0.99  # Slightly lower for Pendulum's shorter episodes
# Separate learning rates for actor and critic
# SAC doesn't usually require schedulers, pretty robust w/ static LRs
CRITIC_LR_START = 1e-4
CRITIC_LR_END = 5e-5
ACTOR_LR_START = 1e-4
ACTOR_LR_END = 5e-5
ENTROPY_LR = 1e-4
LR_DECAY_FRAMES = int(5e7)

# PG related - adjusted for continuous actions
CLIP_GRAD = 0.3   # typical values of clipping the L2 norm is 0.1 to 1.0

# Pendulum success threshold
RNORM_SCALE_FACTOR = 1.0  # reward normalization scale factor
# Automatic entropy temperature tuning
LOG_ALPHA_START = -2.0  # Set a starting value for log_alpha
TARGET_ENTROPY = -3.0  # -action_dim is default but too low in practice; n_actions is 1 for Pendulum
ACTION_PENALTY_COEF = 0.0   # disable to see if it can learn without this
#ACTION_PENALTY_COEF = 0.01  # Prevents actor from going to max torque
PENDULUM_SOLVED_REWARD = -200.0  # Pendulum is solved when avg reward > -200

# Replay buffer related
BATCH_SIZE = 128  # training mini-batch size
REPLAY_BUFFER_SIZE = 10000
BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP = N_ENVS * N_ROLLOUT_STEPS  # 8 x 8


def critic_training_pass(tgt_net, critic_id: int, optimizers, schedulers,
    states_v, actions_v, target_return_v
    ):
    """Pull out this method because it needs to be repeated for each critic"""
    assert critic_id in [1, 2]

    # Q(s, a) <- from collected experience on main net (because of partial SGD)
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
    alpha_optimizer,
    target_entropy,
    log_alpha,
    frame_idx: int,
    iter_no=0,
    reward_norm=True,
    debug=True
    ):
    """DDPG/SAC samples mini-batches from Replay Buffer because it is Off Policy PG
    Returns: Dictionary containing loss components for tracking
    """
    batch = replay_buffer.sample(BATCH_SIZE)
    states_v, actions_v, target_return_v, last_states_v, done_masks_v = \
        common_utils.unpack_batch_ddpg_sac(batch)

    # clamp minimum value for alpha
    entropy_alpha = log_alpha.exp().clamp(min=1e-3).detach()

    # Critic loss:
    # y = r + gamma * [minQT(s', Actor(a'|s')) - log pi(a'|s')]; T <- Target net
    # MSE(y - Q(s, Actor(a|s)))
    # s' <- from collected experience; a' <- from main actor using reparam trick
    target_net = tgt_net.target_model
    with torch.no_grad():
        ls_logproba_actions_v, ls_qv1_v, ls_qv2_v = target_net(last_states_v)
        # Add discounted last state Q value to partial return from trajectory
        ls_q_v = \
            torch.min(ls_qv1_v, ls_qv2_v).squeeze(1) - entropy_alpha * ls_logproba_actions_v.squeeze(-1).data
        ls_q_v[done_masks_v] = 0.0
        target_return_v += (GAMMA ** N_TD_STEPS) * ls_q_v

    critic_loss_v = torch.tensor(0.0)
    for critic_id in [1, 2]:
        critic_loss_v += critic_training_pass(tgt_net, critic_id,
            critic_optimizers, critic_schedulers,
            states_v, actions_v, target_return_v)

    # Actor loss:
    # In SAC, w/ 2 critics, first critic is used for actor loss
    # Sample action from Actor w/ reparametrization trick for differentiability
    actor_optimizer.zero_grad()
    critic_params = list(tgt_net.model.get_critic_parameters(critic_id=1))
    for p in critic_params:
        p.requires_grad_(False)

    logproba_actions_v, qv1_v, qv2_v = tgt_net.model(states_v)
    # Current actions for action penalty
    mean_actions, _ = tgt_net.model.actor(states_v)
    action_penalty = ACTION_PENALTY_COEF * (mean_actions ** 2).mean()
    actor_loss_v = -((torch.min(qv1_v, qv2_v) -
        entropy_alpha * logproba_actions_v) / RNORM_SCALE_FACTOR).squeeze(-1).mean()
    actor_loss_v = actor_loss_v + action_penalty

    actor_loss_v.backward()
    torch.nn.utils.clip_grad_norm_(tgt_net.model.get_actor_parameters(), CLIP_GRAD)
    actor_optimizer.step()
    actor_scheduler.step()

    # Automatic entropy tuning
    alpha_optimizer.zero_grad()
    #  −logproba_action_v is the sample entropy
    # Note: entropy_diff +ve drives alpha aka exploration down; -ve will drive it back up
    entropy_diff = -logproba_actions_v.detach() - target_entropy
    alpha_loss = (log_alpha.exp() * entropy_diff).mean()   # no outer ‘–’
    alpha_loss.backward()
    alpha_optimizer.step()

    # unfreeze critic
    for p in critic_params:
        p.requires_grad_(True)

    # Total loss for logging
    total_loss_v = critic_loss_v + actor_loss_v

    # Smooth blend target net parameters in continuous action for training stability
    # wT = (1 - tau) *wT + tau *w
    # literature calls this param tau; GPT suggests tau <- 0.005
    # ptan uses alpha = 1 - tau
    tgt_net.alpha_sync(alpha=0.995)

    if debug and iter_no % 100 == 0:
        with torch.no_grad():
            actions_dbg = tgt_net.model.sample_action(states_v)
            actions_dbg = actions_dbg.mean(dim=0).cpu().numpy()
            _, qvalues1_dbg, qvalues2_dbg = tgt_net.model(states_v)
            print(f"Action μ={actions_dbg[0]:.3f}, "
                  f"Q1-val range=[{qvalues1_dbg.min():.2f}, {qvalues1_dbg.max():.2f}], "
                  f"Q2-val range=[{qvalues2_dbg.min():.2f}, {qvalues2_dbg.max():.2f}], "
                  f"Target range=[{target_return_v.min():.2f}, {target_return_v.max():.2f}], "
                  f"log_prob mean: {logproba_actions_v.mean().item()}, entropy_alpha: {entropy_alpha.item()}, alpha_loss: {alpha_loss.item()}")

    return {
        'total_loss': total_loss_v.item(),
        'critic_loss': critic_loss_v.item(),
        'actor_loss': actor_loss_v.item(),
        'alpha_loss': alpha_loss.item(),
        'entropy_alpha': entropy_alpha.item(),
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
    # SAC paper uses Adam for all optimizers
    critic1_optimizer = optim.Adam(net.get_critic_parameters(), lr=CRITIC_LR_START, eps=1e-5)
    critic2_optimizer = optim.Adam(net.get_critic_parameters(critic_id=2), lr=CRITIC_LR_START, eps=1e-5)
    actor_optimizer = optim.Adam(net.get_actor_parameters(), lr=ACTOR_LR_START, eps=1e-5)
    # SAC uses a learnable entropy temperature that tunes explore-exploit dynamically
    # Set up learnable log_alpha and optimizer for entropy tuning
    log_alpha = torch.tensor(LOG_ALPHA_START, requires_grad=True)
    alpha_optimizer = optim.Adam([log_alpha], lr=ENTROPY_LR)

    critic1_scheduler = optim.lr_scheduler.LinearLR(
        critic1_optimizer, start_factor=1.0, end_factor=CRITIC_LR_END / CRITIC_LR_START,
        total_iters=LR_DECAY_FRAMES // BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP)
    critic2_scheduler = optim.lr_scheduler.LinearLR(
        critic2_optimizer, start_factor=1.0, end_factor=CRITIC_LR_END / CRITIC_LR_START,
        total_iters=LR_DECAY_FRAMES // BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP)
    actor_scheduler = optim.lr_scheduler.LinearLR(
        actor_optimizer, start_factor=1.0, end_factor=ACTOR_LR_END / ACTOR_LR_START,
        total_iters=LR_DECAY_FRAMES // BUF_ENTRIES_POPULATED_PER_TRAIN_LOOP)

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
            alpha_optimizer, TARGET_ENTROPY, log_alpha,
            frame_idx,
            iter_no
        )
        training_time = time.time() - iter_start_time

        # Print periodic performance summary every 500 iterations
        if iter_no % 2000 == 0:
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
                f"alpha_loss={loss_dict['alpha_loss']:.5f}, "
                f"entropy_alpha={loss_dict['entropy_alpha']:.5f}"
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
